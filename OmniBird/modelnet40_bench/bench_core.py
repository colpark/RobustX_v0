"""ModelNet40 benchmark — shared modules for comparing patch aggregators.

Drop-in comparison: PointNet (mini-PointNet + max-pool) vs RoPE/HRR
(NUDFT + sum) patch aggregator. Identical encoder backbone, only the
aggregator differs.

We import the existing patchifiers from the standalone folder so this is
a pure benchmark — no duplicated implementations.
"""
from __future__ import annotations
import os, sys, time
from urllib.request import urlretrieve
from zipfile import ZipFile

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# Reuse standalone implementations
_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.normpath(os.path.join(_HERE, "..", "standalone")))

from vit_fps_core import (
    NerfPosEnc, FeedForward,
    MultiHeadSelfAttention, TransformerBlock,
    Patchifier as PointNetPatchifier,
    farthest_point_sample, knn_indices,
    short_params, save_atomic,
)
from rope_patch_core import RoPEPatchifier


# ===========================================================================
# Encoder & classifier — IDENTICAL backbone for both aggregators
# ===========================================================================

class FlexibleViTEncoder(nn.Module):
    """Vanilla ViT encoder; only the patchifier varies between methods.

    Both PointNet and RoPE aggregator outputs feed into the SAME:
      - NeRF γ(centroid) added to patch tokens (absolute pos info)
      - Standard MHA transformer blocks (no RoPE in attention)
      - Final LayerNorm
    """
    def __init__(self, patchifier, coord_dim: int, d_model: int,
                 n_layers: int = 6, n_heads: int = 6, dim_head: int = 32,
                 ffn_mult: int = 4, n_freqs: int = 8):
        super().__init__()
        self.patchifier = patchifier
        self.pos_enc = NerfPosEnc(coord_dim, n_freqs=n_freqs, include_input=True)
        self.pos_proj = nn.Linear(self.pos_enc.out_dim, d_model)
        self.blocks = nn.ModuleList([
            TransformerBlock(d_model, n_heads=n_heads, dim_head=dim_head,
                             ffn_mult=ffn_mult)
            for _ in range(n_layers)
        ])
        self.norm = nn.LayerNorm(d_model)

    def forward(self, patch_events, patch_centroids):
        x = self.patchifier(patch_events, patch_centroids)
        x = x + self.pos_proj(self.pos_enc(patch_centroids))
        for blk in self.blocks:
            x = blk(x)
        return self.norm(x)


class ModelNet40Classifier(nn.Module):
    """Mean-pool over patch tokens → MLP head → 40 logits."""
    def __init__(self, encoder, d_model: int, n_classes: int = 40,
                 dropout: float = 0.3):
        super().__init__()
        self.encoder = encoder
        self.head = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Dropout(dropout),
            nn.Linear(d_model, d_model // 2), nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, n_classes),
        )

    def forward(self, patch_events, patch_centroids):
        feat = self.encoder(patch_events, patch_centroids)
        pooled = feat.mean(dim=1)
        return self.head(pooled)


# ===========================================================================
# ModelNet40 download + load (HDF5 from PointNet++ preprocessed release)
# ===========================================================================

MODELNET40_URL = "https://shapenet.cs.stanford.edu/media/modelnet40_ply_hdf5_2048.zip"
MODELNET40_DIR_DEFAULT = os.path.expanduser("~/data/modelnet40_ply_hdf5_2048")


def download_modelnet40(target_dir: str = MODELNET40_DIR_DEFAULT) -> str:
    """Download + unzip the PointNet++ preprocessed ModelNet40 (~400MB).
    Returns the data directory."""
    marker = os.path.join(target_dir, "shape_names.txt")
    if os.path.exists(marker):
        print(f"  ModelNet40 already at {target_dir}")
        return target_dir
    parent = os.path.dirname(target_dir)
    os.makedirs(parent, exist_ok=True)
    zip_path = os.path.join(parent, "modelnet40_ply_hdf5_2048.zip")
    print(f"  downloading ModelNet40 (~400MB) from {MODELNET40_URL}")
    urlretrieve(MODELNET40_URL, zip_path)
    print(f"  extracting to {parent}")
    with ZipFile(zip_path, "r") as z:
        z.extractall(parent)
    os.remove(zip_path)
    assert os.path.exists(marker), f"download/extract failed; missing {marker}"
    print(f"  ready at {target_dir}")
    return target_dir


def load_modelnet40(root: str = MODELNET40_DIR_DEFAULT, train: bool = True):
    """Load ModelNet40 from the preprocessed HDF5 files.
    Returns (points, labels) where points is (N, 2048, 3) and labels is (N,)."""
    import h5py
    download_modelnet40(root)
    list_file = os.path.join(root, "train_files.txt" if train else "test_files.txt")
    with open(list_file) as f:
        files = [line.strip() for line in f if line.strip()]
    all_pts, all_lbl = [], []
    for fpath in files:
        # files list contains absolute paths from the original release;
        # take just the basename and look it up locally.
        h5_path = os.path.join(root, os.path.basename(fpath))
        with h5py.File(h5_path, "r") as h5:
            all_pts.append(np.array(h5["data"], dtype=np.float32))
            all_lbl.append(np.array(h5["label"]).squeeze().astype(np.int64))
    return np.concatenate(all_pts), np.concatenate(all_lbl)


# ===========================================================================
# Cached FPS+KNN per cloud
# ===========================================================================

def precompute_fps_knn_modelnet(
    points: np.ndarray,             # (N, n_input, 3) per-cloud points
    n_patches: int,
    k_neigh: int,
    seed: int = 42,
    cache_dir: str = "./cache_fps_knn_modelnet",
    tag: str = "",
):
    """Per-cloud FPS centroids + K-NN groups, cached to disk."""
    os.makedirs(cache_dir, exist_ok=True)
    N, n_input, D = points.shape
    cache_key = (
        f"mn40_{tag}_N{N}_pts{n_input}_patches{n_patches}_knn{k_neigh}_seed{seed}.npz"
    )
    cache_path = os.path.join(cache_dir, cache_key)
    if os.path.exists(cache_path):
        try:
            data = np.load(cache_path)
            cen = data["centroid_idx"]
            nbr = data["nbr_idx"]
            if cen.shape == (N, n_patches) and nbr.shape == (N, n_patches, k_neigh):
                print(f"  [fps_knn cache HIT] {cache_path}")
                return cen, nbr
        except Exception as e:
            print(f"  [fps_knn cache READ FAIL] {e}; recomputing")
    print(f"  [fps_knn cache MISS] computing FPS+KNN for {N} clouds...")
    t0 = time.time()
    torch.manual_seed(seed)
    cen_all = np.zeros((N, n_patches), dtype=np.int64)
    nbr_all = np.zeros((N, n_patches, k_neigh), dtype=np.int64)
    pts_t = torch.from_numpy(points).float()
    for i in range(N):
        pc = pts_t[i].unsqueeze(0)                       # (1, n_input, 3)
        cen_idx = farthest_point_sample(pc, n_patches).squeeze(0)
        cen_coords = pc[0, cen_idx]
        nbr_idx = knn_indices(cen_coords.unsqueeze(0), pc, k_neigh).squeeze(0)
        cen_all[i] = cen_idx.numpy()
        nbr_all[i] = nbr_idx.numpy()
        if (i + 1) % 1000 == 0:
            print(f"    {i+1}/{N}  ({time.time()-t0:.1f}s)")
    elapsed = time.time() - t0
    print(f"  computed in {elapsed:.1f}s; saving to {cache_path}")
    tmp = cache_path + ".tmp"
    with open(tmp, "wb") as f:
        np.savez(f, centroid_idx=cen_all, nbr_idx=nbr_all)
    os.replace(tmp, cache_path)
    stale = cache_path + ".tmp.npz"
    if os.path.exists(stale):
        try: os.remove(stale)
        except OSError: pass
    return cen_all, nbr_all


# ===========================================================================
# Augmentations (standard PointNet++ recipe)
# ===========================================================================

def augment_pointcloud(points: np.ndarray) -> np.ndarray:
    """Rotation around z-axis + scale + jitter. Operates on (n_input, 3)."""
    pts = points.copy()
    # Rotate around z (canonical "up" axis in ModelNet40)
    theta = np.random.uniform(0.0, 2.0 * np.pi)
    c, s = np.cos(theta), np.sin(theta)
    R = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]], dtype=pts.dtype)
    pts = pts @ R.T
    # Anisotropic scale + small per-point jitter
    scale = np.random.uniform(0.8, 1.25)
    pts = pts * scale
    pts = pts + np.random.normal(0.0, 0.01, pts.shape).astype(pts.dtype)
    return pts


def normalize_to_unit_sphere(points: np.ndarray) -> np.ndarray:
    """Center + normalize each cloud to fit in the unit sphere."""
    centroid = points.mean(axis=-2, keepdims=True)
    pts = points - centroid
    max_norm = np.linalg.norm(pts, axis=-1).max(axis=-1, keepdims=True)
    return pts / np.expand_dims(max_norm, -1)
