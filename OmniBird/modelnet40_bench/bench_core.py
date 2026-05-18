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
from hrr_patch_core import HRRPatchifierTime


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

# Primary: Hugging Face mirror (more reliable from proxied clusters).
# Fallback: original Stanford host.
MODELNET40_URLS = [
    "https://huggingface.co/datasets/Msun/modelnet40/resolve/main/modelnet40_ply_hdf5_2048.zip?download=true",
    "https://shapenet.cs.stanford.edu/media/modelnet40_ply_hdf5_2048.zip",
]
MODELNET40_URL = MODELNET40_URLS[0]   # back-compat alias
MODELNET40_DIR_DEFAULT = os.environ.get(
    "MODELNET40_DIR",
    os.path.expanduser("~/data/modelnet40_ply_hdf5_2048"),
)


def _print_manual_instructions(zip_path: str, marker: str, last_err: Exception | None = None):
    msg = [
        "",
        "=" * 78,
        "  ModelNet40 auto-download failed (likely a proxy/firewall issue).",
    ]
    if last_err is not None:
        msg.append(f"  Last error: {last_err}")
    msg += [
        "",
        "  MANUAL DOWNLOAD INSTRUCTIONS",
        "  ----------------------------",
        "  1. From any machine with web access, download the zip (~400MB):",
        f"       wget -O modelnet40_ply_hdf5_2048.zip \\",
        f"         \"{MODELNET40_URLS[0]}\"",
        "     (HF mirror; the Stanford origin tends to 503 from clusters)",
        "",
        "  2. Place the zip at:",
        f"       {zip_path}",
        "     and re-run this cell — it will detect and extract automatically.",
        "",
        "  3. Or, if you already have the extracted folder somewhere, set:",
        "       export MODELNET40_DIR=/path/to/modelnet40_ply_hdf5_2048",
        f"     (the target must contain  shape_names.txt  → currently missing: {marker})",
        "",
        "=" * 78,
        "",
    ]
    print("\n".join(msg))


def download_modelnet40(target_dir: str = MODELNET40_DIR_DEFAULT,
                        urls: list[str] | None = None) -> str:
    """Download + unzip the PointNet++ preprocessed ModelNet40 (~400MB).

    Robustness:
      - Honors `MODELNET40_DIR` env var (caller can point to a pre-extracted dir).
      - Detects a manually-placed zip at `<parent>/modelnet40_ply_hdf5_2048.zip`
        and extracts it without re-downloading.
      - Tries a list of mirror URLs and falls back to clear manual instructions
        if all of them fail (e.g. behind a proxy that blocks the host).
    """
    marker = os.path.join(target_dir, "shape_names.txt")
    if os.path.exists(marker):
        print(f"  ModelNet40 already at {target_dir}")
        return target_dir

    parent = os.path.dirname(target_dir)
    os.makedirs(parent, exist_ok=True)
    zip_path = os.path.join(parent, "modelnet40_ply_hdf5_2048.zip")

    # Detect a manually-placed zip — skip download entirely
    if os.path.exists(zip_path):
        print(f"  found local zip at {zip_path} ({os.path.getsize(zip_path) / 1e6:.1f} MB); extracting")
        try:
            with ZipFile(zip_path, "r") as z:
                z.extractall(parent)
        except Exception as e:
            _print_manual_instructions(zip_path, marker, last_err=e)
            raise RuntimeError(f"failed to extract {zip_path}: {e}")
        if os.path.exists(marker):
            print(f"  ready at {target_dir}")
            return target_dir
        else:
            _print_manual_instructions(zip_path, marker)
            raise RuntimeError(f"extracted but {marker} is missing")

    # Try downloading from each candidate URL (HF mirror first, Stanford fallback)
    urls = urls or list(MODELNET40_URLS)
    last_err = None
    for url in urls:
        try:
            print(f"  downloading from {url}  (~400MB)")
            urlretrieve(url, zip_path)
            print(f"  extracting to {parent}")
            with ZipFile(zip_path, "r") as z:
                z.extractall(parent)
            try: os.remove(zip_path)
            except OSError: pass
            if os.path.exists(marker):
                print(f"  ready at {target_dir}")
                return target_dir
        except Exception as e:
            last_err = e
            print(f"  download from {url} failed: {type(e).__name__}: {e}")
            # Remove partial zip so a subsequent local-zip path isn't fooled
            if os.path.exists(zip_path):
                try: os.remove(zip_path)
                except OSError: pass
            continue

    _print_manual_instructions(zip_path, marker, last_err=last_err)
    raise RuntimeError(
        f"all download attempts failed; last error: {last_err}. "
        f"See manual download instructions above."
    )


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
