"""ViT-FPS JEPA — standalone, copy-paste portable.

FPS-based patch construction with NeRF positional encoding and a dense
ViT-style encoder. Closest sparse-input analogue of canonical ViT-JEPA:

  • patches are FPS centroids + K-NN aggregations of the sparse pool
  • each patch token carries BOTH content (mini-PointNet over its K members)
    AND position (NeRF-encoded centroid coord)
  • encoder is a standard dense ViT (no BigBird — patches are few)
  • context / target are different SUBSETS of the same FPS centroid set,
    so both encoders work with the same patch definitions

The central design point: FPS is run ONCE per sample over the whole pool;
context and target encoders then operate on disjoint subsets of those
fixed patches. This eliminates the "context and target see different
patches" pitfall that plagued earlier xattn iterations.
"""
from __future__ import annotations
from typing import Optional, Iterator
import math
import os

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


# ===========================================================================
# NeRF positional encoding (matches the canonical formulation)
# ===========================================================================

class NerfPosEnc(nn.Module):
    """γ(x) = [x, sin(2⁰ π x), cos(2⁰ π x), …, sin(2^{L-1} π x), cos(2^{L-1} π x)].

    Output dim per input dim: (1 if include_input else 0) + 2 * n_freqs.
    """
    def __init__(self, coord_dim: int, n_freqs: int = 10,
                 include_input: bool = True, log_sampling: bool = True):
        super().__init__()
        self.coord_dim = coord_dim
        self.n_freqs = n_freqs
        self.include_input = include_input
        if log_sampling:
            freqs = 2.0 ** torch.arange(n_freqs, dtype=torch.float32) * math.pi
        else:
            freqs = torch.linspace(1.0, 2.0 ** (n_freqs - 1), n_freqs) * math.pi
        self.register_buffer("freqs", freqs)
        per_dim = (1 if include_input else 0) + 2 * n_freqs
        self.out_dim = coord_dim * per_dim

    def forward(self, coords: torch.Tensor) -> torch.Tensor:
        # coords: (..., D) → (..., D · per_dim)
        x = coords.unsqueeze(-1) * self.freqs              # (..., D, L)
        sin = torch.sin(x)
        cos = torch.cos(x)
        stacked = torch.stack([sin, cos], dim=-1)          # (..., D, L, 2)
        stacked = stacked.flatten(-3)                       # (..., D · L · 2)
        if self.include_input:
            stacked = torch.cat([coords.reshape(*coords.shape[:-1], -1), stacked], dim=-1)
        return stacked


# ===========================================================================
# Farthest Point Sampling (FPS) + K-NN
# ===========================================================================

@torch.no_grad()
def farthest_point_sample(points: torch.Tensor, n_samples: int,
                            mask: Optional[torch.Tensor] = None) -> torch.Tensor:
    """FPS over (B, N, D). Optional `mask` (B, N) True at padded positions —
    padded points are never selected. Returns indices (B, n_samples) long."""
    B, N, D = points.shape
    device = points.device
    centroid_idx = torch.zeros(B, n_samples, dtype=torch.long, device=device)
    distances = torch.full((B, N), 1e10, device=device)
    if mask is not None:
        distances = distances.masked_fill(mask, 0.0)        # padded points get dist=0 so never selected
    # First centroid: random real index
    if mask is None:
        first = torch.randint(0, N, (B,), device=device)
    else:
        real_counts = (~mask).sum(dim=1).clamp(min=1)
        rand01 = torch.rand(B, device=device)
        first_rank = (rand01 * real_counts.float()).long().clamp(max=real_counts - 1)
        first = torch.zeros(B, dtype=torch.long, device=device)
        for b in range(B):
            real_idx_b = (~mask[b]).nonzero(as_tuple=False).squeeze(-1)
            first[b] = real_idx_b[first_rank[b]]
    centroid_idx[:, 0] = first
    farthest = first
    for i in range(1, n_samples):
        cur = points[torch.arange(B, device=device), farthest, :].unsqueeze(1)   # (B, 1, D)
        d2 = ((points - cur) ** 2).sum(-1)                                          # (B, N)
        if mask is not None:
            d2 = d2.masked_fill(mask, 0.0)
        distances = torch.minimum(distances, d2)
        farthest = distances.argmax(dim=1)
        centroid_idx[:, i] = farthest
    return centroid_idx


@torch.no_grad()
def knn_indices(centroids: torch.Tensor, points: torch.Tensor, k: int,
                 mask: Optional[torch.Tensor] = None) -> torch.Tensor:
    """K nearest neighbors of each centroid in `points`.
    centroids: (B, M, D), points: (B, N, D), mask: (B, N) True at padded.
    Returns (B, M, k) indices into the second dim of `points`."""
    d2 = ((centroids.unsqueeze(2) - points.unsqueeze(1)) ** 2).sum(-1)              # (B, M, N)
    if mask is not None:
        d2 = d2.masked_fill(mask.unsqueeze(1), float("inf"))
    _, idx = d2.topk(k, dim=-1, largest=False)
    return idx


# ===========================================================================
# Mini-PointNet patchifier
# ===========================================================================

class Patchifier(nn.Module):
    """Mini-PointNet: aggregates K (relative-coord + signal) tokens per patch
    into a single content-bearing patch feature."""

    def __init__(self, signal_dim: int, coord_dim: int, d_model: int,
                 hidden: int = 128):
        super().__init__()
        in_dim = signal_dim + coord_dim
        self.mlp_event = nn.Sequential(
            nn.Linear(in_dim, hidden), nn.GELU(),
            nn.Linear(hidden, d_model), nn.GELU(),
        )
        self.mlp_patch = nn.Sequential(
            nn.Linear(d_model, d_model), nn.GELU(),
            nn.Linear(d_model, d_model),
        )

    def forward(self, patch_events: torch.Tensor, patch_centroids: torch.Tensor,
                event_kpm: Optional[torch.Tensor] = None) -> torch.Tensor:
        """patch_events: (B, P, K, coord_dim + signal_dim) — RAW coord + signal
        patch_centroids: (B, P, coord_dim)
        event_kpm: (B, P, K) True at padded.
        Returns: (B, P, d_model)
        """
        coord_dim = patch_centroids.size(-1)
        coords = patch_events[..., :coord_dim]
        signal = patch_events[..., coord_dim:]
        rel    = coords - patch_centroids.unsqueeze(2)
        feat   = self.mlp_event(torch.cat([rel, signal], dim=-1))     # (B, P, K, d_model)
        if event_kpm is not None:
            feat = feat.masked_fill(event_kpm.unsqueeze(-1), float("-inf"))
        pooled = feat.max(dim=2).values
        pooled = torch.where(torch.isinf(pooled), torch.zeros_like(pooled), pooled)
        return self.mlp_patch(pooled)


# ===========================================================================
# Dense ViT encoder
# ===========================================================================

class MultiHeadSelfAttention(nn.Module):
    def __init__(self, dim: int, n_heads: int = 8, dim_head: int = 32, bias_qkv: bool = False):
        super().__init__()
        inner = n_heads * dim_head
        self.n_heads = n_heads
        self.dim_head = dim_head
        self.scale = dim_head ** -0.5
        self.to_qkv = nn.Linear(dim, inner * 3, bias=bias_qkv)
        self.to_out = nn.Linear(inner, dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, N, _ = x.shape
        qkv = self.to_qkv(x).reshape(B, N, 3, self.n_heads, self.dim_head)
        q, k, v = qkv.permute(2, 0, 3, 1, 4).unbind(0)                     # each (B, H, N, Dh)
        scores = torch.einsum("bhnd,bhmd->bhnm", q, k) * self.scale
        attn = F.softmax(scores, dim=-1)
        out = torch.einsum("bhnm,bhmd->bhnd", attn, v).transpose(1, 2).reshape(B, N, -1)
        return self.to_out(out)


class FeedForward(nn.Module):
    def __init__(self, dim: int, mult: int = 4):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(dim, dim * mult), nn.GELU(), nn.Linear(dim * mult, dim))
    def forward(self, x): return self.net(x)


class TransformerBlock(nn.Module):
    def __init__(self, dim: int, n_heads: int = 8, dim_head: int = 32, ffn_mult: int = 4):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn  = MultiHeadSelfAttention(dim, n_heads=n_heads, dim_head=dim_head)
        self.norm2 = nn.LayerNorm(dim)
        self.ffn   = FeedForward(dim, mult=ffn_mult)

    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        return x + self.ffn(self.norm2(x))


class ViTPatchEncoder(nn.Module):
    """Patchifier + NeRF pos + dense ViT encoder."""

    def __init__(self, signal_dim: int, coord_dim: int, d_model: int = 256,
                 n_layers: int = 6, n_heads: int = 8, dim_head: int = 32,
                 ffn_mult: int = 4, n_freqs: int = 10):
        super().__init__()
        self.patchify   = Patchifier(signal_dim, coord_dim, d_model)
        self.pos_enc    = NerfPosEnc(coord_dim, n_freqs=n_freqs, include_input=True)
        self.pos_proj   = nn.Linear(self.pos_enc.out_dim, d_model)
        self.blocks     = nn.ModuleList([
            TransformerBlock(d_model, n_heads=n_heads, dim_head=dim_head, ffn_mult=ffn_mult)
            for _ in range(n_layers)
        ])
        self.norm = nn.LayerNorm(d_model)

    def forward(self, patch_events: torch.Tensor, patch_centroids: torch.Tensor,
                event_kpm: Optional[torch.Tensor] = None) -> torch.Tensor:
        tok = self.patchify(patch_events, patch_centroids, event_kpm=event_kpm)
        pos = self.pos_proj(self.pos_enc(patch_centroids))
        x = tok + pos
        for blk in self.blocks: x = blk(x)
        return self.norm(x)


# ===========================================================================
# Predictor: dense Transformer with mask-tokens at target patch positions
# ===========================================================================

class PredictorBlock(nn.Module):
    def __init__(self, dim, n_heads=6, dim_head=32, ffn_mult=4):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn  = MultiHeadSelfAttention(dim, n_heads=n_heads, dim_head=dim_head)
        self.norm2 = nn.LayerNorm(dim)
        self.ffn   = FeedForward(dim, mult=ffn_mult)
    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        return x + self.ffn(self.norm2(x))


class ViTFPSPredictor(nn.Module):
    """Dense Transformer over [ctx_tokens ‖ mask_tokens@target_centroids].

    Reads off the predicted target features at the target positions.
    pos_symmetric=True: ctx tokens also receive the NeRF pos of their
    centroids before predictor runs.
    """

    def __init__(self, d_model: int = 256, d_pred: int = 192, n_layers: int = 4,
                 n_heads: int = 6, dim_head: int = 32, coord_dim: int = 2,
                 n_freqs: int = 10, ffn_mult: int = 4, pos_symmetric: bool = True):
        super().__init__()
        self.proj_in    = nn.Linear(d_model, d_pred)
        self.pos_enc    = NerfPosEnc(coord_dim, n_freqs=n_freqs, include_input=True)
        self.proj_pos   = nn.Linear(self.pos_enc.out_dim, d_pred)
        self.mask_token = nn.Parameter(torch.zeros(1, 1, d_pred))
        nn.init.trunc_normal_(self.mask_token, std=0.02)
        self.blocks = nn.ModuleList([
            PredictorBlock(d_pred, n_heads=n_heads, dim_head=dim_head, ffn_mult=ffn_mult)
            for _ in range(n_layers)
        ])
        self.norm     = nn.LayerNorm(d_pred)
        self.proj_out = nn.Linear(d_pred, d_model)
        self.pos_symmetric = pos_symmetric

    def forward(self, ctx_feat: torch.Tensor, target_coords: torch.Tensor,
                ctx_coords: Optional[torch.Tensor] = None) -> torch.Tensor:
        B, K_ctx, _ = ctx_feat.shape
        ctx_tok = self.proj_in(ctx_feat)
        if self.pos_symmetric:
            assert ctx_coords is not None
            ctx_tok = ctx_tok + self.proj_pos(self.pos_enc(ctx_coords))
        tgt_tok = self.proj_pos(self.pos_enc(target_coords)) + self.mask_token
        x = torch.cat([ctx_tok, tgt_tok], dim=1)
        for blk in self.blocks: x = blk(x)
        x = self.norm(x[:, K_ctx:])
        return self.proj_out(x)


# ===========================================================================
# Target centering + EMA + loss + helpers
# ===========================================================================

class TargetCenter(nn.Module):
    """DINO-style EMA of per-feature batch mean. Subtract before LN."""
    def __init__(self, embed_dim: int, momentum: float = 0.9):
        super().__init__()
        self.momentum = momentum
        self.register_buffer("center", torch.zeros(1, 1, embed_dim))
    @torch.no_grad()
    def update(self, h: torch.Tensor):
        batch_center = h.mean(dim=(0, 1), keepdim=True)
        self.center.mul_(self.momentum).add_(batch_center, alpha=1.0 - self.momentum)
    def forward(self, h: torch.Tensor) -> torch.Tensor:
        return h - self.center


def ema_update(target_module: nn.Module, online_module: nn.Module, m: float):
    for p_q, p_k in zip(online_module.parameters(), target_module.parameters()):
        p_k.data.mul_(m).add_((1.0 - m) * p_q.detach())


def make_momentum_schedule(start: float, end: float, total_steps: int) -> Iterator[float]:
    for i in range(total_steps + 1):
        yield start + i * (end - start) / total_steps


def jepa_loss(h_pred: torch.Tensor, h_tgt: torch.Tensor,
              loss_type: str = "smooth_l1") -> torch.Tensor:
    if loss_type == "cosine":
        return (1 - F.cosine_similarity(h_pred, h_tgt, dim=-1)).mean()
    if loss_type == "mse":
        return F.mse_loss(h_pred, h_tgt)
    return F.smooth_l1_loss(h_pred, h_tgt)


def short_params(m: nn.Module) -> str:
    n = sum(p.numel() for p in m.parameters())
    return f"{n/1e6:.2f} M params"


def save_atomic(state, path):
    tmp = path + ".tmp"
    torch.save(state, tmp)
    os.replace(tmp, path)


# ===========================================================================
# Cached FPS+KNN precomputation — shared by all FPS-based notebooks
# ===========================================================================

def precompute_fps_knn_cached(
    coords_all: torch.Tensor,
    pool_idx: np.ndarray,
    n_patches: int,
    k_neigh: int,
    seed: int = 42,
    cache_dir: str = "./cache_fps_knn",
    tag: str = "",
):
    """Precompute FPS centroids + K-NN groups for every sample. Cached to disk.

    The cache key is built from `(N_samples, pool_size, n_patches, k_neigh,
    seed, tag)`, so any config change generates a fresh file. Cache files are
    `.npz` archives containing `centroid_idx_all` and `nbr_idx_all`.

    Args
    ----
    coords_all : (N_pix, D) tensor with all per-pixel coords (e.g. 32x32x2).
    pool_idx   : (N_samples, K_pool) int64 — which pixels are in each sample's pool.
    n_patches  : FPS centroid count.
    k_neigh    : KNN size per patch.
    seed       : torch RNG seed (FPS picks the first centroid randomly).
    cache_dir  : where to write/read the cache.
    tag        : extra string in the cache filename (e.g. "train" / "test").

    Returns (centroid_idx_all, nbr_idx_all) as np.int64 arrays.
    """
    os.makedirs(cache_dir, exist_ok=True)
    N_samples, K_pool = pool_idx.shape
    cache_key = (
        f"fpsknn_{tag}_N{N_samples}_pool{K_pool}_patches{n_patches}"
        f"_knn{k_neigh}_seed{seed}.npz"
    )
    cache_path = os.path.join(cache_dir, cache_key)

    if os.path.exists(cache_path):
        try:
            data = np.load(cache_path)
            centroid_idx_all = data["centroid_idx_all"]
            nbr_idx_all = data["nbr_idx_all"]
            if (centroid_idx_all.shape == (N_samples, n_patches)
                and nbr_idx_all.shape == (N_samples, n_patches, k_neigh)):
                print(f"  [fps_knn cache HIT] loaded {cache_path}")
                return centroid_idx_all, nbr_idx_all
            print(f"  [fps_knn cache STALE] shape mismatch, recomputing")
        except Exception as e:
            print(f"  [fps_knn cache READ FAIL] {e}; recomputing")

    print(f"  [fps_knn cache MISS] computing FPS+KNN for {N_samples} samples...")
    import time as _time
    t0 = _time.time()
    torch.manual_seed(seed)
    centroid_idx_all = np.zeros((N_samples, n_patches), dtype=np.int64)
    nbr_idx_all = np.zeros((N_samples, n_patches, k_neigh), dtype=np.int64)
    for i in range(N_samples):
        pc = coords_all[pool_idx[i]].unsqueeze(0)
        cen_idx = farthest_point_sample(pc, n_patches).squeeze(0)
        cen_coords = pc[0, cen_idx]
        nbrs = knn_indices(cen_coords.unsqueeze(0), pc, k_neigh).squeeze(0)
        centroid_idx_all[i] = cen_idx.numpy()
        nbr_idx_all[i] = nbrs.numpy()
    elapsed = _time.time() - t0
    print(f"  computed in {elapsed:.1f}s; saving to {cache_path}")
    # Atomic write: np.savez appends ".npz" to STRING paths, so we pass a
    # file handle instead (handles are written to verbatim). Then os.replace
    # the temp file onto the final cache path.
    tmp = cache_path + ".tmp"
    with open(tmp, "wb") as f:
        np.savez(f, centroid_idx_all=centroid_idx_all, nbr_idx_all=nbr_idx_all)
    os.replace(tmp, cache_path)
    # Best-effort cleanup of any stale leftover from an earlier broken write
    # (when savez was passed a string and auto-appended ".npz")
    stale = cache_path + ".tmp.npz"
    if os.path.exists(stale):
        try:
            os.remove(stale)
        except OSError:
            pass
    return centroid_idx_all, nbr_idx_all
