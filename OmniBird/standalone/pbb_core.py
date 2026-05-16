"""PBB-style JEPA — standalone, copy-paste portable.

A consolidated, self-contained implementation of the PointBigBird JEPA recipe
(PBB v2): per-token sparse encoder + multi-block masking + EMA target + per-
token target LayerNorm with DINO-style centering + smooth-L1 loss.

Two notebooks ride this library:
  - pbb_cifar10.ipynb       (40% sparse pixels on CIFAR-10, 2D coords + RGB)
  - pbb_cifar10_dvs.ipynb   (event-camera variant, 3D coords + polarity)

Architecture (one diagram):

    sample → Tokenizer(signal_proj + pos_proj∘γ) → BigBird encoder (6 layers,
    per-layer random space-filling-curve permutation) → per-token features
    → gather at target positions → CENTER (DINO EMA) → per-token LN → h_tgt

    context subset → same Tokenizer + encoder → per-token context features
    → predictor (dense Transformer with mask tokens at target coords) → h_pred

    loss = smooth_L1(h_pred, h_tgt)   (EMA target with stop-grad)

The recipe deliberately omits later additions we developed for low-content
event tokens (LocalCrossAttention pool, tokenizer skip, symmetric cross-attn
pool). For modalities where each token already carries >10 bits of content,
direct per-token gather + centering + LN is the *right* architecture; the
extra layers were a workaround for 1-bit tokens.
"""
from __future__ import annotations

from typing import Optional, Dict, Iterator
import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


# ===========================================================================
# Space-filling curves + coord quantization
# ===========================================================================

def _morton_2d(y: np.ndarray, x: np.ndarray, n_bits: int) -> np.ndarray:
    code = np.zeros_like(y, dtype=np.int64)
    for i in range(n_bits):
        code |= ((x >> i) & 1).astype(np.int64) << (2 * i)
        code |= ((y >> i) & 1).astype(np.int64) << (2 * i + 1)
    return code


def _morton_3d(z: np.ndarray, y: np.ndarray, x: np.ndarray, n_bits: int) -> np.ndarray:
    code = np.zeros_like(x, dtype=np.int64)
    for i in range(n_bits):
        code |= ((x >> i) & 1).astype(np.int64) << (3 * i)
        code |= ((y >> i) & 1).astype(np.int64) << (3 * i + 1)
        code |= ((z >> i) & 1).astype(np.int64) << (3 * i + 2)
    return code


def _rot_2d(n, x, y, rx, ry):
    if ry == 0:
        if rx == 1:
            x = n - 1 - x
            y = n - 1 - y
        x, y = y, x
    return x, y


def _hilbert_2d_scalar(y: int, x: int, n: int) -> int:
    rx = 0; ry = 0; d = 0
    s = n // 2
    while s > 0:
        rx = 1 if (x & s) else 0
        ry = 1 if (y & s) else 0
        d += s * s * ((3 * rx) ^ ry)
        x, y = _rot_2d(s, x, y, rx, ry)
        s //= 2
    return d


def _hilbert_2d(y: np.ndarray, x: np.ndarray, side: int) -> np.ndarray:
    out = np.empty_like(y, dtype=np.int64)
    flat_y = y.ravel(); flat_x = x.ravel()
    for i in range(flat_y.size):
        out.ravel()[i] = _hilbert_2d_scalar(int(flat_y[i]), int(flat_x[i]), side)
    return out


def _hilbert_3d_scalar(z: int, y: int, x: int, n: int) -> int:
    """Adapted from John Skilling's 2004 algorithm (compact recursive form).
    Returns a Hilbert-curve rank in [0, n^3) for a point (z, y, x) on a
    cubic grid of side n (n must be power of 2)."""
    coords = [x, y, z]
    bits = int(math.log2(n))
    # Transpose
    for q in range(bits - 1, 0, -1):
        p = 1 << q
        for i in range(3):
            if coords[i] & p:
                coords[0] ^= p
            else:
                t = (coords[0] ^ coords[i]) & p
                coords[0] ^= t
                coords[i] ^= t
    # Gray decode
    for i in range(1, 3):
        coords[i] ^= coords[i - 1]
    t = 0
    for q in range(bits - 1, 0, -1):
        p = 1 << q
        if coords[2] & p: t ^= p - 1
    for i in range(3):
        coords[i] ^= t
    # Interleave
    code = 0
    for i in range(bits):
        for d in range(3):
            code |= ((coords[d] >> i) & 1) << (3 * i + d)
    return code


def _hilbert_3d(z: np.ndarray, y: np.ndarray, x: np.ndarray, side: int) -> np.ndarray:
    out = np.empty_like(z, dtype=np.int64)
    fz = z.ravel(); fy = y.ravel(); fx = x.ravel()
    for i in range(fz.size):
        out.ravel()[i] = _hilbert_3d_scalar(int(fz[i]), int(fy[i]), int(fx[i]), side)
    return out


def _codes_to_rank(codes: np.ndarray) -> np.ndarray:
    order = np.argsort(codes, kind="stable")
    rank = np.empty_like(order)
    rank[order] = np.arange(order.size)
    return rank


def precompute_grid_orderings(side: int, ndim: int = 2) -> Dict[str, torch.Tensor]:
    """For a side**ndim grid, return ranks-per-cell for 4 curves: z, z_rev,
    hilbert, hilbert_rev. Each returned tensor has shape (side**ndim,) and
    is a long-valued rank in [0, side**ndim)."""
    assert ndim in (2, 3)
    assert (side & (side - 1)) == 0, "side must be a power of 2 for Hilbert"
    n_bits = int(math.log2(side))
    if ndim == 2:
        grid_y, grid_x = np.meshgrid(np.arange(side), np.arange(side), indexing="ij")
        morton = _morton_2d(grid_y.ravel(), grid_x.ravel(), n_bits)
        hilbert = _hilbert_2d(grid_y.ravel(), grid_x.ravel(), side)
    else:
        grid_z, grid_y, grid_x = np.meshgrid(
            np.arange(side), np.arange(side), np.arange(side), indexing="ij"
        )
        morton = _morton_3d(grid_z.ravel(), grid_y.ravel(), grid_x.ravel(), n_bits)
        hilbert = _hilbert_3d(grid_z.ravel(), grid_y.ravel(), grid_x.ravel(), side)
    z_rank  = _codes_to_rank(morton)
    z_rev   = (z_rank.max() - z_rank).astype(np.int64)
    h_rank  = _codes_to_rank(hilbert)
    h_rev   = (h_rank.max() - h_rank).astype(np.int64)
    return {
        "z":           torch.from_numpy(z_rank).long(),
        "z_rev":       torch.from_numpy(z_rev).long(),
        "hilbert":     torch.from_numpy(h_rank).long(),
        "hilbert_rev": torch.from_numpy(h_rev).long(),
    }


def quantize_coords(coords: torch.Tensor, side: int, value_range=(-1.0, 1.0)) -> torch.Tensor:
    """Quantize continuous coords (in `value_range`) to grid cell indices."""
    lo, hi = value_range
    norm = (coords - lo) / (hi - lo)
    norm = norm.clamp(0.0, 1.0)
    cell = (norm * (side - 1)).round().long()
    cell = cell.clamp(0, side - 1)
    if coords.size(-1) == 2:
        return cell[..., 0] * side + cell[..., 1]
    # 3-D: assume order (x, y, t)
    return cell[..., 2] * side * side + cell[..., 1] * side + cell[..., 0]


def invert_perm(perm: torch.Tensor) -> torch.Tensor:
    """perm: (B, N) Long → inverse (B, N) such that inv[b, perm[b, i]] = i."""
    B, N = perm.shape
    inv = torch.empty_like(perm)
    arange = torch.arange(N, device=perm.device).unsqueeze(0).expand(B, -1)
    inv.scatter_(1, perm, arange)
    return inv


def subset_perm(subset_ids: torch.Tensor, full_rank: torch.Tensor) -> torch.Tensor:
    """Given a subset (B, K) of cell IDs and full grid ranks (S,), return
    a permutation (B, K) that sorts the subset by curve rank."""
    ranks = full_rank[subset_ids]            # (B, K)
    return ranks.argsort(dim=-1)


# ===========================================================================
# Tokenizer
# ===========================================================================

class GaussianFourierFeatures(nn.Module):
    """γ(c) = [sin(B·c), cos(B·c)] with B fixed-random."""
    def __init__(self, in_features=2, mapping_size=96, scale=15.0):
        super().__init__()
        self.register_buffer("B", torch.randn(in_features, mapping_size) * scale)

    def forward(self, coords):
        proj = coords @ self.B
        return torch.cat([torch.sin(proj), torch.cos(proj)], dim=-1)


class Tokenizer(nn.Module):
    """per-token = signal_proj(signal) + pos_proj(γ(coord))."""
    def __init__(self, d_model=256, signal_dim=3, coord_dim=2,
                 fourier_dim=96, fourier_scale=15.0):
        super().__init__()
        self.gff = GaussianFourierFeatures(coord_dim, fourier_dim, scale=fourier_scale)
        self.signal_proj = nn.Linear(signal_dim, d_model)
        self.pos_proj   = nn.Linear(2 * fourier_dim, d_model)

    def forward(self, signal, coords):
        return self.signal_proj(signal) + self.pos_proj(self.gff(coords))


# ===========================================================================
# Attention
# ===========================================================================

class MultiHeadAttention(nn.Module):
    def __init__(self, dim, n_heads=8, dim_head=32, bias_qkv=False):
        super().__init__()
        inner = n_heads * dim_head
        self.n_heads = n_heads
        self.dim_head = dim_head
        self.scale = dim_head ** -0.5
        self.to_qkv = nn.Linear(dim, inner * 3, bias=bias_qkv)
        self.to_out = nn.Linear(inner, dim)

    def _split(self, x):
        B, N, _ = x.shape
        return x.view(B, N, self.n_heads, self.dim_head).transpose(1, 2)

    def forward(self, x, key_padding_mask=None):
        B, N, _ = x.shape
        qkv = self.to_qkv(x)
        q, k, v = qkv.chunk(3, dim=-1)
        q = self._split(q); k = self._split(k); v = self._split(v)
        scores = torch.einsum("bhnd,bhmd->bhnm", q, k) * self.scale
        if key_padding_mask is not None:
            scores = scores.masked_fill(key_padding_mask.view(B, 1, 1, N), float("-inf"))
        attn = F.softmax(scores, dim=-1)
        if key_padding_mask is not None:
            attn = torch.nan_to_num(attn, nan=0.0)
        out = torch.einsum("bhnm,bhmd->bhnd", attn, v)
        out = out.transpose(1, 2).contiguous().view(B, N, -1)
        return self.to_out(out)


class BigBirdSparseAttention(nn.Module):
    """BigBird block-sparse self-attention along a (re-)serialized sequence.

    Each query block attends to: `window` blocks on each side, `n_global`
    fixed leading blocks, and `n_random` random other blocks. Pattern is
    fixed across forward calls (same all-layer pattern; sequence order is
    re-shuffled per layer via the encoder block's perm).
    """

    def __init__(self, dim, n_heads=8, dim_head=32, block_size=8,
                 window=1, n_random=2, n_global=2, bias_qkv=False):
        super().__init__()
        self.block_size = block_size
        self.window = window
        self.n_random = n_random
        self.n_global = n_global
        inner = n_heads * dim_head
        self.n_heads = n_heads
        self.dim_head = dim_head
        self.scale = dim_head ** -0.5
        self.to_qkv = nn.Linear(dim, inner * 3, bias=bias_qkv)
        self.to_out = nn.Linear(inner, dim)
        self._static = {}                       # (NB, device) → attended LongTensor

    def _attended_pattern(self, NB, device):
        key = (NB, str(device))
        if key in self._static:
            return self._static[key]
        # Build per-query-block attended-block indices: (NB, K_attended)
        attended_per_query = []
        for q in range(NB):
            # Window: q-window..q+window (clipped)
            window = list(range(max(0, q - self.window), min(NB, q + self.window + 1)))
            # Globals: leading n_global blocks
            globals_ = list(range(self.n_global))
            # Random: n_random NOT-already-included blocks
            existing = set(window + globals_)
            available = [i for i in range(NB) if i not in existing]
            random = list(np.random.RandomState(q + 13).choice(
                available, size=min(self.n_random, len(available)), replace=False
            )) if available else []
            attended = sorted(set(window + globals_ + random))
            attended_per_query.append(attended)
        # Pad each row to same length
        max_K = max(len(r) for r in attended_per_query)
        padded = np.full((NB, max_K), -1, dtype=np.int64)
        for q, row in enumerate(attended_per_query):
            padded[q, :len(row)] = row
        # For -1 (rare; if NB < total attended for some q), repeat first valid
        for q in range(NB):
            for k in range(max_K):
                if padded[q, k] == -1:
                    padded[q, k] = padded[q, 0]
        static = torch.from_numpy(padded).long().to(device)
        self._static[key] = static
        return static

    def forward(self, x, key_padding_mask=None):
        B, N, _ = x.shape
        BS = self.block_size
        assert N % BS == 0, f"N={N} not divisible by block_size={BS}"
        NB = N // BS
        H = self.n_heads; Dh = self.dim_head

        qkv = self.to_qkv(x)
        q, k, v = qkv.chunk(3, dim=-1)
        q = q.view(B, N, H, Dh).transpose(1, 2)        # (B, H, N, Dh)
        k = k.view(B, N, H, Dh).transpose(1, 2)
        v = v.view(B, N, H, Dh).transpose(1, 2)
        q_b = q.reshape(B, H, NB, BS, Dh)
        k_b = k.reshape(B, H, NB, BS, Dh)
        v_b = v.reshape(B, H, NB, BS, Dh)

        attended = self._attended_pattern(NB, x.device)   # (NB, KH)
        KH = attended.shape[1]
        flat = attended.reshape(-1)                        # (NB*KH,)
        k_sel = k_b[:, :, flat, :, :].reshape(B, H, NB, KH * BS, Dh)
        v_sel = v_b[:, :, flat, :, :].reshape(B, H, NB, KH * BS, Dh)

        scores = torch.einsum("bhnqd,bhnkd->bhnqk", q_b, k_sel) * self.scale
        if key_padding_mask is not None:
            pm = key_padding_mask.view(B, NB, BS)
            pm_sel = pm[:, flat, :].reshape(B, NB, KH * BS)
            scores = scores.masked_fill(pm_sel.unsqueeze(1).unsqueeze(3), float("-inf"))
        attn = F.softmax(scores, dim=-1)
        if key_padding_mask is not None:
            attn = torch.nan_to_num(attn, nan=0.0)
        out_b = torch.einsum("bhnqk,bhnkd->bhnqd", attn, v_sel)
        out = out_b.reshape(B, H, N, Dh).transpose(1, 2).contiguous().view(B, N, -1)
        return self.to_out(out)


class FeedForward(nn.Module):
    def __init__(self, dim, mult=4):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, dim * mult), nn.GELU(),
            nn.Linear(dim * mult, dim),
        )
    def forward(self, x): return self.net(x)


# ===========================================================================
# Encoder
# ===========================================================================

def _gather_along_seq(x, perm):
    B, N, D = x.shape
    return torch.gather(x, 1, perm.unsqueeze(-1).expand(B, N, D))


def _gather_mask(mask, perm):
    return torch.gather(mask, 1, perm)


class EncoderBlock(nn.Module):
    """One block: (gather→BigBird→FFN→scatter)."""

    def __init__(self, dim, n_heads=8, dim_head=32, block_size=8,
                 window=1, n_random=2, n_global=2, ffn_mult=4):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = BigBirdSparseAttention(
            dim, n_heads=n_heads, dim_head=dim_head, block_size=block_size,
            window=window, n_random=n_random, n_global=n_global)
        self.norm2 = nn.LayerNorm(dim)
        self.ffn = FeedForward(dim, mult=ffn_mult)
        self.block_size = block_size

    def forward(self, x, perm, inverse_perm, key_padding_mask=None):
        x_p = _gather_along_seq(x, perm)
        pm_p = _gather_mask(key_padding_mask, perm) if key_padding_mask is not None else None
        x_p = x_p + self.attn(self.norm1(x_p), key_padding_mask=pm_p)
        x_p = x_p + self.ffn(self.norm2(x_p))
        return _gather_along_seq(x_p, inverse_perm)


class PBBEncoder(nn.Module):
    """PointBigBird encoder: per-token Tokenizer → 6 BigBird blocks with
    per-layer random SFC permutation → final LN."""

    def __init__(self, d_model=256, n_layers=6, n_heads=8, dim_head=32,
                 block_size=8, window=1, n_random=2, n_global=2,
                 ffn_mult=4, signal_dim=3, coord_dim=2,
                 fourier_dim=96, fourier_scale=15.0,
                 serial_orders=("z", "z_rev", "hilbert", "hilbert_rev")):
        super().__init__()
        self.tokenizer = Tokenizer(d_model, signal_dim=signal_dim, coord_dim=coord_dim,
                                    fourier_dim=fourier_dim, fourier_scale=fourier_scale)
        self.blocks = nn.ModuleList([
            EncoderBlock(d_model, n_heads=n_heads, dim_head=dim_head,
                         block_size=block_size, window=window,
                         n_random=n_random, n_global=n_global, ffn_mult=ffn_mult)
            for _ in range(n_layers)
        ])
        self.norm = nn.LayerNorm(d_model)
        self.serial_orders = tuple(serial_orders)
        self.block_size = block_size

    @staticmethod
    def _pad_to_multiple(x, block_size, key_padding_mask):
        B, K, D = x.shape
        rem = (-K) % block_size
        if rem == 0:
            return x, key_padding_mask, K
        pad_x = torch.zeros(B, rem, D, device=x.device, dtype=x.dtype)
        x_p = torch.cat([x, pad_x], dim=1)
        if key_padding_mask is None:
            key_padding_mask = torch.zeros(B, K, device=x.device, dtype=torch.bool)
        pad_pm = torch.ones(B, rem, device=x.device, dtype=torch.bool)
        pm_p = torch.cat([key_padding_mask, pad_pm], dim=1)
        return x_p, pm_p, K

    def forward(self, signal, coords,
                orderings: Dict[str, Dict[str, torch.Tensor]],
                key_padding_mask: Optional[torch.Tensor] = None):
        x = self.tokenizer(signal, coords)
        x, pm, K_orig = self._pad_to_multiple(x, self.block_size, key_padding_mask)
        B, Kp, _ = x.shape

        extended = {}
        for name, d in orderings.items():
            p = d["perm"]; inv = d["inverse"]
            if p.shape[1] != Kp:
                tail = torch.arange(p.shape[1], Kp, device=p.device).unsqueeze(0).expand(B, -1)
                p = torch.cat([p, tail], dim=1)
                inv = torch.cat([inv, tail], dim=1)
            extended[name] = (p, inv)

        order_names = list(self.serial_orders)
        for blk in self.blocks:
            name = order_names[torch.randint(0, len(order_names), (1,)).item()]
            perm, inv = extended[name]
            x = blk(x, perm, inv, key_padding_mask=pm)
        x = self.norm(x)
        return x[:, :K_orig]


# ===========================================================================
# Predictor
# ===========================================================================

class PredictorBlock(nn.Module):
    def __init__(self, dim, n_heads=6, dim_head=32, ffn_mult=4):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn  = MultiHeadAttention(dim, n_heads=n_heads, dim_head=dim_head)
        self.norm2 = nn.LayerNorm(dim)
        self.ffn   = FeedForward(dim, mult=ffn_mult)
    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        return x + self.ffn(self.norm2(x))


class PBBPredictor(nn.Module):
    """Dense Transformer over [ctx tokens || mask-tokens at target coords]
    that reads off predicted features at the target positions.

    pos_symmetric=True (PBB v2): ctx tokens also receive a positional
    embedding before the predictor runs, so ctx and tgt sides are
    symmetrically position-aware.
    """

    def __init__(self, d_model=256, d_pred=192, n_layers=4,
                 n_heads=6, dim_head=32, coord_dim=2,
                 fourier_dim=96, fourier_scale=15.0, ffn_mult=4,
                 pos_symmetric: bool = True):
        super().__init__()
        self.proj_in    = nn.Linear(d_model, d_pred)
        self.gff        = GaussianFourierFeatures(coord_dim, fourier_dim, scale=fourier_scale)
        self.proj_pos   = nn.Linear(2 * fourier_dim, d_pred)
        self.mask_token = nn.Parameter(torch.zeros(1, 1, d_pred))
        nn.init.trunc_normal_(self.mask_token, std=0.02)
        self.blocks = nn.ModuleList([
            PredictorBlock(d_pred, n_heads=n_heads, dim_head=dim_head, ffn_mult=ffn_mult)
            for _ in range(n_layers)
        ])
        self.norm     = nn.LayerNorm(d_pred)
        self.proj_out = nn.Linear(d_pred, d_model)
        self.pos_symmetric = pos_symmetric

    def forward(self, ctx_feat, target_coords, ctx_coords=None):
        B, K_ctx, _ = ctx_feat.shape
        ctx_tok = self.proj_in(ctx_feat)
        if self.pos_symmetric:
            assert ctx_coords is not None
            ctx_tok = ctx_tok + self.proj_pos(self.gff(ctx_coords))
        tgt_tok = self.proj_pos(self.gff(target_coords)) + self.mask_token
        x = torch.cat([ctx_tok, tgt_tok], dim=1)
        for blk in self.blocks: x = blk(x)
        x = self.norm(x[:, K_ctx:])
        return self.proj_out(x)


# ===========================================================================
# Target centering + EMA + loss + helpers
# ===========================================================================

class TargetCenter(nn.Module):
    """DINO-style EMA of the per-feature batch mean. Subtract before LN."""
    def __init__(self, embed_dim, momentum=0.9):
        super().__init__()
        self.momentum = momentum
        self.register_buffer("center", torch.zeros(1, 1, embed_dim))

    @torch.no_grad()
    def update(self, h):
        batch_center = h.mean(dim=(0, 1), keepdim=True)
        self.center.mul_(self.momentum).add_(batch_center, alpha=1.0 - self.momentum)

    def forward(self, h):
        return h - self.center


def ema_update(target_module: nn.Module, online_module: nn.Module, m: float):
    """target = m·target + (1-m)·online."""
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


def gather_target_features(g_tgt: torch.Tensor, tgt_pool_pos: torch.Tensor) -> torch.Tensor:
    """g_tgt: (B, K_pool, D) target-encoder features over the full pool.
    tgt_pool_pos: (B, N_tgt) long — index of each target within the pool.
    Returns: (B, N_tgt, D)."""
    B, K_pool, D = g_tgt.shape
    idx = tgt_pool_pos.unsqueeze(-1).expand(B, tgt_pool_pos.shape[1], D)
    return torch.gather(g_tgt, 1, idx)


# ===========================================================================
# Helpers
# ===========================================================================

def save_atomic(state, path):
    tmp = path + ".tmp"
    torch.save(state, tmp)
    os.replace(tmp, path) if False else (__import__("os").replace(tmp, path))


def short_params(m):
    n = sum(p.numel() for p in m.parameters())
    return f"{n/1e6:.2f} M params"
