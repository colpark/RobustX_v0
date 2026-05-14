"""Attention modules: vanilla multi-head + BigBird block-sparse.

`MultiHeadAttention` is plain dense scaled-dot-product attention with an
optional key-padding mask. `BigBirdSparseAttention` implements the
block-sparse pattern from Zaheer et al. 2020 in PyTorch using
`index_select` / `gather` — no custom kernels, but compute is
O(N · (2W+1+G+R) · B) instead of O(N²).

The two classes share Q/K/V projections (same signature) so a model can
swap between them. Set `equivalent_to_dense=True` in BigBird to bypass
the sparse path and call dense attention (useful for testing).
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Vanilla dense attention (reference)
# ---------------------------------------------------------------------------

class MultiHeadAttention(nn.Module):
    """Standard scaled-dot-product attention with optional key-padding mask."""

    def __init__(self, dim, n_heads=8, dim_head=32, bias_qkv=False):
        super().__init__()
        inner = n_heads * dim_head
        self.n_heads = n_heads
        self.dim_head = dim_head
        self.scale = dim_head ** -0.5
        self.to_qkv = nn.Linear(dim, inner * 3, bias=bias_qkv)
        self.to_out = nn.Linear(inner, dim)

    def _split_heads(self, x):
        B, N, _ = x.shape
        return x.view(B, N, self.n_heads, self.dim_head).transpose(1, 2)

    def forward(self, x, key_padding_mask=None):
        """x: (B, N, D); key_padding_mask: (B, N) bool, True at padding."""
        B, N, _ = x.shape
        qkv = self.to_qkv(x)
        q, k, v = qkv.chunk(3, dim=-1)
        q = self._split_heads(q)               # (B, H, N, Dh)
        k = self._split_heads(k)
        v = self._split_heads(v)

        scores = torch.einsum("bhnd,bhmd->bhnm", q, k) * self.scale
        if key_padding_mask is not None:
            scores = scores.masked_fill(
                key_padding_mask.view(B, 1, 1, N), float("-inf")
            )
        attn = F.softmax(scores, dim=-1)
        out = torch.einsum("bhnm,bhmd->bhnd", attn, v)             # (B, H, N, Dh)
        out = out.transpose(1, 2).contiguous().view(B, N, -1)       # (B, N, H*Dh)
        return self.to_out(out)


# ---------------------------------------------------------------------------
# BigBird block-sparse attention
# ---------------------------------------------------------------------------

class BigBirdSparseAttention(nn.Module):
    """Block-sparse BigBird attention.

    Sequence (length N, must be a multiple of `block_size`) is divided into
    `num_blocks = N / block_size` blocks. Each *query block* attends to:
        * `(2*window + 1)` blocks centered on itself (clamped at boundaries),
        * `n_global` "global" blocks (default: first and last),
        * `n_random` randomly sampled blocks (fresh sample per forward).

    The same set of attended *block indices* is reused across all queries
    inside a block, so we gather K/V once per query block.

    Total cost: O(N * K_attended * dim_head) where
        K_attended = (2W+1 + G + R) * block_size
    vs O(N²) for dense. Set `equivalent_to_dense=True` to bypass the sparse
    path entirely (useful for testing & ablation).
    """

    def __init__(
        self,
        dim,
        n_heads=8,
        dim_head=32,
        block_size=32,
        window=1,
        n_random=2,
        n_global=2,
        bias_qkv=False,
        equivalent_to_dense=False,
    ):
        super().__init__()
        inner = n_heads * dim_head
        self.n_heads = n_heads
        self.dim_head = dim_head
        self.scale = dim_head ** -0.5
        self.block_size = block_size
        self.window = window
        self.n_random = n_random
        self.n_global = n_global
        self.equivalent_to_dense = equivalent_to_dense
        self.to_qkv = nn.Linear(dim, inner * 3, bias=bias_qkv)
        self.to_out = nn.Linear(inner, dim)
        # Cached attended-block-index pattern, recomputed on shape change
        self._cached_nb = None
        self._cached_static = None  # window + global indices (deterministic)

    # ------------------------------------------------------------------
    # Static pattern: window + globals (no randomness, can be cached)
    # ------------------------------------------------------------------
    def _static_pattern(self, num_blocks, device):
        if self._cached_nb == num_blocks and self._cached_static is not None \
                and self._cached_static.device == device:
            return self._cached_static
        b = torch.arange(num_blocks, device=device)
        win = torch.arange(-self.window, self.window + 1, device=device)
        win_idx = (b.unsqueeze(1) + win.unsqueeze(0)).clamp(0, num_blocks - 1)   # (NB, 2W+1)
        if self.n_global == 2:
            globals_ = torch.tensor([0, num_blocks - 1], device=device).unsqueeze(0).expand(num_blocks, -1)
        elif self.n_global == 1:
            globals_ = torch.zeros(num_blocks, 1, device=device, dtype=torch.long)
        else:
            globals_ = torch.arange(self.n_global, device=device).unsqueeze(0).expand(num_blocks, -1)
        static = torch.cat([win_idx, globals_], dim=1).contiguous()              # (NB, 2W+1 + G)
        self._cached_nb = num_blocks
        self._cached_static = static
        return static

    def _attended_pattern(self, num_blocks, device, generator=None):
        static = self._static_pattern(num_blocks, device)
        if self.n_random > 0:
            rand_idx = torch.randint(
                0, num_blocks, (num_blocks, self.n_random),
                device=device, generator=generator,
            )
            return torch.cat([static, rand_idx], dim=1)                          # (NB, KH)
        return static

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------
    def _split_heads(self, x):
        B, N, _ = x.shape
        return x.view(B, N, self.n_heads, self.dim_head).transpose(1, 2)

    def _dense_forward(self, q, k, v, key_padding_mask):
        scores = torch.einsum("bhnd,bhmd->bhnm", q, k) * self.scale
        if key_padding_mask is not None:
            scores = scores.masked_fill(
                key_padding_mask.view(q.size(0), 1, 1, -1), float("-inf")
            )
        attn = F.softmax(scores, dim=-1)
        return torch.einsum("bhnm,bhmd->bhnd", attn, v)

    def forward(self, x, key_padding_mask=None):
        """x: (B, N, D); N must be divisible by `block_size`.

        key_padding_mask: (B, N) bool, True at padding positions.
        """
        B, N, _ = x.shape
        BS = self.block_size
        assert N % BS == 0, f"N={N} must be divisible by block_size={BS}"

        qkv = self.to_qkv(x)
        q, k, v = qkv.chunk(3, dim=-1)
        q = self._split_heads(q)                          # (B, H, N, Dh)
        k = self._split_heads(k)
        v = self._split_heads(v)

        if self.equivalent_to_dense:
            out = self._dense_forward(q, k, v, key_padding_mask)
            out = out.transpose(1, 2).contiguous().view(B, N, -1)
            return self.to_out(out)

        # Block-sparse path
        NB = N // BS
        H = self.n_heads
        Dh = self.dim_head

        q_b = q.reshape(B, H, NB, BS, Dh)
        k_b = k.reshape(B, H, NB, BS, Dh)
        v_b = v.reshape(B, H, NB, BS, Dh)

        attended = self._attended_pattern(NB, x.device)   # (NB, KH)
        KH = attended.shape[1]

        flat = attended.reshape(-1)                       # (NB*KH,)
        # Gather K, V blocks per query block.  k_b indexed on block dim (=2).
        k_sel = k_b[:, :, flat, :, :].reshape(B, H, NB, KH * BS, Dh)
        v_sel = v_b[:, :, flat, :, :].reshape(B, H, NB, KH * BS, Dh)

        scores = torch.einsum("bhnqd,bhnkd->bhnqk", q_b, k_sel) * self.scale   # (B, H, NB, BS, KH*BS)

        if key_padding_mask is not None:
            pm = key_padding_mask.view(B, NB, BS)                       # (B, NB, BS)
            pm_sel = pm[:, flat, :].reshape(B, NB, KH * BS)             # (B, NB, KH*BS)
            scores = scores.masked_fill(
                pm_sel.unsqueeze(1).unsqueeze(3), float("-inf")
            )

        attn = F.softmax(scores, dim=-1)
        out_b = torch.einsum("bhnqk,bhnkd->bhnqd", attn, v_sel)         # (B, H, NB, BS, Dh)
        out = out_b.reshape(B, H, N, Dh).transpose(1, 2).contiguous().view(B, N, -1)
        return self.to_out(out)


# ---------------------------------------------------------------------------
# Convenience: tied factory
# ---------------------------------------------------------------------------

def make_attention(kind: str, **kw):
    if kind == "dense":
        # Filter to args MultiHeadAttention accepts
        return MultiHeadAttention(
            kw["dim"], n_heads=kw.get("n_heads", 8),
            dim_head=kw.get("dim_head", 32),
            bias_qkv=kw.get("bias_qkv", False),
        )
    elif kind == "bigbird":
        return BigBirdSparseAttention(**kw)
    raise ValueError(f"unknown attention kind: {kind}")
