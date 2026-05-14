"""PointBigBird encoder + JEPA predictor.

The encoder is a stack of `EncoderBlock`s. Each block:
  1. Samples one of 4 serialization orders (z, z_rev, hilbert, hilbert_rev)
     uniformly at random.
  2. Gathers the token sequence into that order (single torch.gather op).
  3. Runs BigBird block-sparse self-attention + FFN.
  4. Scatters tokens back to the original order.

The predictor takes the encoder's context tokens plus mask tokens injected
at the target coordinates, runs a small dense Transformer, and reads off
predictions at the target positions.
"""
from __future__ import annotations

from typing import Dict, Optional

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from .attention import MultiHeadAttention, BigBirdSparseAttention, GroupedSparseAttention


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def get_sinusoidal_embeddings(n: int, d: int) -> torch.Tensor:
    """Standard sinusoidal positional embeddings, shape (n, d), with d even.

    Used by ICMR to initialize the shared learnable latent set so that even at
    init the latent slots span distinguishable positions in the d-dim space
    (perceiver-style trick — avoids degenerate symmetries when all latents
    start identical).
    """
    assert d % 2 == 0, f"d must be even, got {d}"
    position = torch.arange(n, dtype=torch.float).unsqueeze(1)
    div_term = torch.exp(torch.arange(0, d, 2).float() * -(math.log(10000.0) / d))
    pe = torch.zeros(n, d)
    pe[:, 0::2] = torch.sin(position * div_term)
    pe[:, 1::2] = torch.cos(position * div_term)
    return pe


# ---------------------------------------------------------------------------
# Tokenization
# ---------------------------------------------------------------------------

class GaussianFourierFeatures(nn.Module):
    """γ(c) = [sin(B·c), cos(B·c)] with B fixed-random."""

    def __init__(self, in_features=2, mapping_size=96, scale=15.0):
        super().__init__()
        self.register_buffer("B", torch.randn(in_features, mapping_size) * scale)

    def forward(self, coords):
        proj = coords @ self.B
        return torch.cat([torch.sin(proj), torch.cos(proj)], dim=-1)


class Tokenizer(nn.Module):
    """Per-point token = signal_proj(signal) + pos_proj(γ(coord)).

    For 2-D images, signal_dim=3 (RGB) and coord_dim=2 (y, x).
    For event cameras, signal_dim=1 (polarity) and coord_dim=3 (x, y, t).
    """

    def __init__(self, d_model=256, signal_dim=3, coord_dim=2,
                 fourier_dim=96, fourier_scale=15.0):
        super().__init__()
        self.gff = GaussianFourierFeatures(coord_dim, fourier_dim, scale=fourier_scale)
        self.signal_proj = nn.Linear(signal_dim, d_model)
        self.pos_proj = nn.Linear(2 * fourier_dim, d_model)

    def forward(self, signal, coords):
        """signal: (B, K, signal_dim); coords: (B, K, coord_dim) → tokens (B, K, D)."""
        return self.signal_proj(signal) + self.pos_proj(self.gff(coords))


# ---------------------------------------------------------------------------
# Encoder block — serialize + BigBird attention + FFN
# ---------------------------------------------------------------------------

class FeedForward(nn.Module):
    def __init__(self, dim, mult=4):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, dim * mult), nn.GELU(),
            nn.Linear(dim * mult, dim),
        )

    def forward(self, x):
        return self.net(x)


def _gather_along_seq(x, perm):
    """x: (B, N, D); perm: (B, N) long → x_perm[b, i, :] = x[b, perm[b, i], :]."""
    B, N, D = x.shape
    return torch.gather(x, 1, perm.unsqueeze(-1).expand(B, N, D))


def _gather_mask(mask, perm):
    """mask: (B, N) → permuted (B, N)."""
    return torch.gather(mask, 1, perm)


class EncoderBlock(nn.Module):
    """One block: order-permute → sparse self-attn → FFN → un-permute.

    Two attention flavors, selected by `attention_type`:
      "bigbird": BigBird block-sparse (window + globals + random).
      "grouped": dense self-attention WITHIN windows of size `group_size`;
                 cross-window mixing comes from the next layer's different
                 curve choice. Cheaper, comparable receptive field via depth.
    """

    def __init__(self, dim, n_heads=8, dim_head=32, block_size=32,
                 window=1, n_random=2, n_global=2, ffn_mult=4,
                 attention_type: str = "bigbird", group_size: int = 16):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attention_type = attention_type
        if attention_type == "grouped":
            self.attn = GroupedSparseAttention(
                dim, n_heads=n_heads, dim_head=dim_head, group_size=group_size,
            )
            self.required_multiple = group_size
        else:
            self.attn = BigBirdSparseAttention(
                dim, n_heads=n_heads, dim_head=dim_head,
                block_size=block_size, window=window,
                n_random=n_random, n_global=n_global,
            )
            self.required_multiple = block_size
        self.norm2 = nn.LayerNorm(dim)
        self.ffn = FeedForward(dim, mult=ffn_mult)

    def forward(self, x, perm, inverse_perm, pos_emb=None, key_padding_mask=None):
        """x: (B, N, D)  N divisible by block_size
        perm:         (B, N) — permutes original→sorted-by-curve
        inverse_perm: (B, N) — inverse of perm
        pos_emb:      (B, N, D) — positional embedding to re-inject as a residual
                                  before attention. If None, no re-injection.
        key_padding_mask: (B, N) bool in *original* order

        Re-injecting pos_emb at every layer is what makes the per-layer random
        re-shuffling meaningful: it forces the model to rely on the true spatial
        positional embedding rather than current sequence-position, since sequence
        position changes layer to layer but pos_emb is the same true (y, x).
        """
        # Re-inject true positional embedding as a residual at this layer
        if pos_emb is not None:
            x = x + pos_emb

        # Gather into curve order (and gather padding mask too)
        x_p = _gather_along_seq(x, perm)
        pm_p = _gather_mask(key_padding_mask, perm) if key_padding_mask is not None else None

        # BigBird sparse attention in curve order
        x_p = x_p + self.attn(self.norm1(x_p), key_padding_mask=pm_p)

        # FFN (position-wise, so ordering doesn't matter — still on curve order)
        x_p = x_p + self.ffn(self.norm2(x_p))

        # Scatter back to original order
        return _gather_along_seq(x_p, inverse_perm)


# ---------------------------------------------------------------------------
# Full encoder
# ---------------------------------------------------------------------------

class OmniBirdEncoder(nn.Module):
    """Tokenizer → L stacked EncoderBlocks (each with a randomly chosen ordering).

    Same body as PointBigBird's encoder, but tokenizer now accepts arbitrary
    signal_dim and coord_dim. For event-camera data, signal_dim=1 (polarity)
    and coord_dim=3 (x, y, t).
    """

    def __init__(self, d_model=256, n_layers=6, n_heads=8, dim_head=32,
                 block_size=8, window=1, n_random=2, n_global=2,
                 ffn_mult=4, signal_dim=1, coord_dim=3,
                 fourier_dim=96, fourier_scale=15.0,
                 serial_orders=("z", "z_rev", "hilbert", "hilbert_rev"),
                 reinject_pos=False,
                 attention_type: str = "bigbird", group_size: int = 16):
        super().__init__()
        self.tokenizer = Tokenizer(d_model, signal_dim=signal_dim, coord_dim=coord_dim,
                                    fourier_dim=fourier_dim, fourier_scale=fourier_scale)
        self.attention_type = attention_type
        self.group_size = group_size
        # Padding multiple depends on which attention we use.
        self.pad_multiple = group_size if attention_type == "grouped" else block_size
        self.blocks = nn.ModuleList([
            EncoderBlock(d_model, n_heads=n_heads, dim_head=dim_head,
                         block_size=block_size, window=window,
                         n_random=n_random, n_global=n_global,
                         ffn_mult=ffn_mult,
                         attention_type=attention_type,
                         group_size=group_size)
            for _ in range(n_layers)
        ])
        self.norm = nn.LayerNorm(d_model)
        self.serial_orders = tuple(serial_orders)
        self.block_size = block_size
        # `block_size` retains its name for back-compat in callers and for
        # _pad_to_block_multiple. We pad to `pad_multiple` instead so the
        # group_size case works correctly.
        self.reinject_pos = reinject_pos

    def compute_pos_emb(self, coords):
        """pos_emb = pos_proj(γ(coords)) — same projection used by the tokenizer.

        Re-using the tokenizer's pos_proj ties the input-level and per-layer
        positional embeddings, which keeps the parameter count flat and means
        a single set of pos weights is learned.
        """
        return self.tokenizer.pos_proj(self.tokenizer.gff(coords))

    @staticmethod
    def _pad_to_block_multiple(x, block_size, key_padding_mask):
        """Pad (B, K, D) → (B, K', D) where K' is the next multiple of block_size."""
        B, K, D = x.shape
        rem = (-K) % block_size
        if rem == 0:
            return x, key_padding_mask, K
        pad_x = torch.zeros(B, rem, D, device=x.device, dtype=x.dtype)
        x_p = torch.cat([x, pad_x], dim=1)
        if key_padding_mask is None:
            key_padding_mask = torch.zeros(B, K, device=x.device, dtype=torch.bool)
        pad_pm = torch.ones(B, rem, device=x.device, dtype=torch.bool)  # True = pad
        pm_p = torch.cat([key_padding_mask, pad_pm], dim=1)
        return x_p, pm_p, K

    def forward(
        self,
        signal: torch.Tensor,
        coords: torch.Tensor,
        orderings: Dict[str, Dict[str, torch.Tensor]],
        key_padding_mask: Optional[torch.Tensor] = None,
    ):
        """
        signal: (B, K, signal_dim);  coords: (B, K, coord_dim)
        orderings: dict[order_name] -> dict with 'perm' and 'inverse' tensors (B, K).
                   May include extra orderings for `K' > K` padded positions:
                   if perm/inverse is shorter than K' we extend with identity for pads.

        Returns:
            features (B, K, D) — un-permuted, in input order; padding positions
            already filtered out.
        """
        # Tokenize
        x = self.tokenizer(signal, coords)                  # (B, K, D)

        # Compute the *true* positional embedding once (before padding).
        # We re-add it as a residual at every encoder layer so the model
        # always has access to true spatial position, regardless of how
        # earlier layers re-shuffled the sequence.
        pos_emb = self.compute_pos_emb(coords) if self.reinject_pos else None

        x, pm, K_orig = self._pad_to_block_multiple(x, self.pad_multiple, key_padding_mask)
        B, Kp, D = x.shape

        # Pad pos_emb to the same length with zeros (padded positions don't matter
        # since their attention contribution is masked out anyway).
        if pos_emb is not None and pos_emb.shape[1] != Kp:
            rem = Kp - pos_emb.shape[1]
            pad_pe = torch.zeros(B, rem, D, device=x.device, dtype=x.dtype)
            pos_emb = torch.cat([pos_emb, pad_pe], dim=1)

        # Extend orderings to padded length: pad indices stay at the end in identity order
        extended = {}
        for name, d in orderings.items():
            p = d["perm"]; inv = d["inverse"]
            if p.shape[1] != Kp:
                tail = torch.arange(p.shape[1], Kp, device=p.device).unsqueeze(0).expand(B, -1)
                p = torch.cat([p, tail], dim=1)
                inv = torch.cat([inv, tail], dim=1)
            extended[name] = (p, inv)

        # Stack of encoder blocks, each picks an order at random
        order_names = list(self.serial_orders)
        for blk in self.blocks:
            name = order_names[torch.randint(0, len(order_names), (1,)).item()]
            perm, inv = extended[name]
            x = blk(x, perm, inv, pos_emb=pos_emb, key_padding_mask=pm)

        x = self.norm(x)
        return x[:, :K_orig]    # strip padding


# ---------------------------------------------------------------------------
# JEPA predictor (i-JEPA style)
# ---------------------------------------------------------------------------

class PredictorBlock(nn.Module):
    def __init__(self, dim, n_heads=6, dim_head=32, ffn_mult=4):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = MultiHeadAttention(dim, n_heads=n_heads, dim_head=dim_head)
        self.norm2 = nn.LayerNorm(dim)
        self.ffn = FeedForward(dim, mult=ffn_mult)

    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        return x + self.ffn(self.norm2(x))


class OmniBirdPredictor(nn.Module):
    """Dense Transformer that takes encoder context tokens + mask tokens at
    target coordinates and predicts target features.

    If `pos_symmetric=True`, the predictor expects `ctx_coords` to be passed
    along with `ctx_feat`, and adds `proj_pos(γ(ctx_coords))` to the context
    tokens so that both context and target tokens carry explicit positional
    information (matches i-JEPA's predictor more faithfully).
    """

    def __init__(self, d_model=256, d_pred=192, n_layers=4, n_heads=6, dim_head=32,
                 coord_dim=2, fourier_dim=96, fourier_scale=15.0, ffn_mult=4,
                 pos_symmetric=False):
        super().__init__()
        self.proj_in   = nn.Linear(d_model, d_pred)
        # coord_dim=2 for images (y, x); coord_dim=3 for events (x, y, t).
        self.gff       = GaussianFourierFeatures(coord_dim, fourier_dim, scale=fourier_scale)
        self.proj_pos  = nn.Linear(2 * fourier_dim, d_pred)
        self.mask_token = nn.Parameter(torch.zeros(1, 1, d_pred))
        nn.init.trunc_normal_(self.mask_token, std=0.02)
        self.blocks = nn.ModuleList([
            PredictorBlock(d_pred, n_heads=n_heads, dim_head=dim_head, ffn_mult=ffn_mult)
            for _ in range(n_layers)
        ])
        self.norm = nn.LayerNorm(d_pred)
        self.proj_out = nn.Linear(d_pred, d_model)
        self.pos_symmetric = pos_symmetric

    def forward(self, ctx_feat, target_coords, ctx_coords=None):
        """
        ctx_feat:      (B, K_ctx, D_model)
        target_coords: (B, N_tgt, 2)
        ctx_coords:    (B, K_ctx, 2)  — required if `pos_symmetric=True`
        Returns: h_pred (B, N_tgt, D_model)
        """
        B, K, _ = ctx_feat.shape
        ctx_tok = self.proj_in(ctx_feat)                          # (B, K, D_pred)
        if self.pos_symmetric:
            assert ctx_coords is not None, \
                "pos_symmetric=True requires ctx_coords to be passed"
            ctx_tok = ctx_tok + self.proj_pos(self.gff(ctx_coords))
        tgt_tok = self.proj_pos(self.gff(target_coords)) + self.mask_token  # (B, Nq, D_pred)
        x = torch.cat([ctx_tok, tgt_tok], dim=1)
        for blk in self.blocks:
            x = blk(x)
        x = self.norm(x[:, K:])
        return self.proj_out(x)
