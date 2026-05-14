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

from .attention import MultiHeadAttention, BigBirdSparseAttention


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
    """Per-point token = signal_proj(rgb) + γ_proj(γ(coord))."""

    def __init__(self, d_model=256, rgb_channels=3, fourier_dim=96, fourier_scale=15.0):
        super().__init__()
        self.gff = GaussianFourierFeatures(2, fourier_dim, scale=fourier_scale)
        self.signal_proj = nn.Linear(rgb_channels, d_model)
        self.pos_proj = nn.Linear(2 * fourier_dim, d_model)

    def forward(self, pixels, coords):
        """pixels: (B, K, 3); coords: (B, K, 2) → tokens (B, K, D)."""
        return self.signal_proj(pixels) + self.pos_proj(self.gff(coords))


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
    """One block: order-permute → BigBird sparse self-attn → FFN → un-permute."""

    def __init__(self, dim, n_heads=8, dim_head=32, block_size=32,
                 window=1, n_random=2, n_global=2, ffn_mult=4):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = BigBirdSparseAttention(
            dim, n_heads=n_heads, dim_head=dim_head,
            block_size=block_size, window=window,
            n_random=n_random, n_global=n_global,
        )
        self.norm2 = nn.LayerNorm(dim)
        self.ffn = FeedForward(dim, mult=ffn_mult)

    def forward(self, x, perm, inverse_perm, key_padding_mask=None):
        """x: (B, N, D)  N divisible by block_size
        perm:         (B, N) — permutes original→sorted-by-curve
        inverse_perm: (B, N) — inverse of perm
        key_padding_mask: (B, N) bool in *original* order
        """
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

class PBBEncoder(nn.Module):
    """Tokenizer → L stacked EncoderBlocks (each with a randomly chosen ordering)."""

    def __init__(self, d_model=256, n_layers=6, n_heads=8, dim_head=32,
                 block_size=32, window=1, n_random=2, n_global=2,
                 ffn_mult=4, rgb_channels=3, fourier_dim=96, fourier_scale=15.0,
                 serial_orders=("z", "z_rev", "hilbert", "hilbert_rev")):
        super().__init__()
        self.tokenizer = Tokenizer(d_model, rgb_channels, fourier_dim, fourier_scale)
        self.blocks = nn.ModuleList([
            EncoderBlock(d_model, n_heads=n_heads, dim_head=dim_head,
                         block_size=block_size, window=window,
                         n_random=n_random, n_global=n_global,
                         ffn_mult=ffn_mult)
            for _ in range(n_layers)
        ])
        self.norm = nn.LayerNorm(d_model)
        self.serial_orders = tuple(serial_orders)
        self.block_size = block_size

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
        pixels: torch.Tensor,
        coords: torch.Tensor,
        orderings: Dict[str, Dict[str, torch.Tensor]],
        key_padding_mask: Optional[torch.Tensor] = None,
    ):
        """
        pixels: (B, K, 3); coords: (B, K, 2)
        orderings: dict[order_name] -> dict with 'perm' and 'inverse' tensors (B, K).
                   May include extra orderings for `K' > K` padded positions:
                   if perm/inverse is shorter than K' we extend with identity for pads.

        Returns:
            features (B, K, D) — un-permuted, in input order; padding positions
            already filtered out.
        """
        # Tokenize
        x = self.tokenizer(pixels, coords)                  # (B, K, D)
        x, pm, K_orig = self._pad_to_block_multiple(x, self.block_size, key_padding_mask)
        B, Kp, D = x.shape

        # Extend orderings to padded length: pad indices stay at the end in identity order
        extended = {}
        for name, d in orderings.items():
            p = d["perm"]; inv = d["inverse"]
            if p.shape[1] != Kp:
                # Pad positions stay where they are (identity at tail)
                tail = torch.arange(p.shape[1], Kp, device=p.device).unsqueeze(0).expand(B, -1)
                p = torch.cat([p, tail], dim=1)
                inv = torch.cat([inv, tail], dim=1)
            extended[name] = (p, inv)

        # Stack of encoder blocks, each picks an order at random
        order_names = list(self.serial_orders)
        for blk in self.blocks:
            name = order_names[torch.randint(0, len(order_names), (1,)).item()]
            perm, inv = extended[name]
            x = blk(x, perm, inv, key_padding_mask=pm)

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


class PBBPredictor(nn.Module):
    """Dense Transformer that takes encoder context tokens + mask tokens at
    target coordinates and predicts target features.
    """

    def __init__(self, d_model=256, d_pred=192, n_layers=4, n_heads=6, dim_head=32,
                 fourier_dim=96, fourier_scale=15.0, ffn_mult=4):
        super().__init__()
        self.proj_in   = nn.Linear(d_model, d_pred)
        self.gff       = GaussianFourierFeatures(2, fourier_dim, scale=fourier_scale)
        self.proj_pos  = nn.Linear(2 * fourier_dim, d_pred)
        self.mask_token = nn.Parameter(torch.zeros(1, 1, d_pred))
        nn.init.trunc_normal_(self.mask_token, std=0.02)
        self.blocks = nn.ModuleList([
            PredictorBlock(d_pred, n_heads=n_heads, dim_head=dim_head, ffn_mult=ffn_mult)
            for _ in range(n_layers)
        ])
        self.norm = nn.LayerNorm(d_pred)
        self.proj_out = nn.Linear(d_pred, d_model)

    def forward(self, ctx_feat, target_coords):
        """
        ctx_feat:      (B, K_ctx, D_model)
        target_coords: (B, N_tgt, 2)
        Returns: h_pred (B, N_tgt, D_model)
        """
        B, K, _ = ctx_feat.shape
        Nq = target_coords.shape[1]
        ctx_tok = self.proj_in(ctx_feat)                          # (B, K, D_pred)
        tgt_tok = self.proj_pos(self.gff(target_coords)) + self.mask_token  # (B, Nq, D_pred)
        x = torch.cat([ctx_tok, tgt_tok], dim=1)
        for blk in self.blocks:
            x = blk(x)
        x = self.norm(x[:, K:])
        return self.proj_out(x)
