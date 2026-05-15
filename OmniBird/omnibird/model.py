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

from .attention import (
    MultiHeadAttention, BigBirdSparseAttention,
    GroupedSparseAttention, CrossAttention,
)
from .serialization import (
    precompute_grid_orderings, quantize_coords, invert_perm,
)


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


class NerfEmbedder(nn.Module):
    """NeRF-style frequency encoding (matches FM4NPP):

        γ(x) = [x, sin(2^0 x), cos(2^0 x), sin(2^1 x), cos(2^1 x),
                  ..., sin(2^{L-1} x), cos(2^{L-1} x)]

    Output dim per input dim: (1 if include_input else 0) + 2 * num_freqs.
    """

    def __init__(self, coord_dim: int, num_freqs: int = 10,
                 max_freq_log2: Optional[int] = None,
                 include_input: bool = True, log_sampling: bool = True):
        super().__init__()
        self.coord_dim     = coord_dim
        self.num_freqs     = num_freqs
        self.include_input = include_input
        if max_freq_log2 is None:
            max_freq_log2 = num_freqs - 1
        if log_sampling:
            freqs = 2.0 ** torch.linspace(0., max_freq_log2, num_freqs)
        else:
            freqs = torch.linspace(2.0**0., 2.0**max_freq_log2, num_freqs)
        self.register_buffer("freqs", freqs, persistent=False)
        feats_per_dim = (1 if include_input else 0) + 2 * num_freqs
        self.out_dim = feats_per_dim * coord_dim

    def forward(self, x):
        # x: (..., coord_dim) → (..., out_dim) with FM4NPP component ordering:
        # [raw_x, sin(f0)x, cos(f0)x, sin(f1)x, cos(f1)x, ...]
        scaled = x.unsqueeze(-2) * self.freqs.view(-1, 1)       # (..., L, D)
        sins = torch.sin(scaled)
        coss = torch.cos(scaled)
        sc   = torch.stack([sins, coss], dim=-2)                # (..., L, 2, D)
        sc   = sc.flatten(-3)                                    # (..., L*2*D)
        if self.include_input:
            return torch.cat([x, sc], dim=-1)
        return sc


def _make_pos_embedder(pos_embed: str, coord_dim: int,
                        fourier_dim: int = 96, fourier_scale: float = 15.0,
                        nerf_freqs: int = 10):
    """Returns (embedder_module, output_dim)."""
    if pos_embed == "nerf":
        emb = NerfEmbedder(coord_dim, num_freqs=nerf_freqs)
        return emb, emb.out_dim
    emb = GaussianFourierFeatures(coord_dim, fourier_dim, scale=fourier_scale)
    return emb, 2 * fourier_dim


class Tokenizer(nn.Module):
    """Per-point token = signal_proj(signal) + pos_proj(γ(coord)).

    For 2-D images, signal_dim=3 (RGB) and coord_dim=2 (y, x).
    For event cameras, signal_dim=1 (polarity) and coord_dim=3 (x, y, t).
    """

    def __init__(self, d_model=256, signal_dim=3, coord_dim=2,
                 fourier_dim=96, fourier_scale=15.0,
                 pos_embed: str = "gaussian", nerf_freqs: int = 10):
        super().__init__()
        self.gff, pos_in = _make_pos_embedder(pos_embed, coord_dim,
                                                fourier_dim, fourier_scale, nerf_freqs)
        self.signal_proj = nn.Linear(signal_dim, d_model)
        self.pos_proj = nn.Linear(pos_in, d_model)

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
                 attention_type: str = "bigbird", group_size: int = 16,
                 pool: str = "mean", use_centroid_pos: bool = True,
                 pos_embed: str = "gaussian", nerf_freqs: int = 10):
        super().__init__()
        self.tokenizer = Tokenizer(d_model, signal_dim=signal_dim, coord_dim=coord_dim,
                                    fourier_dim=fourier_dim, fourier_scale=fourier_scale,
                                    pos_embed=pos_embed, nerf_freqs=nerf_freqs)
        self.attention_type = attention_type
        self.group_size = group_size
        # Padding multiple depends on which attention we use.
        if attention_type in ("grouped", "grouped_hierarchical"):
            self.pad_multiple = group_size
        else:
            self.pad_multiple = block_size
        # Construct blocks. Hierarchical needs a callable that maps coords → emb,
        # shared with the tokenizer's pos_proj for parameter efficiency.
        if attention_type == "grouped_hierarchical":
            self.blocks = nn.ModuleList([
                HierarchicalEncoderBlock(
                    d_model, n_heads=n_heads, dim_head=dim_head,
                    group_size=group_size, ffn_mult=ffn_mult,
                    pool=pool, use_centroid_pos=use_centroid_pos,
                    coord_dim=coord_dim, fourier_dim=fourier_dim,
                    fourier_scale=fourier_scale,
                )
                for _ in range(n_layers)
            ])
        else:
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

        # If using HierarchicalEncoderBlock, pad coords to Kp too.
        coords_padded = None
        if any(isinstance(blk, HierarchicalEncoderBlock) for blk in self.blocks):
            if coords.shape[1] != Kp:
                pad_n = Kp - coords.shape[1]
                pad_c = torch.zeros(B, pad_n, coords.shape[-1],
                                     device=coords.device, dtype=coords.dtype)
                coords_padded = torch.cat([coords, pad_c], dim=1)
            else:
                coords_padded = coords

        # Stack of encoder blocks, each picks an order at random
        order_names = list(self.serial_orders)
        for blk in self.blocks:
            name = order_names[torch.randint(0, len(order_names), (1,)).item()]
            perm, inv = extended[name]
            if isinstance(blk, HierarchicalEncoderBlock):
                x = blk(x, perm, inv, pos_emb=pos_emb, coords=coords_padded,
                        key_padding_mask=pm)
            else:
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


# ---------------------------------------------------------------------------
# HierarchicalEncoderBlock — within-group + between-group attention per layer
# ---------------------------------------------------------------------------

class HierarchicalEncoderBlock(nn.Module):
    """Two-level attention block (Swin V2 / PointConT style):

        x = x + within_group_attn(LN(x))                # local, dense within G windows
        g = pool(x, group=G)                            # one summary token per window
        x = x + broadcast(between_group_attn(LN(g)))    # global, dense across N/G group tokens
        x = x + ffn(LN(x))

    Plus optional centroid positional residual added BEFORE both attentions:
        x = x + pos_proj(γ(group_centroid))

    Used by OmniBirdEncoder when attention_type='grouped_hierarchical'.
    """

    def __init__(self, dim, n_heads=8, dim_head=32, group_size=16, ffn_mult=4,
                 pool: str = "mean", use_centroid_pos: bool = True,
                 coord_dim: int = 3, fourier_dim: int = 96,
                 fourier_scale: float = 15.0):
        super().__init__()
        self.G = group_size
        self.pool = pool
        self.use_centroid_pos = use_centroid_pos
        # Own the centroid projection as proper sub-modules so DataParallel
        # replicates them onto each device. A lambda closing over an outer
        # module would keep a reference to the master replica on cuda:0.
        if use_centroid_pos:
            self.cent_gff  = GaussianFourierFeatures(coord_dim, fourier_dim,
                                                     scale=fourier_scale)
            self.cent_proj = nn.Linear(2 * fourier_dim, dim)
        else:
            self.cent_gff  = None
            self.cent_proj = None

        self.norm_w = nn.LayerNorm(dim)
        self.within = GroupedSparseAttention(dim, n_heads=n_heads, dim_head=dim_head,
                                              group_size=group_size)
        self.norm_b = nn.LayerNorm(dim)
        self.between = MultiHeadAttention(dim, n_heads=n_heads, dim_head=dim_head)
        self.norm_ffn = nn.LayerNorm(dim)
        self.ffn = FeedForward(dim, mult=ffn_mult)
        self.required_multiple = group_size

    def forward(self, x, perm, inverse_perm, pos_emb=None, coords=None,
                key_padding_mask=None):
        # Per-token positional residual (existing behavior in OmniBirdEncoder)
        if pos_emb is not None:
            x = x + pos_emb

        x_p = _gather_along_seq(x, perm)
        pm_p = _gather_mask(key_padding_mask, perm) if key_padding_mask is not None else None

        # ── Optional: per-group centroid positional residual ───────────────
        if self.use_centroid_pos and coords is not None and self.cent_proj is not None:
            coords_p = _gather_along_seq(coords, perm)               # (B, N, D_c)
            B, N, D_c = coords_p.shape
            G = self.G
            NG = N // G
            cent = coords_p.view(B, NG, G, D_c).mean(dim=2)           # (B, NG, D_c)
            cent_bcast = cent.unsqueeze(2).expand(B, NG, G, D_c).reshape(B, N, D_c)
            x_p = x_p + self.cent_proj(self.cent_gff(cent_bcast))

        # ── Step 1: within-group attention ──────────────────────────────
        x_p = x_p + self.within(self.norm_w(x_p), key_padding_mask=pm_p)

        # ── Step 2: between-group attention ─────────────────────────────
        B, N, D = x_p.shape
        G = self.G
        NG = N // G
        if self.pool == "mean":
            g_tok = x_p.view(B, NG, G, D).mean(dim=2)
        elif self.pool == "max":
            g_tok = x_p.view(B, NG, G, D).max(dim=2).values
        else:
            g_tok = x_p.view(B, NG, G, D).mean(dim=2)
        # Group key-padding mask: group is "padded" only if ALL its events are padded
        g_kpm = pm_p.view(B, NG, G).all(dim=2) if pm_p is not None else None
        delta_between = self.between(self.norm_b(g_tok), key_padding_mask=g_kpm)
        broadcast = delta_between.unsqueeze(2).expand(B, NG, G, D).reshape(B, N, D)
        x_p = x_p + broadcast

        # ── FFN ────────────────────────────────────────────────────────
        x_p = x_p + self.ffn(self.norm_ffn(x_p))

        return _gather_along_seq(x_p, inverse_perm)


# ---------------------------------------------------------------------------
# PerceiverPredictor — few group-level queries cross-attend to context
# ---------------------------------------------------------------------------

class PerceiverPredictor(nn.Module):
    """Cross-attention predictor with N_q group-level queries.

    For OmniBird-JEPA's per-group prediction:
      - Queries are 4 target-block centroids (or N_q in general).
      - Keys/values are the long context-encoder feature sequence.
    Compute scales as O(N_q · K_ctx · D) per layer — cheap when N_q ≪ K_ctx.

    Each layer:  cross-attn(q ← ctx) → self-attn(q) → ffn
    """

    def __init__(self, d_model=256, d_pred=192, n_layers=4, n_heads=6, dim_head=32,
                 coord_dim=3, fourier_dim=96, fourier_scale=15.0, ffn_mult=4,
                 pos_symmetric: bool = True,
                 pos_embed: str = "gaussian", nerf_freqs: int = 10):
        super().__init__()
        self.proj_in    = nn.Linear(d_model, d_pred)
        self.gff, pos_in = _make_pos_embedder(pos_embed, coord_dim,
                                                fourier_dim, fourier_scale, nerf_freqs)
        self.proj_pos   = nn.Linear(pos_in, d_pred)
        self.mask_token = nn.Parameter(torch.zeros(1, 1, d_pred))
        nn.init.trunc_normal_(self.mask_token, std=0.02)
        self.pos_symmetric = pos_symmetric

        # Stack of (cross-attn, self-attn, ffn) layers
        self.layers = nn.ModuleList()
        for _ in range(n_layers):
            self.layers.append(nn.ModuleDict({
                "norm_q":   nn.LayerNorm(d_pred),
                "norm_kv":  nn.LayerNorm(d_pred),
                "cross":    CrossAttention(d_pred, n_heads=n_heads, dim_head=dim_head),
                "norm_s":   nn.LayerNorm(d_pred),
                "self_attn": MultiHeadAttention(d_pred, n_heads=n_heads, dim_head=dim_head),
                "norm_f":   nn.LayerNorm(d_pred),
                "ffn":      FeedForward(d_pred, mult=ffn_mult),
            }))
        self.norm     = nn.LayerNorm(d_pred)
        self.proj_out = nn.Linear(d_pred, d_model)

    def forward(self, ctx_feat, query_coords, ctx_coords=None, ctx_key_padding_mask=None):
        """
        ctx_feat:     (B, K_ctx, D_model)
        query_coords: (B, N_q,   coord_dim)
        ctx_coords:   (B, K_ctx, coord_dim)  — required if pos_symmetric=True
        Returns: h_pred (B, N_q, D_model)
        """
        B = ctx_feat.size(0)
        ctx_tok = self.proj_in(ctx_feat)
        if self.pos_symmetric:
            assert ctx_coords is not None, "pos_symmetric=True needs ctx_coords"
            ctx_tok = ctx_tok + self.proj_pos(self.gff(ctx_coords))

        # Initialize each query with its γ(coord) + mask_token
        q = self.proj_pos(self.gff(query_coords)) + self.mask_token

        for blk in self.layers:
            # Cross-attn: q ← ctx_tok
            q = q + blk["cross"](
                blk["norm_q"](q),
                blk["norm_kv"](ctx_tok),
                key_padding_mask=ctx_key_padding_mask,
            )
            # Self-attn among queries (cheap, e.g. 4×4)
            q = q + blk["self_attn"](blk["norm_s"](q))
            # FFN
            q = q + blk["ffn"](blk["norm_f"](q))

        return self.proj_out(self.norm(q))


# ---------------------------------------------------------------------------
# Patchifier — mini-PointNet over per-patch event chunks
# ---------------------------------------------------------------------------

class Patchifier(nn.Module):
    """Mini-PointNet that maps a chunk of K events → 1 D-dim token.

    Input per patch is (K, 4): K events × (x, y, t, polarity-or-onehot).
    For each event we compute:
        rel_coord  = coord - patch_centroid             # local frame
        rel_pos_emb = γ(rel_coord)
        feat       = MLP_event([signal ⊕ rel_pos_emb])  # per-event embedding
    Then max-pool over K to one D-dim token, refine with a second MLP.

    The patch's *absolute* position is reintroduced by the caller via
    pos_proj(γ(patch_centroid)) so the token has both local-shape and
    global-position information.
    """

    def __init__(self, signal_dim, coord_dim, d_model,
                 fourier_dim=96, fourier_scale=15.0, hidden=None):
        super().__init__()
        hidden = hidden or d_model
        self.gff = GaussianFourierFeatures(coord_dim, fourier_dim, scale=fourier_scale)
        # Per-event MLP: signal + γ(rel_coord) → hidden
        per_event_in = signal_dim + 2 * fourier_dim
        self.mlp_event = nn.Sequential(
            nn.Linear(per_event_in, hidden), nn.GELU(),
            nn.Linear(hidden, hidden),
        )
        # Post-pool refine
        self.mlp_patch = nn.Sequential(
            nn.Linear(hidden, hidden), nn.GELU(),
            nn.Linear(hidden, d_model),
        )

    def forward(self, patch_events, patch_centroids, kpm_per_event=None):
        """
        patch_events:    (B, P, K, signal_dim + coord_dim)
        patch_centroids: (B, P, coord_dim)
        kpm_per_event:   (B, P, K) bool — True at padded events; sets their
                         contribution to -inf in max-pool.
        Returns: tokens (B, P, D_model)
        """
        coord_dim = patch_centroids.shape[-1]
        coords  = patch_events[..., :coord_dim]                          # (B, P, K, D_c)
        signal  = patch_events[..., coord_dim:]                          # (B, P, K, signal_dim)

        rel = coords - patch_centroids.unsqueeze(2)                       # local-frame coords
        rel_emb = self.gff(rel)                                           # (B, P, K, 2*fourier)

        feat = torch.cat([signal, rel_emb], dim=-1)                       # (B, P, K, S + 2F)
        feat = self.mlp_event(feat)                                       # (B, P, K, hidden)

        if kpm_per_event is not None:
            # Mask out padded events from max-pool
            feat = feat.masked_fill(kpm_per_event.unsqueeze(-1), float("-inf"))

        pooled = feat.max(dim=2).values                                   # (B, P, hidden)
        # Replace any -inf rows (all-padding patch) with zeros to keep grads clean
        pooled = torch.where(torch.isinf(pooled), torch.zeros_like(pooled), pooled)
        return self.mlp_patch(pooled)                                     # (B, P, D_model)


# ---------------------------------------------------------------------------
# PatchOmniBirdEncoder — runs on patch tokens (not events)
# ---------------------------------------------------------------------------

class PatchOmniBirdEncoder(nn.Module):
    """Encoder that consumes pre-organized event patches and runs attention on
    the resulting patch tokens (Point-MAE / Point-BERT style).

    Inputs at forward time:
        patch_events:    (B, P, K, signal_dim + coord_dim)
        patch_centroids: (B, P, coord_dim)
        patch_kpm:       (B, P) bool — True for all-padding patches
        kpm_per_event:   (B, P, K) bool — True for padded events inside patches (optional)

    Output:
        per-patch features (B, P, D_model).
    """

    def __init__(self, d_model=256, n_layers=6, n_heads=8, dim_head=32,
                 ffn_mult=4, signal_dim=1, coord_dim=3,
                 fourier_dim=96, fourier_scale=15.0,
                 patch_size: int = 32):
        super().__init__()
        self.patchify = Patchifier(signal_dim, coord_dim, d_model,
                                    fourier_dim=fourier_dim,
                                    fourier_scale=fourier_scale)
        self.gff_abs = GaussianFourierFeatures(coord_dim, fourier_dim, scale=fourier_scale)
        self.abs_pos_proj = nn.Linear(2 * fourier_dim, d_model)

        # Dense transformer over patch tokens (P typically 64-256, dense is fine)
        self.blocks = nn.ModuleList()
        for _ in range(n_layers):
            self.blocks.append(nn.ModuleDict({
                "norm1": nn.LayerNorm(d_model),
                "attn":  MultiHeadAttention(d_model, n_heads=n_heads, dim_head=dim_head),
                "norm2": nn.LayerNorm(d_model),
                "ffn":   FeedForward(d_model, mult=ffn_mult),
            }))
        self.norm = nn.LayerNorm(d_model)
        self.patch_size = patch_size

    def forward(self, patch_events, patch_centroids, patch_kpm=None,
                kpm_per_event=None):
        # mini-PointNet → patch tokens
        tokens = self.patchify(patch_events, patch_centroids, kpm_per_event=kpm_per_event)
        # Absolute-position residual
        tokens = tokens + self.abs_pos_proj(self.gff_abs(patch_centroids))

        for blk in self.blocks:
            tokens = tokens + blk["attn"](blk["norm1"](tokens),
                                            key_padding_mask=patch_kpm)
            tokens = tokens + blk["ffn"](blk["norm2"](tokens))
        return self.norm(tokens)


# ---------------------------------------------------------------------------
# CentroidPool — cross-attention pool with data-conditional centroid queries
# ---------------------------------------------------------------------------

class CentroidPool(nn.Module):
    """One cross-attention block where per-instance centroids are the queries
    and per-event features are the keys/values. Produces one latent per
    centroid. The centroid itself is encoded only through its coordinate
    (Gaussian Fourier features), making the pool position-aware but
    data-conditional via the centroid positions chosen per sample.
    """

    def __init__(self, d_model: int, n_heads: int = 8, dim_head: int = 32,
                 coord_dim: int = 3, fourier_dim: int = 96,
                 fourier_scale: float = 15.0, ffn_mult: int = 4,
                 pos_embed: str = "gaussian", nerf_freqs: int = 10):
        super().__init__()
        self.gff_q, pos_in = _make_pos_embedder(pos_embed, coord_dim,
                                                  fourier_dim, fourier_scale, nerf_freqs)
        self.q_proj  = nn.Linear(pos_in, d_model)
        self.norm_q  = nn.LayerNorm(d_model)
        self.norm_kv = nn.LayerNorm(d_model)
        self.cross   = CrossAttention(d_model, n_heads=n_heads, dim_head=dim_head)
        self.norm_f  = nn.LayerNorm(d_model)
        self.ffn     = FeedForward(d_model, mult=ffn_mult)
        self.norm    = nn.LayerNorm(d_model)

    def forward(self, event_feat, centroids, event_kpm=None):
        # event_feat: (B, N, D); centroids: (B, P, coord_dim); event_kpm: (B, N)
        q = self.q_proj(self.gff_q(centroids))
        q = q + self.cross(self.norm_q(q), self.norm_kv(event_feat),
                            key_padding_mask=event_kpm)
        q = q + self.ffn(self.norm_f(q))
        return self.norm(q)


# ---------------------------------------------------------------------------
# BigBirdEventEncoderWithPool — BigBird per-event encoder + CentroidPool
# ---------------------------------------------------------------------------

class BigBirdEventEncoderWithPool(nn.Module):
    """Per-event BigBird sparse encoder followed by a cross-attention pool
    that turns event features into one latent per data-conditional centroid.

    Designed so a single forward pass takes (events, centroids, event_kpm) and
    returns (B, P, D) — clean to wrap in DataParallel.
    """

    def __init__(self, d_model: int = 256, n_layers: int = 6,
                 n_heads: int = 8, dim_head: int = 32,
                 block_size: int = 8, window: int = 1,
                 n_random: int = 2, n_global: int = 2,
                 ffn_mult: int = 4,
                 signal_dim: int = 2, coord_dim: int = 3,
                 fourier_dim: int = 96, fourier_scale: float = 15.0,
                 serial_orders=("z", "z_rev", "hilbert", "hilbert_rev"),
                 reinject_pos: bool = False,
                 side: int = 64,
                 pos_embed: str = "nerf", nerf_freqs: int = 10):
        super().__init__()
        self.coord_dim     = coord_dim
        self.side          = side
        self.serial_orders = tuple(serial_orders)
        self.encoder = OmniBirdEncoder(
            d_model=d_model, n_layers=n_layers,
            n_heads=n_heads, dim_head=dim_head,
            block_size=block_size, window=window,
            n_random=n_random, n_global=n_global,
            ffn_mult=ffn_mult,
            signal_dim=signal_dim, coord_dim=coord_dim,
            fourier_dim=fourier_dim, fourier_scale=fourier_scale,
            serial_orders=serial_orders,
            reinject_pos=reinject_pos,
            attention_type="bigbird",
            pos_embed=pos_embed, nerf_freqs=nerf_freqs,
        )
        self.pool = CentroidPool(
            d_model=d_model,
            n_heads=n_heads, dim_head=dim_head,
            coord_dim=coord_dim,
            fourier_dim=fourier_dim, fourier_scale=fourier_scale,
            ffn_mult=ffn_mult,
            pos_embed=pos_embed, nerf_freqs=nerf_freqs,
        )
        for name, ranks in precompute_grid_orderings(side, ndim=coord_dim).items():
            self.register_buffer(f"_rank_{name}", ranks, persistent=False)

    def _orderings(self, coords, event_kpm):
        B, N, _ = coords.shape
        cell_ids = quantize_coords(coords, side=self.side, value_range=(-1.0, 1.0))
        tail_shift = event_kpm.long() * (N + 1) if event_kpm is not None else 0
        out = {}
        for name in self.serial_orders:
            ranks = getattr(self, f"_rank_{name}")[cell_ids]
            eff = ranks + tail_shift
            perm = eff.argsort(dim=-1)
            inv  = invert_perm(perm)
            out[name] = {"perm": perm, "inverse": inv}
        return out

    def forward(self, events, centroids, event_kpm=None):
        coords = events[..., :self.coord_dim]
        signal = events[..., self.coord_dim:]
        orderings = self._orderings(coords, event_kpm)
        event_feat = self.encoder(signal, coords, orderings, key_padding_mask=event_kpm)
        return self.pool(event_feat, centroids, event_kpm=event_kpm)
