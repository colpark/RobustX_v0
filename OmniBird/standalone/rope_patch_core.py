"""RoPE Patch JEPA — standalone, copy-paste portable.

Replaces the mini-PointNet patch aggregator with a Non-Uniform DFT
("Rotary patch aggregation"). Each patch's K events are treated as
samples of a complex-valued field; the patch summary is the truncated
NUDFT of the content sequence at log-spaced frequencies tied to channel
pairs (standard RoPE structure, lifted from sequence positions to
continuous spatial positions inside a patch).

Two-level RoPE:
  Level 1 (within-patch aggregation): rotate each event's content by
          its relative position to the patch centroid, then sum.
  Level 2 (cross-patch attention): rotate Q and K by patch centroids
          inside each attention layer. The Level-1 and Level-2 phases
          cancel exactly, recovering absolute-position event-relative
          attention while preserving the modeling benefits of locality.
"""
from __future__ import annotations
from typing import Optional
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from vit_fps_core import NerfPosEnc, FeedForward


# ===========================================================================
# RoPE Patchifier — within-patch level
# ===========================================================================

class RoPEPatchifier(nn.Module):
    """Aggregate K events per patch into a single d_model token via axial
    RoPE applied at the d_model level (Non-Uniform DFT).

        S = Σ_i  signal_proj(c_i) · exp(j · ω · rel_pos_i)

    where rel_pos_i = position_i − centroid, and frequencies ω vary across
    channel pairs (RoPE-style log-spaced). Output is permutation-invariant
    in K because the sum is symmetric.
    """
    def __init__(self, signal_dim: int, coord_dim: int, d_model: int,
                 base: float = 100.0, agg: str = "mean"):
        super().__init__()
        assert d_model % (2 * coord_dim) == 0, (
            f"d_model={d_model} must be divisible by 2*coord_dim={2*coord_dim}"
        )
        self.coord_dim = coord_dim
        self.d_model = d_model
        self.channels_per_axis = d_model // coord_dim
        self.pairs_per_axis = self.channels_per_axis // 2
        self.agg = agg

        # Content projection: signal → d_model (real-valued)
        self.signal_proj = nn.Sequential(
            nn.Linear(signal_dim, d_model), nn.GELU(),
            nn.Linear(d_model, d_model),
        )
        # Optional output mixing (helps real-vs-imag channel mixing)
        self.out_proj = nn.Sequential(
            nn.Linear(d_model, d_model), nn.GELU(),
            nn.Linear(d_model, d_model),
        )

        # Log-spaced inverse frequencies — one per channel pair (RoPE-style)
        inv_freq = base ** (
            -torch.arange(self.pairs_per_axis).float() * 2 / self.channels_per_axis
        )
        self.register_buffer("inv_freq", inv_freq)

    def forward(self, patch_events: torch.Tensor,
                patch_centroids: torch.Tensor,
                event_kpm: Optional[torch.Tensor] = None) -> torch.Tensor:
        """patch_events: (B, P, K, coord_dim + signal_dim) — RAW coord + signal.
        patch_centroids: (B, P, coord_dim).
        event_kpm: (B, P, K) True at padded.
        Returns: (B, P, d_model).
        """
        coord_dim = self.coord_dim
        coords = patch_events[..., :coord_dim]               # (B, P, K, coord_dim)
        signal = patch_events[..., coord_dim:]               # (B, P, K, signal_dim)
        rel = coords - patch_centroids.unsqueeze(2)          # (B, P, K, coord_dim)

        # Project content to d_model (real-valued)
        z = self.signal_proj(signal)                         # (B, P, K, d_model)
        B, P, K, _ = z.shape

        # Reshape into (axes, pairs, 2) for axial RoPE rotation
        z = z.view(B, P, K, coord_dim, self.pairs_per_axis, 2)

        # Per-(event, axis, pair) angles. rel: (B,P,K,axes); inv_freq: (pairs,)
        angles = rel.unsqueeze(-1) * self.inv_freq            # (B,P,K,axes,pairs)
        cos_a = angles.cos().unsqueeze(-1)                    # (B,P,K,axes,pairs,1)
        sin_a = angles.sin().unsqueeze(-1)

        z_even = z[..., 0:1]
        z_odd = z[..., 1:2]
        z_rot_even = z_even * cos_a - z_odd * sin_a
        z_rot_odd = z_even * sin_a + z_odd * cos_a
        z_rot = torch.cat([z_rot_even, z_rot_odd], dim=-1)    # (B,P,K,axes,pairs,2)
        z_rot = z_rot.reshape(B, P, K, self.d_model)

        if event_kpm is not None:
            z_rot = z_rot.masked_fill(event_kpm.unsqueeze(-1), 0.0)

        if self.agg == "sum":
            summary = z_rot.sum(dim=2)
        else:
            if event_kpm is not None:
                count = (~event_kpm).float().sum(dim=-1, keepdim=True).clamp(min=1)
                summary = z_rot.sum(dim=2) / count
            else:
                summary = z_rot.mean(dim=2)

        return self.out_proj(summary)


# ===========================================================================
# Centroid-RoPE attention — cross-patch level
# ===========================================================================

class CentroidRoPEMultiHeadAttention(nn.Module):
    """Self-attention with axial RoPE applied to Q and K using each token's
    centroid coordinate. This is the "level 2" rotation that cancels the
    parasitic phase introduced by the within-patch RoPE aggregation.
    """
    def __init__(self, dim: int, n_heads: int = 8, dim_head: int = 32,
                 coord_dim: int = 2, base: float = 100.0):
        super().__init__()
        assert dim_head % (2 * coord_dim) == 0, (
            f"dim_head={dim_head} must be divisible by 2*coord_dim={2*coord_dim}"
        )
        self.n_heads = n_heads
        self.dim_head = dim_head
        self.scale = dim_head ** -0.5
        self.coord_dim = coord_dim
        self.channels_per_axis = dim_head // coord_dim
        self.pairs_per_axis = self.channels_per_axis // 2

        inner = n_heads * dim_head
        self.to_qkv = nn.Linear(dim, inner * 3, bias=False)
        self.to_out = nn.Linear(inner, dim)

        inv_freq = base ** (
            -torch.arange(self.pairs_per_axis).float() * 2 / self.channels_per_axis
        )
        self.register_buffer("inv_freq", inv_freq)

    def _rope(self, x: torch.Tensor, centroids: torch.Tensor) -> torch.Tensor:
        """x: (B, H, N, Dh); centroids: (B, N, coord_dim) → (B, H, N, Dh)."""
        B, H, N, Dh = x.shape
        coord_dim = self.coord_dim
        x = x.view(B, H, N, coord_dim, self.pairs_per_axis, 2)
        # angles: (B, 1, N, axes, pairs)
        angles = centroids.unsqueeze(1).unsqueeze(-1) * self.inv_freq
        cos_a = angles.cos().unsqueeze(-1)
        sin_a = angles.sin().unsqueeze(-1)
        x_even = x[..., 0:1]
        x_odd = x[..., 1:2]
        x_rot_even = x_even * cos_a - x_odd * sin_a
        x_rot_odd = x_even * sin_a + x_odd * cos_a
        x_rot = torch.cat([x_rot_even, x_rot_odd], dim=-1)
        return x_rot.reshape(B, H, N, Dh)

    def forward(self, x: torch.Tensor, centroids: torch.Tensor) -> torch.Tensor:
        B, N, _ = x.shape
        qkv = self.to_qkv(x).reshape(B, N, 3, self.n_heads, self.dim_head)
        q, k, v = qkv.permute(2, 0, 3, 1, 4).unbind(0)   # each (B, H, N, Dh)
        q = self._rope(q, centroids)
        k = self._rope(k, centroids)
        scores = torch.einsum("bhnd,bhmd->bhnm", q, k) * self.scale
        attn = F.softmax(scores, dim=-1)
        out = torch.einsum("bhnm,bhmd->bhnd", attn, v).transpose(1, 2).reshape(B, N, -1)
        return self.to_out(out)


class RoPETransformerBlock(nn.Module):
    def __init__(self, dim: int, n_heads: int = 8, dim_head: int = 32,
                 ffn_mult: int = 4, coord_dim: int = 2, base: float = 100.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = CentroidRoPEMultiHeadAttention(
            dim, n_heads=n_heads, dim_head=dim_head,
            coord_dim=coord_dim, base=base,
        )
        self.norm2 = nn.LayerNorm(dim)
        self.ffn = FeedForward(dim, mult=ffn_mult)

    def forward(self, x: torch.Tensor, centroids: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.norm1(x), centroids)
        return x + self.ffn(self.norm2(x))


# ===========================================================================
# Full encoder + predictor
# ===========================================================================

class RoPEViTEncoder(nn.Module):
    """RoPE patchifier + ViT blocks with centroid-RoPE attention.

    Two-level RoPE means the parasitic centroid-offset phase introduced by
    the within-patch aggregation cancels exactly under the centroid-RoPE
    attention. Output patch tokens are equivalent (up to learned mixing)
    to those produced by absolute-position RoPE aggregation.
    """
    def __init__(self, signal_dim: int, coord_dim: int, d_model: int = 256,
                 n_layers: int = 6, n_heads: int = 8, dim_head: int = 32,
                 ffn_mult: int = 4, base_within: float = 100.0,
                 base_cross: float = 100.0,
                 add_nerf_centroid: bool = False, n_freqs: int = 10):
        super().__init__()
        self.patchify = RoPEPatchifier(signal_dim, coord_dim, d_model, base=base_within)
        self.add_nerf_centroid = add_nerf_centroid
        if add_nerf_centroid:
            self.pos_enc = NerfPosEnc(coord_dim, n_freqs=n_freqs, include_input=True)
            self.pos_proj = nn.Linear(self.pos_enc.out_dim, d_model)
        self.blocks = nn.ModuleList([
            RoPETransformerBlock(
                d_model, n_heads=n_heads, dim_head=dim_head,
                ffn_mult=ffn_mult, coord_dim=coord_dim, base=base_cross,
            )
            for _ in range(n_layers)
        ])
        self.norm = nn.LayerNorm(d_model)

    def forward(self, patch_events: torch.Tensor, patch_centroids: torch.Tensor,
                event_kpm: Optional[torch.Tensor] = None) -> torch.Tensor:
        x = self.patchify(patch_events, patch_centroids, event_kpm=event_kpm)
        if self.add_nerf_centroid:
            x = x + self.pos_proj(self.pos_enc(patch_centroids))
        for blk in self.blocks:
            x = blk(x, patch_centroids)
        return self.norm(x)


class RoPEViTPredictor(nn.Module):
    """Dense predictor with centroid-RoPE attention. Mask tokens are placed
    at target centroids; centroid-RoPE attention provides position-aware
    interaction with context tokens.
    """
    def __init__(self, d_model: int = 256, d_pred: int = 192,
                 n_layers: int = 4, n_heads: int = 6, dim_head: int = 32,
                 ffn_mult: int = 4, coord_dim: int = 2, base: float = 100.0,
                 add_nerf: bool = True, n_freqs: int = 10):
        super().__init__()
        self.proj_in = nn.Linear(d_model, d_pred)
        self.mask_token = nn.Parameter(torch.zeros(1, 1, d_pred))
        nn.init.trunc_normal_(self.mask_token, std=0.02)
        self.add_nerf = add_nerf
        if add_nerf:
            self.pos_enc = NerfPosEnc(coord_dim, n_freqs=n_freqs, include_input=True)
            self.pos_proj = nn.Linear(self.pos_enc.out_dim, d_pred)
        self.blocks = nn.ModuleList([
            RoPETransformerBlock(
                d_pred, n_heads=n_heads, dim_head=dim_head,
                ffn_mult=ffn_mult, coord_dim=coord_dim, base=base,
            )
            for _ in range(n_layers)
        ])
        self.norm = nn.LayerNorm(d_pred)
        self.proj_out = nn.Linear(d_pred, d_model)

    def forward(self, ctx_feat: torch.Tensor, target_coords: torch.Tensor,
                ctx_coords: torch.Tensor) -> torch.Tensor:
        B, K_ctx, _ = ctx_feat.shape
        K_tgt = target_coords.size(1)
        ctx_tok = self.proj_in(ctx_feat)
        tgt_tok = self.mask_token.expand(B, K_tgt, -1).contiguous()
        if self.add_nerf:
            ctx_tok = ctx_tok + self.pos_proj(self.pos_enc(ctx_coords))
            tgt_tok = tgt_tok + self.pos_proj(self.pos_enc(target_coords))
        x = torch.cat([ctx_tok, tgt_tok], dim=1)
        coords_all = torch.cat([ctx_coords, target_coords], dim=1)
        for blk in self.blocks:
            x = blk(x, coords_all)
        x = self.norm(x[:, K_ctx:])
        return self.proj_out(x)


# ===========================================================================
# Pure-function helpers for visualization
# ===========================================================================

def rope_aggregate_complex(
    content: torch.Tensor, pos: torch.Tensor, freq: float
) -> torch.Tensor:
    """Visualization helper: aggregate K complex contents by RoPE rotation.

    Treats one channel-pair (real, imag) and one axis. Used in the
    visualization notebook to build intuition.

    content: (..., K, 2)  — (real, imag) per event
    pos:     (..., K)     — scalar position per event
    freq:    scalar       — single RoPE frequency
    Returns: (..., 2)     — summed complex (real, imag)
    """
    ang = pos * freq                            # (..., K)
    cos_a = ang.cos().unsqueeze(-1)             # (..., K, 1)
    sin_a = ang.sin().unsqueeze(-1)
    re = content[..., 0:1] * cos_a - content[..., 1:2] * sin_a
    im = content[..., 0:1] * sin_a + content[..., 1:2] * cos_a
    rot = torch.cat([re, im], dim=-1)           # (..., K, 2)
    return rot.sum(dim=-2)                      # (..., 2)
