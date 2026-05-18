"""Iterative Cross-Modal Refinement (ICMR) for multimodal OmniBird.

This is the cross-modal fusion layer adapted from the original OmniField
(Cascaded Perceiver IO + ICMR). It sits on top of the per-modality OmniBird
encoders and fuses information across modalities by iteratively cross-attending
a shared latent set to each modality's token stream.

Status: PHASE 2 — single-modality OmniBird already works end-to-end via
`omnibird/encoder.py`. ICMR is provided here as the documented bridge to
the multimodal case (events + RGB). Wire it into a notebook when ready.

Design (from OmniField):
    latents L ∈ R^{N_lat × D}                    # shared learnable set
    for iter in 1..icmr_iters:
        for modality m in present_modalities:
            L = L + cross_attn(L, K=V=g_m)        # m-specific cross-attention
        L = L + self_attn(L)
        L = L + FFN(L)
    return L

Fleximodal: at inference some modalities may be absent. `present_modalities`
is a per-sample boolean mask; the cross-attention is skipped for absent
modalities. The latents still iterate; they just attend to whichever
modalities are available.
"""
from __future__ import annotations

from typing import Dict, List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import repeat

from .attention import MultiHeadAttention
from .model import FeedForward, get_sinusoidal_embeddings


class ModalityCrossAttn(nn.Module):
    """One cross-attention layer from latent queries → modality keys/values."""

    def __init__(self, dim: int, n_heads: int = 8, dim_head: int = 32):
        super().__init__()
        inner = n_heads * dim_head
        self.heads = n_heads
        self.scale = dim_head ** -0.5
        self.norm_q  = nn.LayerNorm(dim)
        self.norm_kv = nn.LayerNorm(dim)
        self.to_q  = nn.Linear(dim, inner, bias=False)
        self.to_kv = nn.Linear(dim, inner * 2, bias=False)
        self.to_out = nn.Linear(inner, dim)

    def forward(self, latents: torch.Tensor, modality_tokens: torch.Tensor,
                key_padding_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """latents (B, N_lat, D); modality_tokens (B, K, D). Returns (B, N_lat, D)."""
        B, NL, D = latents.shape
        K = modality_tokens.shape[1]
        H = self.heads
        Dh = D // H if D % H == 0 else self.to_q.out_features // H

        q = self.to_q(self.norm_q(latents))
        kv = self.to_kv(self.norm_kv(modality_tokens))
        k, v = kv.chunk(2, dim=-1)
        q = q.view(B, NL, H, -1).transpose(1, 2)             # (B, H, NL, Dh)
        k = k.view(B, K,  H, -1).transpose(1, 2)
        v = v.view(B, K,  H, -1).transpose(1, 2)
        scores = (q @ k.transpose(-2, -1)) * self.scale
        if key_padding_mask is not None:
            scores = scores.masked_fill(key_padding_mask.view(B, 1, 1, K), float("-inf"))
        attn = scores.softmax(dim=-1)
        out  = (attn @ v).transpose(1, 2).contiguous().view(B, NL, -1)
        return self.to_out(out)


class ICMR(nn.Module):
    """Iterative Cross-Modal Refinement over a learnable latent set.

    Args:
        n_latents:   size of shared latent set L.
        dim:         latent dim (same as encoder d_model).
        modalities:  tuple of modality names (e.g. ('events', 'rgb')).
        n_iters:     number of refinement iterations.

    Forward:
        modality_tokens:    dict[name -> (B, K_m, D)]  per-modality encoder outputs
        modality_present:   dict[name -> (B,) bool]    fleximodal mask (default: all present)
        modality_kpm:       dict[name -> (B, K_m) bool] optional key-padding masks
    Returns:
        L:  refined latent set (B, n_latents, D)
    """

    def __init__(self, n_latents: int, dim: int, modalities: tuple,
                 n_iters: int = 2, n_heads: int = 8, dim_head: int = 32,
                 ffn_mult: int = 4):
        super().__init__()
        self.n_latents  = n_latents
        self.modalities = tuple(modalities)
        self.n_iters    = n_iters
        # Sinusoidal-initialized learnable latents (perceiver-style)
        self.latents = nn.Parameter(get_sinusoidal_embeddings(n_latents, dim),
                                     requires_grad=True)
        # Per-modality cross-attention layer per iteration
        self.cross = nn.ModuleDict({
            m: nn.ModuleList([
                ModalityCrossAttn(dim, n_heads=n_heads, dim_head=dim_head)
                for _ in range(n_iters)
            ]) for m in modalities
        })
        # Shared self-attention + FFN per iteration
        self.self_attn = nn.ModuleList([
            nn.Sequential(nn.LayerNorm(dim), MultiHeadAttention(dim, n_heads=n_heads, dim_head=dim_head))
            for _ in range(n_iters)
        ])
        self.ffn = nn.ModuleList([
            nn.Sequential(nn.LayerNorm(dim), FeedForward(dim, mult=ffn_mult))
            for _ in range(n_iters)
        ])

    def forward(self, modality_tokens: Dict[str, torch.Tensor],
                modality_present: Optional[Dict[str, torch.Tensor]] = None,
                modality_kpm: Optional[Dict[str, torch.Tensor]] = None) -> torch.Tensor:
        # Pick batch size from any modality
        any_mod = next(iter(modality_tokens.values()))
        B = any_mod.shape[0]
        L = repeat(self.latents, 'n d -> b n d', b=B)             # (B, N_lat, D)

        for it in range(self.n_iters):
            # Cross-attend to each present modality
            for m in self.modalities:
                if m not in modality_tokens:
                    continue
                if modality_present is not None and not bool(modality_present.get(m, torch.tensor(True)).any()):
                    continue
                kpm = modality_kpm.get(m) if modality_kpm is not None else None
                L = L + self.cross[m][it](L, modality_tokens[m], key_padding_mask=kpm)
            # Self-attention + FFN
            sa_norm, sa_attn = self.self_attn[it][0], self.self_attn[it][1]
            L = L + sa_attn(sa_norm(L))
            ffn_norm, ffn_block = self.ffn[it][0], self.ffn[it][1]
            L = L + ffn_block(ffn_norm(L))

        return L
