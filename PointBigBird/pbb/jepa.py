"""JEPA training utilities — the "gist" of v1–v8 distilled.

Carried over from v8:
  - EMA target encoder (m: 0.999 → 1.0 over training)
  - DINO-style centering of the target features
  - smooth-L1 distance between predictor output and target features
  - Multi-block disjoint masking lives in `data.py`

Dropped (OmniField-specific, doesn't apply to per-token encoder):
  - latent_pos with Gaussian attention bias
  - deterministic Gaussian soft-pool (h_tgt now comes from direct lookup
    since target_coords ⊂ pool_coords by v8 construction)
  - aux variance loss, predictor warmup
"""
from __future__ import annotations

from typing import Optional

import copy
import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# EMA & centering
# ---------------------------------------------------------------------------

@torch.no_grad()
def ema_update(target_module: nn.Module, online_module: nn.Module, m: float):
    for p_q, p_k in zip(online_module.parameters(), target_module.parameters()):
        p_k.data.mul_(m).add_((1.0 - m) * p_q.detach())


def make_momentum_schedule(start: float, end: float, total_steps: int):
    """Linear schedule from `start` to `end` over `total_steps`."""
    for i in range(total_steps + 1):
        yield start + i * (end - start) / total_steps


class TargetCenter(nn.Module):
    """DINO-style EMA of the per-feature mean — subtracted from target features."""

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


# ---------------------------------------------------------------------------
# Target lookup
# ---------------------------------------------------------------------------

def gather_target_features(g_tgt: torch.Tensor, tgt_pool_pos: torch.Tensor) -> torch.Tensor:
    """g_tgt:        (B, K_pool, D)  target-encoder features over the full pool
    tgt_pool_pos: (B, N_tgt) long — index of each target coord within the pool
    Returns: h_tgt_raw (B, N_tgt, D).
    """
    B, K_pool, D = g_tgt.shape
    idx = tgt_pool_pos.unsqueeze(-1).expand(B, tgt_pool_pos.shape[1], D)
    return torch.gather(g_tgt, 1, idx)


# ---------------------------------------------------------------------------
# JEPA loss
# ---------------------------------------------------------------------------

def jepa_loss(h_pred: torch.Tensor, h_tgt: torch.Tensor, loss_type: str = "smooth_l1") -> torch.Tensor:
    """JEPA distance between predictor output and (centered + LayerNormed) targets.

    Both inputs: (B, N_tgt, D).

    - "smooth_l1" (legacy): penalizes pointwise magnitude differences.
    - "cosine":             1 - cos(h_pred, h_tgt) averaged over (B, N_tgt).
                            Direction-only — no pull on h_pred's magnitude.
                            Tends to be more robust over long training.
    """
    if loss_type == "smooth_l1":
        return F.smooth_l1_loss(h_pred, h_tgt)
    elif loss_type == "cosine":
        return (1.0 - F.cosine_similarity(h_pred, h_tgt, dim=-1)).mean()
    raise ValueError(f"unknown loss_type: {loss_type!r} (expected 'smooth_l1' or 'cosine')")


# ---------------------------------------------------------------------------
# Diagnostics
# ---------------------------------------------------------------------------

@torch.no_grad()
def diag_dict(loss, h_pred, h_tgt, g_ctx, target_center: TargetCenter):
    cos = F.cosine_similarity(h_pred, h_tgt, dim=-1)
    return {
        "loss":        loss.item(),
        "|g_ctx|":     g_ctx.norm(dim=-1).mean().item(),
        "|h_pred|":    h_pred.norm(dim=-1).mean().item(),
        "|h_tgt|":     h_tgt.norm(dim=-1).mean().item(),
        "std_b(hp)":   h_pred.std(dim=0).mean().item(),     # collapse-↓ signal
        "std_t(g)":    g_ctx.std(dim=1).mean().item(),
        "cos_mean":    cos.mean().item(),
        "cos_std":     cos.std().item(),
        "|center|":    target_center.center.norm().item(),
    }


def fmt_diag(d, step, epoch, lr, m):
    return (f"[ep{epoch:02d} st{step:05d}]  "
            f"L={d['loss']:.4f}  "
            f"|g|={d['|g_ctx|']:.2f} |hp|={d['|h_pred|']:.2f} |ht|={d['|h_tgt|']:.2f}  "
            f"σb(hp)={d['std_b(hp)']:.3f} σt(g)={d['std_t(g)']:.3f}  "
            f"cos={d['cos_mean']:.3f}±{d['cos_std']:.3f}  "
            f"|cen|={d['|center|']:.2f}  lr={lr:.1e} m={m:.4f}")
