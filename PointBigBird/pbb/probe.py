"""Linear probe utilities — usable from both train.py and the training notebook."""
from __future__ import annotations

import torch
import torch.nn as nn
from torch.optim import AdamW

from .data import orderings_from_batch


class LinearProbe(nn.Module):
    def __init__(self, in_dim, n_classes=10):
        super().__init__()
        self.fc = nn.Linear(in_dim, n_classes)

    def forward(self, z):
        return self.fc(z)


def _move_ords(ords, device):
    return {k: {kk: vv.to(device) for kk, vv in v.items()} for k, v in ords.items()}


@torch.no_grad()
def extract_z(context_encoder, pixels, coords, orderings):
    """Frozen-encoder mean-pooled feature for the linear probe.

    Returns a (B, D_model) tensor.
    """
    context_encoder.eval()
    g = context_encoder(pixels, coords, orderings)
    return g.mean(dim=1)


def quick_probe(
    context_encoder: nn.Module,
    train_eval_loader,
    test_loader,
    d_model: int,
    n_classes: int = 10,
    num_epochs: int = 3,
    lr: float = 1e-3,
    weight_decay: float = 1e-4,
    device: str = "cpu",
):
    """Train a fresh `LinearProbe` on top of the frozen `context_encoder` for
    `num_epochs` epochs through `train_eval_loader`, then report test accuracy
    on `test_loader`.

    Saves and restores the encoder's `requires_grad` flags and `training` mode,
    so calling this mid-training does not perturb the JEPA pre-training state.
    """
    saved_train = context_encoder.training
    saved_rg    = [p.requires_grad for p in context_encoder.parameters()]
    for p in context_encoder.parameters():
        p.requires_grad_(False)

    probe = LinearProbe(in_dim=d_model, n_classes=n_classes).to(device)
    opt   = AdamW(probe.parameters(), lr=lr, weight_decay=weight_decay)
    ce    = nn.CrossEntropyLoss()

    for _ in range(num_epochs):
        probe.train()
        for b in train_eval_loader:
            ctx_p = b["ctx_pixels"].to(device)
            ctx_c = b["ctx_coords"].to(device)
            ctx_o = _move_ords(orderings_from_batch(b, "ctx"), device)
            y = b["label"].to(device)
            z = extract_z(context_encoder, ctx_p, ctx_c, ctx_o)
            logits = probe(z)
            loss = ce(logits, y)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()

    probe.eval()
    correct = total = 0
    with torch.no_grad():
        for b in test_loader:
            ctx_p = b["ctx_pixels"].to(device)
            ctx_c = b["ctx_coords"].to(device)
            ctx_o = _move_ords(orderings_from_batch(b, "ctx"), device)
            y = b["label"].to(device)
            z = extract_z(context_encoder, ctx_p, ctx_c, ctx_o)
            correct += (probe(z).argmax(-1) == y).sum().item()
            total   += y.size(0)
    acc = correct / max(total, 1)

    for p, rg in zip(context_encoder.parameters(), saved_rg):
        p.requires_grad_(rg)
    context_encoder.train(saved_train)
    return acc
