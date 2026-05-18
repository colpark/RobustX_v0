"""Linear probe utilities — usable from both train.py and the training notebook."""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW

from .data import orderings_from_batch


class LinearProbe(nn.Module):
    def __init__(self, in_dim, n_classes=10):
        super().__init__()
        self.fc = nn.Linear(in_dim, n_classes)

    def forward(self, z):
        return self.fc(z)


class AttnPoolHead(nn.Module):
    """Attention-pool head: a learnable query Q ∈ ℝᴰ cross-attends to the
    encoder's per-token features g ∈ ℝᴮˣᴷˣᴰ and returns one D-dim feature
    z ∈ ℝᴮˣᴰ. Used as a more expressive alternative to `g.mean(dim=1)` for
    linear probing without retraining the encoder.
    """

    def __init__(self, dim, n_heads=4, dim_head=64):
        super().__init__()
        inner = n_heads * dim_head
        self.heads = n_heads
        self.dim_head = dim_head
        self.scale = dim_head ** -0.5
        self.query  = nn.Parameter(torch.randn(1, 1, dim) * 0.02)
        self.norm_q = nn.LayerNorm(dim)
        self.norm_kv = nn.LayerNorm(dim)
        self.to_q  = nn.Linear(dim, inner, bias=False)
        self.to_kv = nn.Linear(dim, inner * 2, bias=False)
        self.to_out = nn.Linear(inner, dim)

    def forward(self, g):
        B, K, D = g.shape
        q  = self.to_q(self.norm_q(self.query.expand(B, -1, -1)))         # (B, 1, inner)
        kv = self.to_kv(self.norm_kv(g))                                  # (B, K, 2*inner)
        k, v = kv.chunk(2, dim=-1)
        q = q.view(B, 1, self.heads, self.dim_head).transpose(1, 2)       # (B, H, 1, Dh)
        k = k.view(B, K, self.heads, self.dim_head).transpose(1, 2)       # (B, H, K, Dh)
        v = v.view(B, K, self.heads, self.dim_head).transpose(1, 2)
        scores = (q @ k.transpose(-2, -1)) * self.scale                   # (B, H, 1, K)
        attn = scores.softmax(dim=-1)
        out  = (attn @ v).transpose(1, 2).contiguous().view(B, 1, -1)     # (B, 1, inner)
        return self.to_out(out).squeeze(1)                                # (B, D)


def _move_ords(ords, device):
    return {k: {kk: vv.to(device) for kk, vv in v.items()} for k, v in ords.items()}


@torch.no_grad()
def extract_z(context_encoder, signal, coords, orderings):
    """Frozen-encoder mean-pooled feature for the linear probe.

    Returns a (B, D_model) tensor.
    """
    context_encoder.eval()
    g = context_encoder(signal, coords, orderings)
    return g.mean(dim=1)


@torch.no_grad()
def _encode_g(context_encoder, signal, coords, orderings):
    """Forward through the (frozen) encoder, return per-token features `g`."""
    context_encoder.eval()
    return context_encoder(signal, coords, orderings)


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
    use_attn_pool: bool = False,
    attn_pool_heads: int = 4,
    attn_pool_dim_head: int = 64,
):
    """Train a fresh probe head on top of the frozen `context_encoder` for
    `num_epochs` epochs through `train_eval_loader`, then report test accuracy
    on `test_loader`.

    - `use_attn_pool=False` (default): probe = `LinearProbe` on `g.mean(dim=1)`.
    - `use_attn_pool=True`           : probe = `AttnPoolHead(g) → LinearProbe`,
                                       trains both head and classifier together.

    The encoder is always frozen. `requires_grad` flags and `training` mode are
    saved and restored, so calling this mid-training does not perturb the JEPA
    pre-training state.
    """
    saved_train = context_encoder.training
    saved_rg    = [p.requires_grad for p in context_encoder.parameters()]
    for p in context_encoder.parameters():
        p.requires_grad_(False)

    classifier = LinearProbe(in_dim=d_model, n_classes=n_classes).to(device)
    if use_attn_pool:
        pool = AttnPoolHead(d_model, n_heads=attn_pool_heads,
                            dim_head=attn_pool_dim_head).to(device)
        trainables = list(pool.parameters()) + list(classifier.parameters())
    else:
        pool = None
        trainables = list(classifier.parameters())

    opt = AdamW(trainables, lr=lr, weight_decay=weight_decay)
    ce  = nn.CrossEntropyLoss()

    def _z_from_batch(b):
        # OmniBird uses ctx_signal (event polarity) rather than ctx_pixels (RGB)
        ctx_s = b["ctx_signal"].to(device)
        ctx_c = b["ctx_coords"].to(device)
        ctx_o = _move_ords(orderings_from_batch(b, "ctx"), device)
        ctx_kpm = b["ctx_kpm"].to(device) if "ctx_kpm" in b else None
        with torch.no_grad():
            context_encoder.eval()
            g = context_encoder(ctx_s, ctx_c, ctx_o, key_padding_mask=ctx_kpm)
        if pool is not None:
            return pool(g)                                              # (B, D)
        # Mean-pool only over real (non-pad) positions when a kpm is provided.
        if ctx_kpm is not None:
            mask = (~ctx_kpm).unsqueeze(-1).float()                     # (B, K, 1)
            return (g * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1)
        return g.mean(dim=1)                                            # (B, D)

    for _ in range(num_epochs):
        classifier.train()
        if pool is not None: pool.train()
        for b in train_eval_loader:
            y = b["label"].to(device)
            z = _z_from_batch(b)
            logits = classifier(z)
            loss = ce(logits, y)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()

    classifier.eval()
    if pool is not None: pool.eval()
    correct = total = 0
    with torch.no_grad():
        for b in test_loader:
            y = b["label"].to(device)
            z = _z_from_batch(b)
            correct += (classifier(z).argmax(-1) == y).sum().item()
            total   += y.size(0)
    acc = correct / max(total, 1)

    for p, rg in zip(context_encoder.parameters(), saved_rg):
        p.requires_grad_(rg)
    context_encoder.train(saved_train)
    return acc
