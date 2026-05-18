"""Checkpoint and small utilities."""
from __future__ import annotations

import os
import torch


def save_atomic(state, path):
    """Atomic save via temp + os.replace. Avoids dotfile names PyTorch rejects."""
    tmp = path + ".tmp"
    try:
        torch.save(state, tmp)
        os.replace(tmp, path)
    except Exception:
        if os.path.exists(tmp):
            try: os.remove(tmp)
            except OSError: pass
        raise


def ensure_dir(d):
    os.makedirs(d, exist_ok=True)


def count_params(module):
    return sum(p.numel() for p in module.parameters())


def short_params(module):
    n = count_params(module)
    return f"{n/1e6:.2f}M" if n >= 1e6 else f"{n/1e3:.1f}K"
