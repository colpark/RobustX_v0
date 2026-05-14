"""Synthetic event-camera dataset for development (no download required).

Generates 10-class classification samples: for each class we emit events along
a class-specific 3-D "trajectory" with class-specific noise, so a model that
learns spatio-temporal structure should be able to separate them. Useful for
verifying the OmniBird pipeline end-to-end before plugging in real data.

Each sample is N_RAW events of shape (4,) = (x, y, t, polarity).
"""
from __future__ import annotations

import numpy as np
import torch


CLASSES = [f"class_{i}" for i in range(10)]


def _make_clip(class_id: int, n_events: int, rng: np.random.Generator):
    """Generate n_events with class-specific spatio-temporal structure."""
    # 10 distinct trajectories: each class has a different curve in (x, y, t) space.
    base_freq = 0.5 + 0.4 * class_id                                 # 0.5 .. 4.1
    phase_x   = (class_id * 0.628) % (2 * np.pi)
    phase_y   = (class_id * 1.107) % (2 * np.pi)
    drift_x   = 0.05 * (class_id - 4.5)
    drift_y   = 0.05 * ((class_id * 3) % 10 - 4.5)

    t = np.linspace(-1.0, 1.0, n_events).astype(np.float32) + 0.02 * rng.standard_normal(n_events).astype(np.float32)
    x = np.sin(base_freq * np.pi * t + phase_x) * 0.7 + drift_x + 0.05 * rng.standard_normal(n_events).astype(np.float32)
    y = np.cos(base_freq * np.pi * t + phase_y) * 0.7 + drift_y + 0.05 * rng.standard_normal(n_events).astype(np.float32)
    x = np.clip(x, -0.99, 0.99); y = np.clip(y, -0.99, 0.99); t = np.clip(t, -0.99, 0.99)
    polarity = (2 * rng.integers(0, 2, n_events) - 1).astype(np.float32)
    events = np.stack([x, y, t, polarity], axis=1)
    return events


class SyntheticEventDataset:
    """Drop-in for OmniBirdEventDataset.base.

    Each __getitem__ returns (events ndarray (N_raw, 4), label int).
    """

    def __init__(self, n_samples: int = 5000, n_events_per_sample: int = 2048,
                 n_classes: int = 10, seed: int = 0):
        self.n_samples = n_samples
        self.n_events  = n_events_per_sample
        self.n_classes = n_classes
        self.seed = seed
        # Deterministic class assignment per index
        self.labels = (np.arange(n_samples) % n_classes).astype(np.int64)

    def __len__(self):
        return self.n_samples

    def __getitem__(self, idx):
        # Per-sample RNG so labels are deterministic but events vary per call
        rng = np.random.default_rng(self.seed * 1_000_003 + idx)
        label = int(self.labels[idx])
        events = _make_clip(label, self.n_events, rng)
        return events, label


def build_synthetic_loaders(cfg, n_train: int = 5000, n_test: int = 1000):
    """Convenience: build train+test SyntheticEventDataset and wrap in OmniBird loaders."""
    from omnibird import build_loaders
    train_base = SyntheticEventDataset(n_samples=n_train, n_events_per_sample=cfg.n_events_total, seed=0)
    test_base  = SyntheticEventDataset(n_samples=n_test,  n_events_per_sample=cfg.n_events_total, seed=1)
    return build_loaders(train_base, test_base, cfg)
