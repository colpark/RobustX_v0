"""EventScape (CARLA-simulation driving) loader for OmniBird.

EventScape: Gehrig et al., RAL 2021 — "Combining Events and Frames using
Recurrent Asynchronous Multimodal Networks for Monocular Depth Prediction"
Dataset page:  https://rpg.ifi.uzh.ch/RAMNet.html

The dataset contains synchronized:
  - Event streams (raw (x, y, t, polarity) lists per chunk)
  - RGB frames (one every ~50 ms)
  - Depth maps
  - Semantic segmentation
  - Camera poses (IMU)
recorded inside CARLA's driving simulation.

This loader reads the per-clip event files and yields windows of N_RAW events
ending at each RGB frame timestamp. For now we support:

  - "events_only" mode (single-modality): just yields the event window + label
                                          (we use the dominant semantic class
                                          in the window as a coarse label)
  - "events_rgb"   mode (multimodal):     yields events + paired RGB frame
                                          (used by ICMR Phase 2; see icmr.py)

The official EventScape format stores events as `events.h5` per clip; we
adapt the lightweight format used by `rpg_e2depth` / `RAMNet` examples.

To run without downloading the full dataset, use `datasets.synthetic`
instead — same OmniBird API.
"""
from __future__ import annotations

import os
import glob
from pathlib import Path

import numpy as np
import torch


def _normalize_events(ev: np.ndarray, sensor_h: int, sensor_w: int,
                      t_min: float, t_max: float) -> np.ndarray:
    """ev: (N_raw, 4) with columns (x_int, y_int, t_us, polarity ∈ {0,1}).
    Returns float32 (N_raw, 4) with columns (x, y, t, polarity) where
    x, y, t are in [-1, 1] and polarity ∈ {-1, +1}.
    """
    out = np.empty_like(ev, dtype=np.float32)
    out[:, 0] = (ev[:, 0] / (sensor_w - 1)) * 2.0 - 1.0
    out[:, 1] = (ev[:, 1] / (sensor_h - 1)) * 2.0 - 1.0
    out[:, 2] = ((ev[:, 2] - t_min) / max(t_max - t_min, 1.0)) * 2.0 - 1.0
    out[:, 3] = ev[:, 3] * 2.0 - 1.0
    return out


class EventScapeDataset:
    """Reads pre-extracted event windows from a directory tree:

        root/
          clip_000/
            events_0.npy        # (N_raw, 4): x, y, t, polarity   (pre-window)
            label_0.txt         # integer class label for this window
            rgb_0.png           # paired RGB frame  (multimodal mode)
            ...
          clip_001/
            ...

    The per-window .npy is the same `events` array the OmniBird dataset wrapper
    expects. Adapt your conversion pipeline to write this directory layout
    (rpg's example scripts make this straightforward), or use the synthetic
    dataset for development.

    Use `EventScapeDataset(root, mode="events_only")` for single-modality.
    """

    def __init__(self, root: str, mode: str = "events_only",
                 sensor_hw=(256, 256)):
        self.root = Path(root)
        self.mode = mode
        self.sensor_h, self.sensor_w = sensor_hw

        # Index every (events_X.npy, label_X.txt[, rgb_X.png]) triplet
        self.windows = []
        for clip_dir in sorted(self.root.glob("clip_*")):
            for ev_file in sorted(clip_dir.glob("events_*.npy")):
                stem = ev_file.stem.replace("events_", "")
                label_file = clip_dir / f"label_{stem}.txt"
                rgb_file   = clip_dir / f"rgb_{stem}.png"
                if not label_file.exists():
                    continue
                if mode == "events_rgb" and not rgb_file.exists():
                    continue
                self.windows.append((ev_file, label_file, rgb_file if rgb_file.exists() else None))

        if len(self.windows) == 0:
            raise RuntimeError(
                f"EventScapeDataset({self.root}) found no windows. "
                f"Expected directory layout: clip_*/events_*.npy + label_*.txt.\n"
                "For development without the dataset, use datasets.synthetic instead."
            )

    def __len__(self):
        return len(self.windows)

    def __getitem__(self, idx):
        ev_file, label_file, rgb_file = self.windows[idx]
        raw = np.load(ev_file)                                  # (N_raw, 4)  x, y, t_us, polarity
        if raw.size == 0:
            raw = np.zeros((1, 4), dtype=np.float32)
        # Normalize coords to [-1, 1] and polarity to {-1, +1}
        t_min = raw[:, 2].min(); t_max = raw[:, 2].max()
        events = _normalize_events(raw, self.sensor_h, self.sensor_w, t_min, t_max)
        label = int(label_file.read_text().strip())

        if self.mode == "events_only":
            return events, label

        # Multimodal mode also returns the paired RGB frame
        if rgb_file is None or self.mode != "events_rgb":
            return events, label
        from PIL import Image
        rgb = np.asarray(Image.open(rgb_file).convert("RGB")) / 255.0    # (H, W, 3) float
        return {"events": events, "rgb": rgb.astype(np.float32), "label": label}
