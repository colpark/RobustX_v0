"""Generic event-camera dataset wrapper for OmniBird-JEPA.

Each underlying dataset must yield, for index `idx`, a tuple:

    (events, label)
        events: (N_raw, 4) float32 — columns are (x, y, t, polarity)
                x, y in [-1, 1]; t in [-1, 1]; polarity in {-1, +1}
        label : int

`OmniBirdEventDataset` then:
  1. Sub-samples to `cfg.n_events_total` events per sample (random per-call).
  2. (train=True) Splits those events into:
       - 4 disjoint target blocks of `cfg.n_tgt_per_block` events each
       - 1 context block of `cfg.n_ctx` events, disjoint from targets
     "Blocks" here are contiguous chunks in *3-D Euclidean space* (events
     near each other in (x, y, t) form a block — same logic as PBB).
  3. (train=False) Returns all sub-sampled events as context (probe input).
  4. Looks up per-sample serialization orderings on a `cfg.side`-resolution
     3-D Morton/Hilbert grid (precomputed once at dataset construction time).

The downstream model is agnostic to where the events came from — synthetic,
CIFAR10-DVS, or EventScape (CARLA).  See `datasets/` for adapters.
"""
from __future__ import annotations

from typing import Dict

import numpy as np
import torch
from torch.utils.data import Dataset

from .config import OmniBirdConfig
from .serialization import precompute_grid_orderings, invert_perm, quantize_coords


class OmniBirdEventDataset(Dataset):
    """Wraps any "event clip" dataset into JEPA-ready batches."""

    def __init__(self, base_dataset, cfg: OmniBirdConfig, train: bool = True,
                 sub_sample_seed: int = 12345):
        self.base  = base_dataset
        self.cfg   = cfg
        self.train = train
        self.rng_default = np.random.RandomState(sub_sample_seed)
        self.grid_ranks: Dict[str, torch.Tensor] = precompute_grid_orderings(cfg.side, ndim=3)

    def __len__(self):
        return len(self.base)

    # --- helpers ----------------------------------------------------------
    def _quantize(self, coords_xyz: torch.Tensor) -> torch.Tensor:
        """(K, 3) float coords in [-1, 1] -> flat cell ids in [0, side**3)."""
        return quantize_coords(coords_xyz, self.cfg.side, value_range=(-1.0, 1.0))

    def _build_orderings(self, cell_ids: torch.Tensor):
        out = {}
        for name, full_rank in self.grid_ranks.items():
            ranks = full_rank[cell_ids]
            p = torch.argsort(ranks)
            inv = invert_perm(p)
            out[name] = (p, inv)
        return out

    def _pack_orderings(self, prefix, orderings):
        d = {}
        for name, (p, inv) in orderings.items():
            d[f"{prefix}_perm_{name}"] = p
            d[f"{prefix}_inv_{name}"]  = inv
        return d

    # --- core -------------------------------------------------------------
    def __getitem__(self, idx):
        events, label = self.base[idx]                 # (N_raw, 4) float32
        cfg = self.cfg
        events = torch.as_tensor(events, dtype=torch.float32)
        N_raw = events.shape[0]

        # 1. Sub-sample to n_events_total
        if N_raw >= cfg.n_events_total:
            if self.train:
                sel = np.random.choice(N_raw, cfg.n_events_total, replace=False)
            else:
                rng = np.random.RandomState(idx + 7919)        # deterministic per-sample
                sel = rng.choice(N_raw, cfg.n_events_total, replace=False)
                sel.sort()
            events = events[sel]
        elif N_raw < cfg.n_events_total:
            pad = cfg.n_events_total - N_raw
            extra = events[np.random.randint(0, N_raw, pad)]
            events = torch.cat([events, extra], dim=0)

        coords  = events[:, :3]                          # (N, 3) — x, y, t in [-1, 1]
        signal  = events[:, 3:4]                         # (N, 1) — polarity in {-1, +1}
        N = events.shape[0]

        if self.train:
            rng = np.random.RandomState()
            forbidden = np.zeros(N, dtype=bool)
            tgt_blocks_local = []

            for _ in range(cfg.n_pred_blocks):
                if cfg.disjoint_targets:
                    allowed_now = np.where(~forbidden)[0]
                    a = allowed_now[rng.randint(len(allowed_now))]
                    d2 = (coords - coords[a]).pow(2).sum(-1).numpy()
                    d2[forbidden] = np.inf
                else:
                    a = rng.randint(N)
                    d2 = (coords - coords[a]).pow(2).sum(-1).numpy()
                blk = np.argsort(d2, kind="stable")[:cfg.n_tgt_per_block]
                tgt_blocks_local.append(blk)
                forbidden[blk] = True
            tgt_local = np.concatenate(tgt_blocks_local).astype(np.int64)

            allowed = np.where(~forbidden)[0]
            a_ctx = allowed[rng.randint(len(allowed))]
            d2 = (coords[allowed] - coords[a_ctx]).pow(2).sum(-1).numpy()
            ctx_local = allowed[np.argsort(d2, kind="stable")[:cfg.n_ctx]].astype(np.int64)

            ctx_cells  = self._quantize(coords[ctx_local])
            pool_cells = self._quantize(coords)
            tgt_pool_pos = torch.from_numpy(tgt_local)

            ctx_ords  = self._build_orderings(ctx_cells)
            pool_ords = self._build_orderings(pool_cells)

            sample = {
                "ctx_signal":     signal[ctx_local],
                "ctx_coords":     coords[ctx_local],
                "pool_signal":    signal,
                "pool_coords":    coords,
                "tgt_coords":     coords[tgt_local],
                "tgt_pool_pos":   tgt_pool_pos,
                "label":          int(label),
            }
            sample.update(self._pack_orderings("ctx",  ctx_ords))
            sample.update(self._pack_orderings("pool", pool_ords))
            return sample

        # ---- test / probe path: all events as context, no targets ----
        ctx_cells = self._quantize(coords)
        ctx_ords  = self._build_orderings(ctx_cells)
        sample = {
            "ctx_signal": signal,
            "ctx_coords": coords,
            "label":      int(label),
        }
        sample.update(self._pack_orderings("ctx", ctx_ords))
        return sample


def orderings_from_batch(batch: dict, prefix: str,
                         names=("z", "z_rev", "hilbert", "hilbert_rev")):
    return {
        name: {
            "perm":    batch[f"{prefix}_perm_{name}"],
            "inverse": batch[f"{prefix}_inv_{name}"],
        }
        for name in names
    }


def build_loaders(base_train, base_test, cfg: OmniBirdConfig, num_workers: int = 2):
    from torch.utils.data import DataLoader
    train_ds      = OmniBirdEventDataset(base_train, cfg, train=True)
    train_eval_ds = OmniBirdEventDataset(base_train, cfg, train=False)
    test_ds       = OmniBirdEventDataset(base_test,  cfg, train=False)

    train_loader      = DataLoader(train_ds,      batch_size=cfg.batch_size, shuffle=True,
                                    num_workers=num_workers, pin_memory=True)
    train_eval_loader = DataLoader(train_eval_ds, batch_size=cfg.batch_size, shuffle=True,
                                    num_workers=num_workers, pin_memory=True)
    test_loader       = DataLoader(test_ds,       batch_size=cfg.batch_size, shuffle=False,
                                    num_workers=num_workers, pin_memory=True)
    return train_loader, train_eval_loader, test_loader
