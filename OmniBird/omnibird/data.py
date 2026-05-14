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

    # --- helpers ----------------------------------------------------------
    def _build_signal(self, events: torch.Tensor) -> torch.Tensor:
        """Per-event signal vector. `cfg.signal_dim`:
            1 -> scalar polarity ∈ {-1, +1}  (legacy)
            2 -> one-hot [ON, OFF] channels  (recommended for events)
        """
        if self.cfg.signal_dim == 2:
            pol = events[:, 3]
            sig = torch.zeros(events.shape[0], 2, dtype=torch.float32)
            sig[pol > 0, 0] = 1.0          # ON channel
            sig[pol < 0, 1] = 1.0          # OFF channel
            return sig
        return events[:, 3:4]

    # --- core -------------------------------------------------------------
    def __getitem__(self, idx):
        events, label = self.base[idx]                 # (N_raw, 4) float32
        cfg = self.cfg
        events = torch.as_tensor(events, dtype=torch.float32)
        N_raw = events.shape[0]

        # 1. Either keep all events (n_events_total <= 0) or cap at n_events_total
        if cfg.n_events_total <= 0:
            # Use every event from the clip; per-batch padding handled by collate
            # to the max event count in that batch. For now we still pad here to
            # a global ceiling so DataLoader's default collate works.
            global_max = max(int(getattr(cfg, "n_events_max", 16384)), N_raw)
            N_target = global_max
        else:
            N_target = cfg.n_events_total

        if N_raw > N_target:
            # Rare with our converter (which caps clips at events_per_clip).
            if self.train:
                sel = np.random.choice(N_raw, N_target, replace=False)
            else:
                rng = np.random.RandomState(idx + 7919)
                sel = rng.choice(N_raw, N_target, replace=False)
                sel.sort()
            events = events[sel]
            N_real = N_target
        else:
            N_real = N_raw

        # Pad with zeros up to N_target. Real events occupy indices [0, N_real),
        # pad events occupy [N_real, N_target).
        if events.shape[0] < N_target:
            pad_count = N_target - events.shape[0]
            pad_ev = torch.zeros(pad_count, 4, dtype=events.dtype)
            events = torch.cat([events, pad_ev], dim=0)

        N = events.shape[0]                              # = N_target
        coords  = events[:, :3]                          # (N, 3) — x, y, t in [-1, 1]
        signal  = self._build_signal(events)             # (N, signal_dim)
        pool_kpm = torch.zeros(N, dtype=torch.bool)
        pool_kpm[N_real:] = True                         # True = pad (mask out)

        if self.train:
            rng = np.random.RandomState()
            # Sample blocks ONLY from real events (indices in [0, N_real))
            forbidden = np.zeros(N_real, dtype=bool)
            tgt_blocks_local = []

            for _ in range(cfg.n_pred_blocks):
                if cfg.disjoint_targets:
                    allowed_now = np.where(~forbidden)[0]
                    if len(allowed_now) < cfg.n_tgt_per_block:
                        # Clip too sparse — just take what's available
                        blk = allowed_now[:cfg.n_tgt_per_block]
                    else:
                        a = allowed_now[rng.randint(len(allowed_now))]
                        d2 = (coords[:N_real] - coords[a]).pow(2).sum(-1).numpy()
                        d2[forbidden] = np.inf
                        blk = np.argsort(d2, kind="stable")[:cfg.n_tgt_per_block]
                else:
                    a = rng.randint(N_real)
                    d2 = (coords[:N_real] - coords[a]).pow(2).sum(-1).numpy()
                    blk = np.argsort(d2, kind="stable")[:cfg.n_tgt_per_block]
                # Pad block if undersized so all batches have fixed n_tgt_per_block
                if len(blk) < cfg.n_tgt_per_block:
                    extra = np.full(cfg.n_tgt_per_block - len(blk), blk[0] if len(blk) else 0, dtype=blk.dtype)
                    blk = np.concatenate([blk, extra])
                tgt_blocks_local.append(blk)
                forbidden[blk] = True
            tgt_local = np.concatenate(tgt_blocks_local).astype(np.int64)

            allowed = np.where(~forbidden)[0]
            n_ctx = min(cfg.n_ctx, len(allowed))
            if n_ctx > 0:
                a_ctx = allowed[rng.randint(len(allowed))]
                d2 = (coords[allowed] - coords[a_ctx]).pow(2).sum(-1).numpy()
                ctx_local = allowed[np.argsort(d2, kind="stable")[:n_ctx]].astype(np.int64)
            else:
                ctx_local = np.array([], dtype=np.int64)
            # Pad context to fixed cfg.n_ctx so batches stack
            if len(ctx_local) < cfg.n_ctx:
                pad_n = cfg.n_ctx - len(ctx_local)
                extra = np.full(pad_n, ctx_local[0] if len(ctx_local) else 0, dtype=ctx_local.dtype)
                ctx_local = np.concatenate([ctx_local, extra])

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
                "pool_kpm":       pool_kpm,            # True = pad event
                "tgt_coords":     coords[tgt_local],
                "tgt_pool_pos":   tgt_pool_pos,
                "label":          int(label),
            }
            sample.update(self._pack_orderings("ctx",  ctx_ords))
            sample.update(self._pack_orderings("pool", pool_ords))
            return sample

        # ---- test / probe path: full padded event set as context, no targets ----
        # We keep the padded shape so default batch collation works; the encoder
        # will mask out padded positions via ctx_kpm.
        ctx_cells = self._quantize(coords)
        ctx_ords  = self._build_orderings(ctx_cells)
        sample = {
            "ctx_signal": signal,
            "ctx_coords": coords,
            "ctx_kpm":    pool_kpm,                  # True = pad
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
