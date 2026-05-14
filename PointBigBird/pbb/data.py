"""CIFAR-10 dataset with v8 i-JEPA multi-block masking + per-sample serializations.

Each sample provides:
  - ctx_pixels  (K_ctx, 3)     context pixels (the trained encoder's input)
  - ctx_coords  (K_ctx, 2)     coords in [-1, 1]^2
  - ctx_pixel_ids (K_ctx,)     row-major pixel indices in [0, N_pix)
  - pool_pixels (K_pool, 3)    target-encoder input (the full 40% pool)
  - pool_coords (K_pool, 2)
  - pool_pixel_ids (K_pool,)
  - tgt_coords  (N_tgt, 2)     coordinates the predictor must predict at
  - tgt_pool_pos (N_tgt,)      position of each target coord within `pool_*`
                                (used to look up h_tgt from the target encoder)
  - label (int)
  - {ctx,pool}_perm_<order>     (K_ctx,) or (K_pool,) — curve permutations
  - {ctx,pool}_inv_<order>      inverses of the above

For train=False (test loader), the sampling is deterministic: context = the
fixed first K_HALF of the pool, no target blocks.
"""
from __future__ import annotations

from typing import Dict

import numpy as np
import torch
import torchvision
from torch.utils.data import Dataset
from torchvision import transforms

from .config import PBBConfig
from .serialization import precompute_grid_orderings, subset_perm, invert_perm


CIFAR_CLASSES = [
    "airplane", "car", "bird", "cat", "deer",
    "dog", "frog", "horse", "ship", "truck",
]


class PBBChunkCIFAR10(Dataset):
    def __init__(self, base_dataset, cfg: PBBConfig, train=True):
        self.base = base_dataset
        self.cfg = cfg
        self.train = train

        rng = np.random.RandomState(cfg.pool_seed)
        self.pool_idx = np.stack(
            [rng.permutation(cfg.n_pix)[:cfg.k_pool] for _ in range(len(base_dataset))],
            axis=0,
        ).astype(np.int64)

        # Coords grid in [-1, 1]^2
        ys, xs = torch.meshgrid(
            torch.linspace(-1.0, 1.0, cfg.image_size),
            torch.linspace(-1.0, 1.0, cfg.image_size),
            indexing="ij",
        )
        self.coords_all = torch.stack([ys, xs], dim=-1).view(cfg.n_pix, 2)

        # Precomputed grid orderings
        self.grid_ranks: Dict[str, torch.Tensor] = precompute_grid_orderings(cfg.image_size)

    def __len__(self):
        return len(self.base)

    def _build_orderings(self, pixel_ids: torch.Tensor):
        """For a (K,) LongTensor of pixel ids, return dict of {name: (perm, inverse)}."""
        out = {}
        for name, full_rank in self.grid_ranks.items():
            ranks = full_rank[pixel_ids]
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

    def __getitem__(self, idx):
        img, label = self.base[idx]
        pix_all = img.permute(1, 2, 0).reshape(-1, 3)
        pool = self.pool_idx[idx]
        pool_xy = self.coords_all[pool]
        cfg = self.cfg

        if self.train:
            rng = np.random.RandomState()
            disjoint_targets = getattr(cfg, "disjoint_targets", False)

            # --- v8 multi-block masking ---
            forbidden = np.zeros(cfg.k_pool, dtype=bool)
            tgt_blocks_local = []
            for _ in range(cfg.n_pred):
                if disjoint_targets:
                    # Sample anchor from positions NOT yet in any target block,
                    # and restrict the k-nearest search to those positions too.
                    allowed_now = np.where(~forbidden)[0]
                    a_pos_in_allowed = rng.randint(len(allowed_now))
                    a = allowed_now[a_pos_in_allowed]
                    d2 = (pool_xy - pool_xy[a]).pow(2).sum(-1).numpy()
                    d2[forbidden] = np.inf
                else:
                    a = rng.randint(cfg.k_pool)
                    d2 = (pool_xy - pool_xy[a]).pow(2).sum(-1).numpy()
                blk = np.argsort(d2, kind="stable")[:cfg.k_tgt]
                tgt_blocks_local.append(blk)
                forbidden[blk] = True
            tgt_local = np.concatenate(tgt_blocks_local).astype(np.int64)

            allowed = np.where(~forbidden)[0]
            a_ctx = allowed[rng.randint(len(allowed))]
            d2 = (pool_xy[allowed] - pool_xy[a_ctx]).pow(2).sum(-1).numpy()
            ctx_local = allowed[np.argsort(d2, kind="stable")[:cfg.k_ctx]]

            ctx_idx = pool[ctx_local]
            tgt_idx = pool[tgt_local]

            ctx_pixel_ids = torch.from_numpy(ctx_idx)
            pool_pixel_ids = torch.from_numpy(pool)
            tgt_pool_pos = torch.from_numpy(tgt_local.astype(np.int64))

            ctx_ords  = self._build_orderings(ctx_pixel_ids)
            pool_ords = self._build_orderings(pool_pixel_ids)

            sample = {
                "ctx_pixels":     pix_all[ctx_idx],
                "ctx_coords":     self.coords_all[ctx_idx],
                "ctx_pixel_ids":  ctx_pixel_ids,
                "pool_pixels":    pix_all[pool],
                "pool_coords":    self.coords_all[pool],
                "pool_pixel_ids": pool_pixel_ids,
                "tgt_coords":     self.coords_all[tgt_idx],
                "tgt_pool_pos":   tgt_pool_pos,
                "label":          int(label),
            }
            sample.update(self._pack_orderings("ctx",  ctx_ords))
            sample.update(self._pack_orderings("pool", pool_ords))
            return sample
        else:
            ctx_idx = pool[:cfg.k_half]
            ctx_pixel_ids = torch.from_numpy(ctx_idx)
            ctx_ords = self._build_orderings(ctx_pixel_ids)

            sample = {
                "ctx_pixels":    pix_all[ctx_idx],
                "ctx_coords":    self.coords_all[ctx_idx],
                "ctx_pixel_ids": ctx_pixel_ids,
                "label":         int(label),
            }
            sample.update(self._pack_orderings("ctx", ctx_ords))
            return sample


# ---------------------------------------------------------------------------
# Builders
# ---------------------------------------------------------------------------

def build_loaders(cfg: PBBConfig, num_workers: int = 2):
    from torch.utils.data import DataLoader
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,)*3, (0.5,)*3),
    ])
    cifar_train = torchvision.datasets.CIFAR10(root="./data", train=True,  download=True, transform=transform)
    cifar_test  = torchvision.datasets.CIFAR10(root="./data", train=False, download=True, transform=transform)

    train_ds      = PBBChunkCIFAR10(cifar_train, cfg, train=True)
    train_eval_ds = PBBChunkCIFAR10(cifar_train, cfg, train=False)
    test_ds       = PBBChunkCIFAR10(cifar_test,  cfg, train=False)

    train_loader      = DataLoader(train_ds,      batch_size=cfg.batch_size, shuffle=True,
                                    num_workers=num_workers, pin_memory=True)
    train_eval_loader = DataLoader(train_eval_ds, batch_size=cfg.batch_size, shuffle=True,
                                    num_workers=num_workers, pin_memory=True)
    test_loader       = DataLoader(test_ds,       batch_size=cfg.batch_size, shuffle=False,
                                    num_workers=num_workers, pin_memory=True)
    return train_loader, train_eval_loader, test_loader


def orderings_from_batch(batch: dict, prefix: str, names=("z", "z_rev", "hilbert", "hilbert_rev")):
    """Extract per-sample orderings dict from a batch dict.

    Returns:
        dict[name] -> {"perm": (B, K) long, "inverse": (B, K) long}
    """
    return {
        name: {
            "perm":    batch[f"{prefix}_perm_{name}"],
            "inverse": batch[f"{prefix}_inv_{name}"],
        }
        for name in names
    }
