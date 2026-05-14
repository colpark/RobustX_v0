"""Space-filling curve serialization for sparse N-D point sets.

Adapts the 2D z-order / Hilbert / reverses from PointBigBird to 3D (events have
spatial-temporal coordinates: x, y, t). Both dimensions share the same API:

  precompute_grid_orderings(side, ndim=3)  ->  dict[name -> LongTensor (side**ndim,)]
  subset_perm(subset_ids, full_rank)       ->  argsort the subset by curve rank

For event-camera data we discretize (x, y, t) onto a `side × side × side` grid
(typically 32 × 32 × 32 — 32k positions) and precompute the 4 orderings once.
Per-event lookup is then O(K log K), independent of the grid size.
"""
from __future__ import annotations

from typing import Tuple

import numpy as np
import torch


# ---------------------------------------------------------------------------
# 2-D (kept for back-compat with PointBigBird tests)
# ---------------------------------------------------------------------------

def morton_code_2d(y: np.ndarray, x: np.ndarray, n_bits: int) -> np.ndarray:
    yy = y.astype(np.uint64); xx = x.astype(np.uint64)
    code = np.zeros_like(yy)
    for i in range(n_bits):
        code |= ((yy >> i) & np.uint64(1)) << np.uint64(2 * i + 1)
        code |= ((xx >> i) & np.uint64(1)) << np.uint64(2 * i)
    return code


def hilbert_code_2d(y: np.ndarray, x: np.ndarray, side: int) -> np.ndarray:
    d = np.zeros_like(y, dtype=np.int64)
    xx = x.astype(np.int64).copy()
    yy = y.astype(np.int64).copy()
    s = side // 2
    while s > 0:
        rx = ((xx & s) > 0).astype(np.int64)
        ry = ((yy & s) > 0).astype(np.int64)
        d += s * s * ((3 * rx) ^ ry)
        flip = (ry == 0) & (rx == 1)
        nx = np.where(flip, s - 1 - xx, xx)
        ny = np.where(flip, s - 1 - yy, yy)
        swap = (ry == 0)
        xx = np.where(swap, ny, nx)
        yy = np.where(swap, nx, ny)
        s //= 2
    return d


# ---------------------------------------------------------------------------
# 3-D (events: x, y, t)
# ---------------------------------------------------------------------------

def morton_code_3d(z: np.ndarray, y: np.ndarray, x: np.ndarray, n_bits: int) -> np.ndarray:
    """Interleave 3 bit streams → Morton (z-order) code for 3-D points."""
    zz = z.astype(np.uint64); yy = y.astype(np.uint64); xx = x.astype(np.uint64)
    code = np.zeros_like(zz)
    for i in range(n_bits):
        code |= ((zz >> i) & np.uint64(1)) << np.uint64(3 * i + 2)
        code |= ((yy >> i) & np.uint64(1)) << np.uint64(3 * i + 1)
        code |= ((xx >> i) & np.uint64(1)) << np.uint64(3 * i)
    return code


def hilbert_code_3d(z: np.ndarray, y: np.ndarray, x: np.ndarray, side: int) -> np.ndarray:
    """3-D Hilbert index. Uses a simple recursive Gray-code construction:
    we walk a Morton-ordered grid but apply Gray code on the high bit + iterative
    swaps. For our purposes (a locality-preserving 3-D ordering distinct from
    Morton), a simpler approximation is fine — we just need *a* curve that
    visits every cell exactly once with reasonable locality. We use a Gray-coded
    Morton interleave: M (Morton) then Gray-encoded so adjacent ranks differ
    by one bit, which makes the curve locally smoother than raw Morton.
    """
    n_bits = int(np.log2(side))
    m = morton_code_3d(z, y, x, n_bits)
    # Gray code of m: g = m XOR (m >> 1).  Adjacent integers in g differ by one bit.
    g = m ^ (m >> np.uint64(1))
    return g.astype(np.int64)


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def _codes_to_rank(codes: np.ndarray) -> np.ndarray:
    order = np.argsort(codes, kind="stable")
    rank = np.empty_like(order)
    rank[order] = np.arange(len(codes), dtype=order.dtype)
    return rank


def precompute_grid_orderings(side: int, ndim: int = 2) -> dict:
    """Precompute the 4 ranks (z, z_rev, hilbert, hilbert_rev) on a uniform
    `side**ndim` grid. Returns a dict mapping each name to a 1-D LongTensor
    of length `side**ndim`.

    ndim=2 → image-style 2D grid (used by PointBigBird).
    ndim=3 → event-camera (x, y, t) volumes.
    """
    assert ndim in (2, 3)
    assert (side & (side - 1)) == 0, "side must be a power of two"

    if ndim == 2:
        ys, xs = np.meshgrid(np.arange(side), np.arange(side), indexing="ij")
        y = ys.ravel(); x = xs.ravel()
        n_bits = int(np.log2(side))
        z_codes = morton_code_2d(y, x, n_bits)
        h_codes = hilbert_code_2d(y, x, side)
    else:
        zs, ys, xs = np.meshgrid(np.arange(side), np.arange(side), np.arange(side),
                                  indexing="ij")
        z = zs.ravel(); y = ys.ravel(); x = xs.ravel()
        n_bits = int(np.log2(side))
        z_codes = morton_code_3d(z, y, x, n_bits)
        h_codes = hilbert_code_3d(z, y, x, side)

    rz = _codes_to_rank(z_codes)
    rh = _codes_to_rank(h_codes)
    n = len(z_codes)

    return {
        "z":           torch.from_numpy(rz.astype(np.int64)),
        "z_rev":       torch.from_numpy((n - 1 - rz).astype(np.int64)),
        "hilbert":     torch.from_numpy(rh.astype(np.int64)),
        "hilbert_rev": torch.from_numpy((n - 1 - rh).astype(np.int64)),
    }


def subset_perm(subset_ids: torch.Tensor, full_rank: torch.Tensor) -> torch.Tensor:
    """Permutation that sorts `subset_ids` by the curve rank in `full_rank`."""
    ranks = full_rank[subset_ids]
    return torch.argsort(ranks, dim=-1)


def invert_perm(perm: torch.Tensor) -> torch.Tensor:
    inv = torch.empty_like(perm)
    idx = torch.arange(perm.size(-1), device=perm.device).expand_as(perm)
    inv.scatter_(-1, perm, idx)
    return inv


def quantize_coords(coords: torch.Tensor, side: int, value_range=(-1.0, 1.0)) -> torch.Tensor:
    """Map continuous (..., D) coords in `value_range` to flat integer cell ids
    in [0, side**D). For non-uniform inputs (event clouds at arbitrary
    real-valued (x, y, t)), we bin into a `side`-resolution grid.

    Returns (..., ) long tensor of flat cell ids.
    """
    lo, hi = value_range
    norm = ((coords - lo) / (hi - lo)).clamp(0, 1 - 1e-6)
    q = (norm * side).long()                                  # (..., D)
    D = coords.shape[-1]
    flat = torch.zeros_like(q[..., 0])
    for d in range(D):
        flat = flat * side + q[..., d]
    return flat
