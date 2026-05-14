"""Space-filling curve serialization for 2D point sets.

Provides z-order (Morton) and Hilbert orderings, plus their reverses.
Each ordering is represented as a *rank* array: `rank[i]` = position of
pixel `i` (in row-major order) along the curve.

For a fixed grid (e.g. CIFAR-10 32×32), we precompute the 4 rank arrays
once with `precompute_grid_orderings(image_size)`. For any subset of pool
pixel ids, sorting the pool by `rank[subset_pixel_ids]` gives the curve
order on the subset — O(K log K), independent of the full grid size.
"""
from __future__ import annotations

import numpy as np
import torch


# ---------------------------------------------------------------------------
# Codes
# ---------------------------------------------------------------------------

def morton_code_2d(y: np.ndarray, x: np.ndarray, n_bits: int) -> np.ndarray:
    """Interleave bits of (y, x) → Morton (z-order) code."""
    yy = y.astype(np.uint64)
    xx = x.astype(np.uint64)
    code = np.zeros_like(yy)
    for i in range(n_bits):
        code |= ((yy >> i) & np.uint64(1)) << np.uint64(2 * i + 1)
        code |= ((xx >> i) & np.uint64(1)) << np.uint64(2 * i)
    return code


def hilbert_code_2d(y: np.ndarray, x: np.ndarray, side: int) -> np.ndarray:
    """Standard 2D Hilbert curve index for points on a `side × side` grid.
    `side` must be a power of two and ≥ max(y, x) + 1.
    """
    d = np.zeros_like(y, dtype=np.int64)
    xx = x.astype(np.int64).copy()
    yy = y.astype(np.int64).copy()
    s = side // 2
    while s > 0:
        rx = ((xx & s) > 0).astype(np.int64)
        ry = ((yy & s) > 0).astype(np.int64)
        d += s * s * ((3 * rx) ^ ry)
        # Rotate / reflect quadrants:
        #   if ry == 0:
        #     if rx == 1:  x = s - 1 - x;  y = s - 1 - y
        #     swap x, y
        flip = (ry == 0) & (rx == 1)
        nx = np.where(flip, s - 1 - xx, xx)
        ny = np.where(flip, s - 1 - yy, yy)
        swap = (ry == 0)
        xx = np.where(swap, ny, nx)
        yy = np.where(swap, nx, ny)
        s //= 2
    return d


def _codes_to_rank(codes: np.ndarray) -> np.ndarray:
    """rank[i] = position of element i in the sorted order."""
    order = np.argsort(codes, kind="stable")
    rank = np.empty_like(order)
    rank[order] = np.arange(len(codes), dtype=order.dtype)
    return rank


# ---------------------------------------------------------------------------
# Grid precomputation
# ---------------------------------------------------------------------------

def precompute_grid_orderings(image_size: int) -> dict:
    """Compute 4 rank arrays for a full `image_size × image_size` grid.

    Returns:
        dict with keys 'z', 'z_rev', 'hilbert', 'hilbert_rev', each mapping
        to a 1-D `torch.LongTensor` of length `image_size**2`. Entry `i` is
        the rank (0-based position) of the pixel at row-major index `i` in
        that ordering.
    """
    ys, xs = np.meshgrid(np.arange(image_size), np.arange(image_size), indexing="ij")
    y = ys.ravel()
    x = xs.ravel()
    side = 1 << int(np.ceil(np.log2(max(image_size, 2))))
    n_bits = int(np.log2(side))

    z_codes = morton_code_2d(y, x, n_bits)
    h_codes = hilbert_code_2d(y, x, side)
    rz = _codes_to_rank(z_codes)
    rh = _codes_to_rank(h_codes)
    n = len(z_codes)

    return {
        "z":           torch.from_numpy(rz.astype(np.int64)),
        "z_rev":       torch.from_numpy((n - 1 - rz).astype(np.int64)),
        "hilbert":     torch.from_numpy(rh.astype(np.int64)),
        "hilbert_rev": torch.from_numpy((n - 1 - rh).astype(np.int64)),
    }


# ---------------------------------------------------------------------------
# Subset ordering
# ---------------------------------------------------------------------------

def subset_perm(subset_pixel_ids: torch.Tensor, full_rank: torch.Tensor) -> torch.Tensor:
    """Permutation that sorts `subset_pixel_ids` by curve order.

    Args:
        subset_pixel_ids: (..., K) long — row-major pixel ids in [0, N)
        full_rank:        (N,)     long — rank for each full-grid pixel

    Returns:
        perm: (..., K) long. `perm[..., i]` is the index into the *subset*
              whose pixel ranks i-th in the curve order. Use as:
                  sorted_subset = subset[perm]
    """
    ranks = full_rank[subset_pixel_ids]      # (..., K)
    return torch.argsort(ranks, dim=-1)


def invert_perm(perm: torch.Tensor) -> torch.Tensor:
    """Inverse of a permutation along the last dim."""
    inv = torch.empty_like(perm)
    idx = torch.arange(perm.size(-1), device=perm.device).expand_as(perm)
    inv.scatter_(-1, perm, idx)
    return inv


def precompute_subset_orderings(
    subset_pixel_ids: torch.Tensor,
    grid_ranks: dict,
) -> dict:
    """For a subset of pixel ids, build the 4 permutations and their inverses.

    Args:
        subset_pixel_ids: (..., K) long
        grid_ranks:       dict[str, (N,) long] from `precompute_grid_orderings`

    Returns:
        dict with keys = ordering names, values = dict(perm=..., inverse=...)
        each of shape (..., K).
    """
    out = {}
    for name, rank in grid_ranks.items():
        p = subset_perm(subset_pixel_ids, rank)
        out[name] = {"perm": p, "inverse": invert_perm(p)}
    return out
