"""Test serialization correctness and speed."""
import sys, os, time
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import numpy as np
import torch
from pbb.serialization import (
    morton_code_2d, hilbert_code_2d,
    precompute_grid_orderings, subset_perm, invert_perm,
    precompute_subset_orderings,
)


def test_morton_4x4():
    # Hand-verified: z-order on 4x4 grid (n_bits=2)
    # Visiting order (row-major y,x): (0,0)(0,1)(0,2)(0,3)(1,0)... → z codes
    ys, xs = np.meshgrid(np.arange(4), np.arange(4), indexing="ij")
    y, x = ys.ravel(), xs.ravel()
    codes = morton_code_2d(y, x, n_bits=2)
    # Expected codes:  (0,0)→0  (0,1)→1  (1,0)→2  (1,1)→3  (0,2)→4  (0,3)→5
    # (row-major flat positions: 0=(0,0), 1=(0,1), 2=(0,2), 3=(0,3), 4=(1,0)...)
    expected_at_index_0 = 0
    expected_at_index_1 = 1
    expected_at_index_4 = 2     # (1,0)
    expected_at_index_5 = 3     # (1,1)
    expected_at_index_2 = 4     # (0,2)
    assert int(codes[0]) == expected_at_index_0
    assert int(codes[1]) == expected_at_index_1
    assert int(codes[4]) == expected_at_index_4
    assert int(codes[5]) == expected_at_index_5
    assert int(codes[2]) == expected_at_index_2
    print("  morton_code_2d on 4x4 grid: OK")


def test_hilbert_bijective_8x8():
    side = 8
    ys, xs = np.meshgrid(np.arange(side), np.arange(side), indexing="ij")
    codes = hilbert_code_2d(ys.ravel(), xs.ravel(), side)
    # Hilbert must be a bijection from N² points to {0..N²-1}
    assert sorted(codes.tolist()) == list(range(side * side)), "hilbert not bijective"
    print(f"  hilbert_code_2d on {side}x{side} grid: bijective ({side*side} unique codes)")


def test_grid_orderings_shape():
    g = precompute_grid_orderings(32)
    assert set(g.keys()) == {"z", "z_rev", "hilbert", "hilbert_rev"}
    for name, t in g.items():
        assert t.shape == (1024,)
        assert t.dtype == torch.int64
        # Each rank array must be a permutation of [0..1023]
        assert torch.equal(torch.sort(t).values, torch.arange(1024))
    print("  precompute_grid_orderings(32): all 4 orderings are bijective permutations of [0..1023]")


def test_subset_perm_correctness():
    g = precompute_grid_orderings(8)
    # Pick a random subset of pool pixels
    torch.manual_seed(0)
    subset = torch.randperm(64)[:20]
    p = subset_perm(subset, g["z"])
    # Sort subset by z-rank manually
    ranks = g["z"][subset]
    expected_perm = torch.argsort(ranks)
    assert torch.equal(p, expected_perm)
    # Inverse should round-trip
    inv = invert_perm(p)
    assert torch.equal(p[inv], torch.arange(20))
    assert torch.equal(inv[p], torch.arange(20))
    print("  subset_perm + invert_perm: round-trip OK on 20-pixel subset of 8x8 grid")


def test_subset_perm_batched():
    g = precompute_grid_orderings(32)
    torch.manual_seed(0)
    subset = torch.stack([torch.randperm(1024)[:100] for _ in range(4)])
    p = subset_perm(subset, g["hilbert"])
    assert p.shape == (4, 100)
    # Each row independently permutes [0..99]
    for b in range(4):
        assert torch.equal(torch.sort(p[b]).values, torch.arange(100))
    print("  subset_perm: batched (4, 100) shape OK and each row is a valid permutation")


def test_subset_orderings_dict():
    g = precompute_grid_orderings(32)
    subset = torch.randperm(1024)[:100]
    out = precompute_subset_orderings(subset, g)
    assert set(out.keys()) == {"z", "z_rev", "hilbert", "hilbert_rev"}
    for name, d in out.items():
        assert set(d.keys()) == {"perm", "inverse"}
        assert d["perm"].shape == (100,)
        assert d["inverse"].shape == (100,)
        # Round trip
        assert torch.equal(d["perm"][d["inverse"]], torch.arange(100))
    print("  precompute_subset_orderings: all 4 orderings + inverses produced correctly")


def test_speed_subset_perm():
    g = precompute_grid_orderings(32)
    B, K = 64, 100
    subset = torch.stack([torch.randperm(1024)[:K] for _ in range(B)])
    t0 = time.perf_counter()
    for _ in range(100):
        for name in g:
            _ = subset_perm(subset, g[name])
    dt = (time.perf_counter() - t0) / 100
    print(f"  speed: 4 orderings × batch ({B}, {K}) → {dt*1000:.2f} ms/iter")
    assert dt < 0.05, "subset_perm should be under 50ms per batch"


if __name__ == "__main__":
    test_morton_4x4()
    test_hilbert_bijective_8x8()
    test_grid_orderings_shape()
    test_subset_perm_correctness()
    test_subset_perm_batched()
    test_subset_orderings_dict()
    test_speed_subset_perm()
    print("\nAll serialization tests passed ✓")
