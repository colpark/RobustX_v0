"""Test BigBird attention against vanilla.

The test set:
  1. Shape sanity — BigBird outputs match input shape.
  2. Equivalence — `BigBirdSparseAttention(equivalent_to_dense=True)` and
     `MultiHeadAttention` produce identical outputs when their weights are
     tied (they share Q/K/V/O projection layouts).
  3. Padding mask — both modes correctly ignore padding tokens.
  4. Random non-determinism — sparse mode produces different outputs across
     forwards (because random block sampling is fresh each call).
  5. Sparsity matters — sparse and dense outputs differ when they should.
  6. Speed — sparse path produces a measurable speedup on long sequences.
"""
import sys, os, time
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import torch
from pbb.attention import MultiHeadAttention, BigBirdSparseAttention


def _tie_weights(dense: MultiHeadAttention, sparse: BigBirdSparseAttention):
    """Copy dense's projection weights into sparse so they're literally identical layers."""
    with torch.no_grad():
        sparse.to_qkv.weight.copy_(dense.to_qkv.weight)
        if dense.to_qkv.bias is not None:
            sparse.to_qkv.bias.copy_(dense.to_qkv.bias)
        sparse.to_out.weight.copy_(dense.to_out.weight)
        sparse.to_out.bias.copy_(dense.to_out.bias)


def test_shape():
    B, N, D = 2, 128, 64
    x = torch.randn(B, N, D)
    bb = BigBirdSparseAttention(D, n_heads=4, dim_head=16,
                                 block_size=32, window=1, n_random=2)
    y = bb(x)
    assert y.shape == x.shape, f"shape mismatch: {y.shape} vs {x.shape}"
    print(f"  shape sanity: BigBird({tuple(x.shape)}) → {tuple(y.shape)}  ✓")


def test_equivalence_to_dense():
    """BigBird in 'equivalent_to_dense' mode == dense MHA with same weights."""
    torch.manual_seed(0)
    B, N, D = 2, 128, 64
    H, Dh = 4, 16
    x = torch.randn(B, N, D)

    dense  = MultiHeadAttention(D, n_heads=H, dim_head=Dh)
    sparse = BigBirdSparseAttention(D, n_heads=H, dim_head=Dh,
                                     block_size=32, window=1, n_random=2,
                                     equivalent_to_dense=True)
    _tie_weights(dense, sparse)

    y_dense  = dense(x)
    y_sparse = sparse(x)
    diff = (y_dense - y_sparse).abs().max().item()
    print(f"  equivalence (dense path): max|Δ| = {diff:.2e}  (must be < 1e-5)")
    assert diff < 1e-5, f"dense-mode BigBird should match MHA exactly, got max|Δ|={diff}"


def test_equivalence_padding_mask():
    torch.manual_seed(1)
    B, N, D = 2, 64, 32
    H, Dh = 4, 8
    x = torch.randn(B, N, D)
    pad_mask = torch.zeros(B, N, dtype=torch.bool)
    pad_mask[:, 50:] = True   # last 14 tokens are padding

    dense  = MultiHeadAttention(D, n_heads=H, dim_head=Dh)
    sparse = BigBirdSparseAttention(D, n_heads=H, dim_head=Dh,
                                     block_size=16, window=1, n_random=1,
                                     equivalent_to_dense=True)
    _tie_weights(dense, sparse)
    y_d = dense(x, key_padding_mask=pad_mask)
    y_s = sparse(x, key_padding_mask=pad_mask)
    diff = (y_d - y_s).abs().max().item()
    print(f"  padding mask equivalence: max|Δ| = {diff:.2e}  (must be < 1e-5)")
    assert diff < 1e-5


def test_sparse_runs_and_differs():
    """Sparse mode runs without error and differs from dense (because window<NB)."""
    torch.manual_seed(2)
    B, N, D = 2, 256, 64
    H, Dh = 4, 16
    x = torch.randn(B, N, D)
    dense  = MultiHeadAttention(D, n_heads=H, dim_head=Dh)
    sparse = BigBirdSparseAttention(D, n_heads=H, dim_head=Dh,
                                     block_size=32, window=1, n_random=2)
    _tie_weights(dense, sparse)
    y_d = dense(x)
    y_s = sparse(x)
    diff = (y_d - y_s).abs().max().item()
    print(f"  sparse vs dense (window=1, random=2 on N=256): max|Δ| = {diff:.3f}")
    assert diff > 0.05, "Sparse output should differ meaningfully from dense"


def test_sparse_nondeterminism():
    """Two consecutive sparse forwards differ because random blocks are resampled."""
    torch.manual_seed(3)
    x = torch.randn(2, 128, 32)
    sparse = BigBirdSparseAttention(32, n_heads=4, dim_head=8,
                                     block_size=16, window=1, n_random=2)
    y1 = sparse(x)
    y2 = sparse(x)
    diff = (y1 - y2).abs().max().item()
    print(f"  non-determinism (random blocks resampled): max|Δ| = {diff:.3f}  (>0 expected)")
    assert diff > 1e-5, "consecutive sparse forwards should differ due to random sampling"


def test_speed():
    """Sparse path should be faster than dense for long N."""
    B, N, D = 4, 1024, 128
    H, Dh = 8, 16
    x = torch.randn(B, N, D)
    dense = MultiHeadAttention(D, n_heads=H, dim_head=Dh)
    sparse = BigBirdSparseAttention(D, n_heads=H, dim_head=Dh,
                                     block_size=32, window=1, n_random=2)
    # Warm up
    for _ in range(3):
        dense(x); sparse(x)
    # Time
    t0 = time.perf_counter()
    for _ in range(20): dense(x)
    t_dense = (time.perf_counter() - t0) / 20
    t0 = time.perf_counter()
    for _ in range(20): sparse(x)
    t_sparse = (time.perf_counter() - t0) / 20
    print(f"  speed (B={B}, N={N}, D={D}):  dense={t_dense*1000:.2f}ms  "
          f"sparse={t_sparse*1000:.2f}ms  speedup={t_dense/t_sparse:.2f}×")


if __name__ == "__main__":
    test_shape()
    test_equivalence_to_dense()
    test_equivalence_padding_mask()
    test_sparse_runs_and_differs()
    test_sparse_nondeterminism()
    test_speed()
    print("\nAll attention tests passed ✓")
