# ModelNet40 — PointNet vs RoPE/HRR aggregator benchmark

A minimal head-to-head test of whether the per-patch aggregator algorithm
matters on a canonical point-cloud classification task, isolating that
single axis.

## What changes between the two methods

**Only the patch aggregator.** Everything else — FPS+KNN patch
construction, NeRF γ(centroid) position embedding added to patch tokens,
6-layer ViT encoder with vanilla MHA, mean-pool + MLP classifier head,
optimizer, schedule, augmentations — is identical.

| | PointNet | RoPE/HRR |
|---|---|---|
| Per-point op | `MLP(concat(rel_coord, signal))` | `MLP(signal)` then rotate by rel_coord |
| Cross-point op | **max-pool** over K | **sum** over K |
| Position-content interaction | learned in MLP | multiplicative Fourier modulation |

With per-point signal set to a constant 1 (vanilla ModelNet40 has no
per-point features), the RoPE aggregator reduces to literally the
truncated Fourier transform of the patch's point density:

    S = signal_proj(1) · Σ_i exp(jω · rel_pos_i)

This is the regime where the spectral-aggregation prior is most
distinguishable from PointNet's max-pool feature selection.

## Files

| File | Purpose |
|---|---|
| `bench_core.py` | Shared modules: `FlexibleViTEncoder`, `ModelNet40Classifier`, ModelNet40 download/load, FPS+KNN cache, augmentations. Imports the existing `PointNetPatchifier` (from `vit_fps_core`) and `RoPEPatchifier` (from `rope_patch_core`) so this is a benchmark wrapper, not a reimplementation. |
| `_build_modelnet40_bench.py` | Generator script for the notebook. |
| `modelnet40_bench.ipynb` | The full benchmark — download → FPS+KNN cache → train PointNet variant → train RoPE variant → head-to-head plots + per-class breakdown. |

## Running

```bash
cd OmniBird/modelnet40_bench
jupyter notebook modelnet40_bench.ipynb
# (downloads ~400MB of ModelNet40 HDF5 on first run; cached thereafter)
```

Two 100-epoch training runs at batch=32, D_MODEL=192, 6-layer encoder.
On a single modern GPU each run takes ~30–60 min depending on FPS+KNN
precompute caching. Both runs together: about 1–2 hours.

## Hyperparameters

| | Value |
|---|---|
| N_INPUT (points per cloud after subsample) | 1024 |
| N_PATCHES (FPS centroids) | 64 |
| K_NEIGH (K-NN per patch) | 32 |
| D_MODEL | 192 |
| Encoder layers / heads / dim_head | 6 / 6 / 32 |
| Optimizer | AdamW, lr 1e-3, wd 1e-4 |
| Schedule | linear warmup 5ep + cosine to 0 |
| Epochs / batch | 100 / 32 |
| Augmentations | rotation around z, scale ∼U(0.8, 1.25), σ=0.01 jitter |
| RoPE base frequency | 30.0 (for intra-patch positions in [-1, 1]) |

## Interpretation guide

The notebook reports `Δ = best_RoPE − best_PointNet` in percentage points
at the end.

| Δ | Interpretation |
|---|---|
| > +1.5 pts | RoPE/HRR clearly wins; the aggregator matters here. Worth scaling to ShapeNet segmentation + other point-cloud benchmarks. |
| -1.0 to +1.5 pts | Effective tie within run-to-run noise. Aggregator choice doesn't matter at this scale on this benchmark. Try sparser or per-point-readout tasks. |
| < -1.0 pts | PointNet wins. Max-pool's feature-selection prior is helping; RoPE's spectral prior is wrong for this regime. |

This is a smoke test, not a definitive benchmark. A real publication-grade
claim would require: multiple seeds, fine-tuning of both methods, and
extension to ShapeNet part-seg + ScanNet for per-point readout tasks
where the aggregator distinction should survive more strongly.

## Cache

FPS+KNN per cloud is computed once and cached to `./cache_fps_knn_modelnet/`.
First run takes ~5 minutes on CPU; subsequent runs are instant. Cache is
keyed by `(n_samples, n_input, n_patches, k_neigh, seed, tag)` so any
config change invalidates and recomputes automatically.
