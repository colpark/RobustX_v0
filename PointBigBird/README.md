# PointBigBird-JEPA

A new sparse-input self-supervised pre-training architecture. Replaces the
OmniField + perceiver-cascade backbone used in `JEPA_CIFAR10.ipynb` (v1–v8)
with a **per-token transformer encoder** that uses

- **Space-filling-curve serialization** (z-order / Hilbert / reverses), one
  ordering chosen at random per encoder layer, so spatial neighbours become
  sequence neighbours under *some* ordering at every layer.
- **BigBird block-sparse self-attention** (Zaheer et al. 2020) — window +
  global + random blocks, implemented in plain PyTorch via `index_select` /
  `gather`. Verified equivalent to dense MHA when configured for full
  attention.
- **i-JEPA-style JEPA** on top — exactly the v8 multi-block disjoint masking
  recipe, with a small dense predictor that injects mask tokens at the
  target coordinates.

The "gist" of v1–v8 we kept: EMA target encoder, DINO-style centering,
smooth-L1 distance, multi-block disjoint masking. The "machinery" we
dropped: learnable `latent_pos`, Gaussian attention bias, deterministic
soft-pool target, aux variance loss, predictor warmup — those were
workarounds for the latent-set representation; with one token per point
they are not needed (targets are *direct lookups* in the pool, since
`target_coords ⊂ pool_coords` by v8 construction).

## Layout

```
PointBigBird/
├── README.md
├── train.py                 # CLI training entry point
├── pbb/
│   ├── __init__.py
│   ├── config.py            # PBBConfig (all hparams)
│   ├── serialization.py     # z-order, Hilbert, subset_perm, invert_perm
│   ├── attention.py         # MultiHeadAttention, BigBirdSparseAttention
│   ├── model.py             # Tokenizer, EncoderBlock, PBBEncoder, PBBPredictor
│   ├── data.py              # PBBChunkCIFAR10, build_loaders
│   ├── jepa.py              # EMA, centering, loss, diagnostics
│   └── utils.py             # checkpoint save, param counting
├── tests/
│   ├── test_serialization.py    # 4 orderings bijective; subset round-trip
│   └── test_attention.py        # BigBird == dense in full mode; padding; speed
└── notebooks/
    ├── PBB_JEPA_walkthrough.ipynb   # visual walkthrough (no training)
    └── PBB_JEPA_train.ipynb         # training notebook (mirrors train.py + plots)
```

## Quick start

```bash
cd PointBigBird
python -m tests.test_serialization
python -m tests.test_attention
python train.py --epochs 100
```

## Key numbers (CIFAR-10, 32×32)

| Knob                          | Value                                |
|-------------------------------|--------------------------------------|
| Pool                          | 410 px (40 % of 1024)                |
| Context (train)               | 100 px (1 contiguous block)          |
| Target blocks                 | 4 × 50 px (disjoint from context)    |
| `d_model` / `n_layers_enc`    | 256 / 6                              |
| `n_heads` / `dim_head`        | 8 / 32                               |
| BigBird `block_size`          | 32                                   |
| BigBird `window` / random / global | 1 / 2 / 2  → 7 blocks per query |
| Predictor                     | 4 layers, dim 192, 6 heads           |
| EMA momentum                  | 0.999 → 1.000                        |
| Center momentum (DINO)        | 0.9                                  |

## What's tested

`tests/test_serialization.py`
- Morton + Hilbert encodings are bijective on 4×4, 8×8, 16×16, 32×32 grids
- `subset_perm` round-trips with `invert_perm`
- Batched (4, 100)-shape subset permutations are each valid permutations
- 4 orderings on (64, 100) batch sub-50 ms per iter

`tests/test_attention.py`
- BigBird shape matches input
- **BigBird in `equivalent_to_dense=True` mode matches `MultiHeadAttention` exactly**
  (max |Δ| < 1e-5) — including under padding masks
- Sparse mode differs meaningfully from dense (max |Δ| > 0.05)
- Sparse mode is non-deterministic across forwards (random block resampling)
- Speed benchmark: sparse vs dense on (B=4, N=1024, D=128)

## Notebooks

**`notebooks/PBB_JEPA_walkthrough.ipynb`** — visual walkthrough (20 cells):
1. Tokenization (RGB + γ features)
2. The 4 curve orderings on a 32×32 grid
3. Per-sample subset orderings on a 100-pixel context
4. BigBird attention pattern heatmaps (N=128 toy, N=1024 toy)
5. Equivalence sanity: BigBird(`equivalent_to_dense=True`) == MHA
6. Per-layer random shuffling (log which ordering each layer picked)
7. Full forward pass on one CIFAR-10 image
8. v8-style 5-panel data viz (context + 4 disjoint target blocks)
9. ASCII architecture diagram

**`notebooks/PBB_JEPA_train.ipynb`** — training notebook (19 cells):
1. Setup + config overrides
2. Data loaders (train / train_eval / test)
3. Build models (encoder, EMA target, predictor, target center)
4. Optimizer + cosine LR + EMA momentum schedule
5. Resume from `pbb_last.pt` if present
6. Embedded linear probe (uses `train_eval_loader` for K_HALF input)
7. Training loop with per-step diagnostics and per-epoch probe
8. Plot training curves (loss, cosine, probe accuracy)
9. Collapse diagnostics (‖g‖, σ_batch(h_pred), σ_token(g), ‖center‖, …)
