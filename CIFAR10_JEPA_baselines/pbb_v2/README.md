# PBB v2 — per-pixel BigBird JEPA on CIFAR-10 (40% sparse pool)

The PointBigBird v2 recipe. Self-contained, copy-paste portable; only
depends on PyTorch + torchvision.

## Architecture

```
sample → Tokenizer(signal_proj + pos_proj∘γ_Fourier) → BigBird encoder
   → per-token features
   → gather at target positions          gather at context positions
   → target_center (DINO-EMA mean sub)   → predictor input
   → per-token LayerNorm                 ↓
   → h_tgt                              dense Transformer with
                                        mask-tokens at target coords
                                        (pos_symmetric=True)
                                         ↓
                                        h_pred
   ↓________________ smooth-L1 ____________↓
                    loss = smooth_L1(h_pred, h_tgt)
                    EMA target encoder (stop-grad)
```

**Anti-collapse mechanism:** target centering before per-token LayerNorm.
DINO-style EMA of the per-feature batch mean is subtracted from the
gathered target features before per-token LN. This removes the "all
targets point in the same direction across the batch" minimum that pure
per-token LN cannot prevent. No VICReg, no other regularizer.

## Files

| File | Purpose |
|---|---|
| `pbb_core.py` | Self-contained library (~580 lines): Tokenizer, BigBird sparse attention, encoder, dense predictor, EMA target, DINO-style target centering, smooth-L1 loss, Hilbert/Morton space-filling curves. |
| `pbb_cifar10.ipynb` | Full training pipeline. 410 per-pixel tokens per sample, 200 target tokens, direct gather. |

## Hyperparameters

| | Value |
|---|---|
| Pool size | 410 pixels (40% of 32×32) |
| Context size | 100 pixels |
| Target blocks | 4 × 50 = 200 pixels |
| Epochs | 1000 |
| Batch | 64 |
| LR | 2e-4 |
| EMA | 0.999 → 1.0 |
| d_model | 256 |
| Encoder | 6-layer BigBird (block=8, window=1, n_random=2, n_global=2) |
| Predictor | 4-layer dense Transformer, d_pred=192, pos_symmetric=True |

## Running

```bash
jupyter notebook pbb_cifar10.ipynb
```

To start fresh, delete `./checkpoints_pbb_cifar10/` or set `RESUME = False`
in cell 5.
