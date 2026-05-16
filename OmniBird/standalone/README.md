# Standalone PBB-style JEPA notebooks

This folder is **copy-paste portable**. It does not depend on the parent `omnibird/` package. Drop the folder anywhere with PyTorch + torchvision installed and run the notebooks.

## Contents

| File | Purpose |
|---|---|
| `pbb_core.py` | Self-contained library: Tokenizer, BigBird sparse attention, encoder, dense predictor, EMA target, DINO-style target centering, smooth-L1 loss, Hilbert/Morton space-filling curves. ~580 lines. |
| `pbb_cifar10.ipynb` | PBB v2 recipe on CIFAR-10 (40% sparse pixel pool). 410 per-pixel tokens, 200 targets per sample, direct gather. The high-confidence baseline. |
| `pbb_cifar10_dvs.ipynb` | Same recipe ported to CIFAR10-DVS event-camera data. 8192-event pool, 2048 targets per sample. Lower per-token entropy (1-bit polarity vs 24-bit RGB) makes this closer to JEPA's limit; provided as the cleanest possible baseline. |
| `_build_pbb_cifar10.py` | Generator script for `pbb_cifar10.ipynb`. |
| `_build_pbb_cifar10_dvs.py` | Generator script for the DVS notebook. |

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

**Anti-collapse mechanism:** target centering before per-token LayerNorm. DINO-style EMA of the per-feature batch mean is subtracted from the gathered target features before per-token LN. This removes the "all targets point in the same direction across the batch" minimum that pure per-token LN cannot prevent. No VICReg, no other regularizer.

## Usage

```python
# In Jupyter:
%cd standalone
# Open pbb_cifar10.ipynb (or the DVS variant) and run all cells.
```

The notebooks resume from their respective `./checkpoints_*/` directories. To start fresh, delete the checkpoint dir or set `RESUME = False` in cell 5.

## Hyperparameters (PBB v2 defaults)

| | CIFAR-10 | CIFAR10-DVS |
|---|---|---|
| pool size | 410 pixels (40% of 32x32) | 8192 events |
| context size | 100 pixels | 2048 events |
| target blocks | 4 × 50 = 200 pixels | 4 × 512 = 2048 events |
| epochs | 1000 | 200 |
| batch | 64 | 32 |
| lr | 2e-4 | 2e-4 |
| EMA | 0.999 → 1.0 | 0.999 → 1.0 |
| d_model | 256 | 256 |
| encoder | 6-layer BigBird (block=8, window=1, n_random=2, n_global=2) | same |
| predictor | 4-layer dense Transformer, d_pred=192, pos_symmetric=True | same |

## Why two variants?

- **CIFAR-10 (pbb_cifar10):** each pixel token carries 24 bits of RGB content. This is the regime where JEPA's "stop-grad + EMA + target centering + per-token LN" recipe works without further architectural defenses. The high-confidence path to a strong linear-probe baseline.

- **CIFAR10-DVS (pbb_cifar10_dvs):** each event token carries 1 bit of polarity. This is past the entropy threshold where naked JEPA reliably works on sparse modalities. Provided here as the cleanest possible event-side baseline; if the linear probe is weak, switch to `cifar10_dvs_temporal.ipynb` (causal time-axis JEPA) or `cifar10_dvs_eventmae.ipynb` (MAE reconstruction) in the parent `notebooks/` directory.
