# PBB-style JEPA on CIFAR10-DVS (event-camera variant)

A direct port of the PBB JEPA recipe to event-camera data. This folder is **copy-paste portable** — drop it anywhere with `torch` and `numpy` installed and run `pbb_cifar10_dvs.ipynb`. The accompanying `pbb_core.py` is the same self-contained library used by the CIFAR-10 variant; no parent-package dependency.

> **Honest caveat first.** Each event-camera token carries 1 bit of polarity (vs. 24 bits of RGB per pixel in the CIFAR-10 variant). This is past the per-token entropy threshold where the naked PBB recipe (direct gather + target centering + per-token LN + EMA, no further regularizer) is most stable. Provided here as the **cleanest possible event baseline** for analyzing JEPA dynamics on a sparse low-entropy modality. If linear probe is weak, the right next step is one of:
> - **Voxelize first** (3-channel event-image: ON-count / time-surface / OFF-count) and train a standard MAE — see `notebooks/cifar10_dvs_eventmae.ipynb` in the parent OmniBird codebase.
> - **Switch the pretext task to causal temporal prediction** — see `notebooks/cifar10_dvs_temporal.ipynb`.

## TL;DR

- **Pretext task:** i-JEPA-style multi-block masking on the events of one CIFAR10-DVS clip. Predict target-event features from disjoint context events.
- **Token granularity:** per-event tokens. Each event = `(x, y, t, polarity)`, embedded as `signal_proj(one_hot_polarity) + pos_proj(γ_Fourier(x, y, t))`.
- **Anti-collapse:** same as the CIFAR-10 variant — DINO-style EMA target centering + per-token LayerNorm + EMA target.
- **Architecture:** same `pbb_core` library. Only the dataset, coord_dim, signal_dim, and pool/block sizes differ.

## Architecture (end to end)

```
Event clip                                           (B, K_pool, 4)   ← (x, y, t, polarity)
        │
        ├─ cap/pad at K_POOL = 8192
        ├─ 3D Hilbert-curve sort
        ▼
Tokenizer:  signal_proj(one_hot_polarity)  +  pos_proj(γ_Fourier(x, y, t))
        │
        ▼
BigBird sparse encoder                              (B, 8192, D)
   • 6 layers
   • per-layer random space-filling-curve permutation
                (z, z-rev, hilbert, hilbert-rev — 3D curves)
   • block-sparse attention (block=8, window=1, n_random=2, n_global=2)
   • per-event key-padding mask threaded through every layer
        │
        ▼
per-event encoder features
        │
        ├── gather at TARGET event positions         (B, 2048, D)
        │           │     4 KNN-disjoint blocks × 512 events
        │           ▼
        │   TargetCenter  ← DINO-style EMA of per-feature batch mean
        │           │
        │           ▼
        │   per-token LayerNorm                      → h_tgt
        │
        └── gather at CONTEXT event positions        (B, 2048, D)
                    │     K_CTX=2048, one KNN block disjoint from targets
                    ▼
            Predictor input
                    │
                    ▼
            Dense Transformer over [ctx_tokens ‖ mask_tokens@target_coords]
              • mask_tokens = pos_proj(γ_Fourier(tgt_xyt)) + learned mask_token
              • pos_symmetric = True
              • 4 layers
                    │
                    ▼
            read off at target positions             → h_pred

                  loss = smooth_L1(h_pred, h_tgt)
                  EMA target (stop-grad, 0.999 → 1.0)
```

**No "pool" / cross-attention readout** between encoder and JEPA targets. Direct per-event gather, same as the CIFAR-10 variant.

## What's different from `pbb_cifar10`

| | CIFAR-10 (RGB pixels) | CIFAR10-DVS (events) |
|---|---|---|
| `coord_dim` | 2 (y, x) | 3 (x, y, t) |
| `signal_dim` | 3 (RGB) | 2 (one-hot ON/OFF polarity) |
| Pool size `K_POOL` | 410 pixels | 8192 events |
| Context size `K_CTX` | 100 | 2048 |
| Targets per sample | 200 (4 × 50) | 2048 (4 × 512) |
| Encoder backbone | same | same |
| Predictor | same | same |
| Target centering / LN / EMA / loss | same | same |
| Grid `side` | 32 | 64 |
| Per-token content | 24 bits (RGB) | 1 bit (polarity) |
| Epochs (default) | 1000 | 200 |

The architectural template is identical; only the dataset adapter and a few dimensionality knobs change.

## Hyperparameters

| Parameter | Value | Notes |
|---|---|---|
| Pool size | 8192 events | cap or pad to this length |
| Context size | 2048 events | One anchor-grown KNN block in (x, y, t) |
| Target blocks | 4 × 512 = 2048 events | 4 KNN-disjoint blocks |
| `d_model` | 256 | |
| Encoder | 6-layer BigBird (block=8, window=1, n_random=2, n_global=2) | |
| Predictor | 4-layer dense Transformer, `d_pred=192`, `pos_symmetric=True` | |
| EMA momentum | 0.999 → 1.0 linear ramp | |
| Optim | AdamW, lr=2e-4, wd=0, warmup 5 epochs, cosine decay | |
| Batch size | 32 | |
| Epochs | 200 | |
| Probe interval | every 10 epochs, 2 epochs each; final = 30 ep from best | |

## How to run

```bash
# Open pbb_cifar10_dvs.ipynb in Jupyter and run all cells.
# Auto-resumes from ./checkpoints_pbb_cifar10_dvs/ if present.
```

Requires the CIFAR10-DVS clips at `../data/cifar10_dvs_omnibird/`. Each `clip_*` subdirectory should contain `events_0.npy` (N×4 array of `[x, y, t, polarity]`) and `label_0.txt`.

## What to expect

This is the **harder modality** for naked JEPA. Per-token entropy is roughly an order of magnitude lower than the CIFAR-10 variant. The recipe should still train without collapse (centering + LN + EMA take care of that), but the linear probe ceiling is expected to be lower because the supervision signal per event is much sparser:

| | Expected | Risk |
|---|---|---|
| `tgt_std` | Stable around 0.5 – 0.8 (lower than CIFAR-10's 0.6 – 0.9) | Goes below 0.2 — target collapse, centering not active |
| `cos(h_pred, h_tgt)` | Climbs to 0.5 – 0.75 over many epochs | Saturates to 1.0 within 50 steps — full collapse |
| Linear probe | 15 – 30% (above 10% chance, below 40%) | At 10% chance — features carry no class info |

For state-of-the-art CIFAR10-DVS, switching pretext task is more impactful than tuning this recipe further:
- **Causal temporal-JEPA**: predict future-window features from past — aligns with the sensor's physics.
- **Voxelized event-MAE**: 3-channel event image + standard MAE — provably collapse-free since the target is data.

Both are in the parent `OmniBird/notebooks/` directory.

## Files in this folder

| File | Purpose |
|---|---|
| `pbb_core.py` | Self-contained library, identical to the CIFAR-10 variant's. Contains all model code. |
| `pbb_cifar10_dvs.ipynb` | Training notebook. End-to-end: dataset → models → training loop → linear probe. |
| `architecture.html` | Visual architecture overview, comparison with the CIFAR-10 variant and with i-JEPA, novel-contributions analysis. |
| `README.md` | This document. |

## Acknowledgements

Same lineage as the CIFAR-10 variant: PointBigBird (PBB) recipe + i-JEPA loss family + Point-JEPA / DINO target-side techniques. The specific port to event-camera data is what's new here, with full awareness that the per-token content drop (24 bits → 1 bit) puts naked JEPA closer to its working limit.
