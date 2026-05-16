# PBB_baselines — PointBigBird JEPA baselines

Two self-contained sub-folders, each fully copy-paste portable. Either folder can be sent / shared independently — they have no cross-dependencies and no external package dependency beyond `torch`, `torchvision`, `numpy`, `matplotlib`.

## Folders

| Folder | Modality | Status |
|---|---|---|
| `pbb_cifar10/` | 40% sparse pixel pool of CIFAR-10 (24 bits of content per token) | **Recommended baseline.** Works without further architectural defenses. Reproduces strong linear probe. |
| `pbb_cifar10_dvs/` | CIFAR10-DVS event-camera data (1 bit of content per token) | **Stress test.** Same recipe on a much lower per-token entropy modality. Provided as the cleanest baseline for studying JEPA collapse dynamics on sparse modalities; expected to underperform the CIFAR-10 variant by a wide margin and not be a competitive event-camera SSL method on its own. |

## What each sub-folder contains

```
pbb_*/                              # one self-contained folder per variant
├── pbb_core.py                     # standalone model library (~580 lines)
├── pbb_<variant>.ipynb             # training notebook (training + linear probe)
├── architecture.html               # visual architecture overview (open in browser)
└── README.md                       # extensive documentation
```

## Shared architecture (one diagram, both variants)

```
sparse input (pixels or events)
   ↓
Tokenizer: signal_proj + pos_proj(γ_Fourier(coord))
   ↓
BigBird sparse encoder × 6 layers
   • per-layer random space-filling-curve permutation (z / z-rev / hilbert / hilbert-rev)
   • block-sparse attention (block=8, window=1, n_random=2, n_global=2)
   ↓
per-token features
   ↓                              ↓
gather @ ctx positions    gather @ tgt positions (EMA encoder)
   ↓                              ↓
                          TargetCenter (DINO-EMA mean sub)
                                  ↓
                          per-token LayerNorm  →  h_tgt
   ↓
predictor: dense Transformer over [ctx ‖ mask+pos(tgt_coord)]
   • pos_symmetric=True
   • reads off at target positions
   ↓
h_pred

loss = smooth_L1(h_pred, h_tgt)
EMA target (stop-grad, 0.999 → 1.0)
```

**No "pool" / cross-attention readout** between encoder and JEPA targets. Direct per-token gather on both context and target sides. This is the i-JEPA-faithful recipe for sparse inputs.

## Anti-collapse mechanism — important for sparse modalities

**Target-side EMA centering + per-token LayerNorm**, applied *before* the smooth-L1 loss. This is the key piece that lets the recipe operate without VICReg or contrastive terms.

- **Per-token LayerNorm alone** (the canonical i-JEPA recipe) is sufficient for high-content modalities like ImageNet patches but **insufficient** for sparse inputs. It can't see across the batch dimension.
- **DINO-style EMA centering** subtracts a running batch mean from each target before LN. This removes the "every target points in the same direction across the batch" failure mode that per-token LN cannot prevent.

```python
target_center.update(h_tgt_raw)                              # EMA: c ← 0.9·c + 0.1·mean_batch(h_tgt_raw)
h_tgt = F.layer_norm(target_center(h_tgt_raw), (D,))         # center, then LN per token
```

This is the only "non-canonical" piece; everything else is straight i-JEPA. No VICReg, no centering+sharpening softmax, no contrastive negatives.

## When to use which variant

- **Reproduce the CIFAR-10 baseline:** `pbb_cifar10/`. Should reliably reach a non-trivial linear-probe accuracy. This is the clean reference implementation of the working PBB v2 setup.
- **Study JEPA dynamics on sparse modalities:** `pbb_cifar10_dvs/`. Same recipe at low per-token entropy. Useful as a diagnostic — if naked PBB collapses on events but works on CIFAR-10, that pinpoints content density as the load-bearing variable.
- **Want strong event-camera SSL numbers:** **neither.** The naked PBB recipe is not competitive on raw events. Use one of the parent codebase's variants (`OmniBird/notebooks/cifar10_dvs_eventmae.ipynb` for MAE reconstruction, `cifar10_dvs_temporal.ipynb` for causal temporal-JEPA), or voxelize first.

## Requirements

```bash
pip install torch torchvision numpy matplotlib
```

That's it. No `omnibird/`, no other internal packages.

## Citation lineage

| Origin | Contribution |
|---|---|
| i-JEPA (Assran et al., CVPR 2023) | Overall JEPA recipe: stop-grad + EMA target + multi-block masking + smooth-L1 in latent space |
| Point-JEPA (Saito & Poovvancheri, WACV 2025) | Sparse-input adaptations, target LayerNorm protocol |
| DINO (Caron et al., ICCV 2021) | Target centering via EMA of batch mean |
| BigBird (Zaheer et al., NeurIPS 2020) | Block-sparse attention with window + random + global blocks |

The novel integration: BigBird sparse encoder + anchor-KNN multi-block masking for sparse pools + target centering before LN + multi-curve SFC permutations per layer for globally-connected receptive field through depth.
