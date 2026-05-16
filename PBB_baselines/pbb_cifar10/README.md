# PBB-style JEPA on CIFAR-10 (40% sparse pixel pool)

A self-contained PointBigBird-style Joint Embedding Predictive Architecture for sparse-input self-supervised learning on CIFAR-10. This folder is **copy-paste portable** — no external dependencies beyond `torch`, `torchvision`, `numpy`, `matplotlib`.

> **Standalone contract.** Drop this folder anywhere on a machine with PyTorch installed and run `pbb_cifar10.ipynb`. The accompanying `pbb_core.py` provides every model/loss/utility the notebook imports. No reference to a parent package or sibling folder.

## TL;DR — what this is and why

* **Pretext task:** i-JEPA-style multi-block masking on a 40% sparse pool of CIFAR-10 pixels. The encoder predicts target-pixel features from a disjoint context block of pixels.
* **Why sparse:** modality realism. The encoder must work with a partial, irregularly-sampled view of the image — the same situation point-cloud / event-camera / LiDAR encoders face. CIFAR-10 sparse pool is a cleaner, faster testbed for the same architectural recipe.
* **Why this works without VICReg or any explicit anti-collapse regularizer:** target-side **DINO-style EMA centering + per-token LayerNorm** + multi-block masking + EMA target is sufficient when per-token content is rich (24 bits of RGB per pixel). The trivial constant-output JEPA minimum is far from initialization in this content regime.

## Architecture (one path, end to end)

```
40% sampled pixels per image                       (B, 410, 5)   ← (y, x, R, G, B)
        │
        ▼
Tokenizer:  signal_proj(RGB)  +  pos_proj(γ_Fourier(y, x))
        │
        ▼
BigBird sparse encoder                              (B, 410, D)
   • 6 layers
   • each layer:    random space-filling-curve permutation
                    (z / z-rev / hilbert / hilbert-rev)
                    → block-sparse attention (block=8, window=1,
                       n_random=2, n_global=2)
                    → FFN → residual + LayerNorm
        │
        ▼
per-pixel encoder features
        │
        ├── gather at TARGET pixel positions        (B, 200, D)
        │           │
        │           ▼
        │   TargetCenter  ← DINO-style EMA of per-feature batch mean
        │           │
        │           ▼
        │   per-token LayerNorm                     → h_tgt
        │
        └── gather at CONTEXT pixel positions       (B, 100, D)
                    │
                    ▼
            Predictor input
                    │
                    ▼
            Dense Transformer over [ctx_tokens ‖ mask_tokens@target_coords]
              • mask_tokens = pos_proj(γ_Fourier(tgt_coords)) + learned mask_token
              • pos_symmetric = True (γ also added to ctx tokens)
              • 4 layers
                    │
                    ▼
            read off at target positions
                    │
                    ▼
            proj_out                                 → h_pred

                 loss = smooth_L1(h_pred, h_tgt)
                 EMA target (stop-grad, momentum 0.999 → 1.0)
```

**Note on terminology:** there is **no "pool" or learned compression layer** between the encoder and the JEPA targets. We directly index the encoder's per-token output at target and context positions. The compression layer that earlier iterations of this codebase included (a "cross-attention readout" / "centroid pool") has been removed for this architecture — for content-rich tokens like RGB pixels, direct gather preserves more signal than learned compression.

## Why this architecture (vs. alternatives)

| Choice | Why |
|---|---|
| **40% sparse pool** (instead of full 1024 pixels) | Models real sparse-input modalities (point clouds, events). Forces the encoder to handle missing structure. |
| **i-JEPA multi-block masking** | Standard, well-validated SSL pretext for sparse inputs. 4 KNN-disjoint blocks of 50 target pixels per sample → 200 supervised positions. |
| **Direct per-pixel target gather** | The encoder already does spatial mixing across 6 BigBird layers. Each output token contextualizes its neighborhood. No need for a second learned aggregation step. |
| **DINO-style target centering + per-token LN before loss** | Per-token LN alone allows "all targets point in the same batch direction" collapse. Subtracting the EMA batch mean before LN removes this minimum. No VICReg / contrastive / centering+sharpening softmax needed. |
| **BigBird sparse attention** | Scalable to large pools (we run at 410 tokens, but the same code handles thousands). Multi-curve serialization per layer gives global receptive field through depth. |
| **EMA target with stop-grad** | Standard JEPA recipe. Momentum ramp 0.999 → 1.0 over training. |
| **No VICReg, no centering+sharpening, no contrastive** | We don't need anti-collapse regularization at this content density. The recipe is i-JEPA-faithful. |

## What's novel vs. plain i-JEPA

This setup keeps i-JEPA's loss family and overall structure, but adapts it to **sparse, irregularly-sampled inputs**. The specific additions:

1. **Multi-curve space-filling-curve permutations per encoder layer.** Each layer randomly picks one of `{z, z_rev, hilbert, hilbert_rev}` to re-shuffle the token sequence before BigBird's block-sparse attention. This converts BigBird's locally-windowed attention into a globally-connected receptive field through depth, without giving up the per-layer compute efficiency. Standard i-JEPA on dense images doesn't need this because the image grid is regular and ViT just uses dense attention.

2. **Mini-PointNet-free per-pixel tokens for sparse inputs.** PointNet-style aggregation collapses local-neighborhood structure too aggressively for 24-bit RGB pixels. Direct per-pixel tokens + BigBird preserves more local content than patch-based pooling, at the cost of more tokens.

3. **Anchor + KNN-disjoint multi-block masking.** Rather than rectangular blocks (i-JEPA on images), we pick 4 random anchors in the pool and grow each block by k-nearest pool members in 2D coordinates, with subsequent blocks excluding previously-claimed pixels. This is the natural generalization of i-JEPA's block masking to non-grid sparse inputs.

4. **DINO-style target centering as the anti-collapse mechanism** (vs i-JEPA's per-token LN alone). Per-token LN is necessary but insufficient for sparse inputs — it can't see across the batch dimension. Centering before LN removes the "constant direction across batch" minimum. This single ~5-line addition replaces what VICReg / centering+sharpening softmax / variance-covariance regularizers do in heavier recipes.

## Hyperparameters

| Parameter | Value | Notes |
|---|---|---|
| Image size | 32 × 32 | CIFAR-10 |
| Pool fraction | 0.4 | 410 of 1024 pixels per image |
| Context size `K_CTX` | 100 pixels | One anchor-grown KNN block from remaining pool |
| Target blocks | 4 × 50 = 200 pixels | 4 KNN-disjoint blocks |
| `d_model` | 256 | |
| Encoder | 6 layers, BigBird (block=8, window=1, n_random=2, n_global=2) | |
| Predictor | 4 layers, dense Transformer, `d_pred=192`, `pos_symmetric=True` | |
| EMA momentum | 0.999 → 1.0 (linear ramp over epochs) | |
| Optim | AdamW, `lr=2e-4`, `wd=0`, warmup 5 ep, cosine decay | |
| Batch size | 64 | |
| Epochs | 1000 | |
| Probe | every 10 epochs, 2 epochs each; final probe = 30 epochs from best ckpt | |

## How to run

```bash
# Just open in Jupyter and run all cells in pbb_cifar10.ipynb.
# The notebook auto-resumes from ./checkpoints_pbb_cifar10/ if present.
# Set RESUME=False in cell 5 to start fresh.
```

Outputs:
- `checkpoints_pbb_cifar10/pbb_last.pt` — latest checkpoint
- `checkpoints_pbb_cifar10/pbb_best.pt` — best-loss checkpoint
- Live diagnostics printed per training step: `loss`, `pred_std`, `tgt_std`, `cos`, `lr`, `m`
- Linear probe every 10 epochs during training (2-epoch fit); final 30-epoch linear probe at the end

## Collapse diagnostics — what to watch for

| Signal | Healthy | Collapse |
|---|---|---|
| `tgt_std` | gradually settles around 0.6 – 0.9 | rapidly approaches 0 (target features colinear) |
| `pred_std` | grows over training to roughly track `tgt_std` | stays near 0, or oscillates wildly |
| `cos(h_pred, h_tgt)` | climbs gradually 0 → 0.5 – 0.85 over many epochs | saturates to 1.0 within ~25 steps |
| Probe acc | climbs over epochs, well above chance (10%) by epoch 50 | stays at chance |

If `cos` jumps to 1.0 in the first 25 steps, training has reached the trivial constant minimum and target centering is somehow not active. If `tgt_std` is healthy (>0.5) but probe stays at chance, the encoder is learning features that aren't class-discriminative — try training longer.

## Files in this folder

| File | Purpose |
|---|---|
| `pbb_core.py` | Self-contained library (~580 lines): Tokenizer, BigBird, encoder, predictor, target centering, EMA, loss, Hilbert/Morton SFC. |
| `pbb_cifar10.ipynb` | Training notebook. End-to-end: dataset → models → training loop → linear probe. |
| `architecture.html` | Visual architecture overview, comparison vs. i-JEPA, novel-contributions analysis, ideas to strengthen. Open in any browser. |
| `README.md` | This document. |

## Citation / acknowledgements

This setup descends from the PointBigBird (PBB) JEPA v2 implementation. Specific design choices borrow from:

- **i-JEPA** (Assran et al., CVPR 2023) — overall JEPA recipe, multi-block masking, smooth-L1 in latent space.
- **Point-JEPA** (Saito & Poovvancheri, WACV 2025) — sparse-input adaptation, target LayerNorm protocol.
- **DINO** (Caron et al., ICCV 2021) — target centering via EMA of batch mean.
- **BigBird** (Zaheer et al., NeurIPS 2020) — block-sparse attention with window + random + global blocks.
