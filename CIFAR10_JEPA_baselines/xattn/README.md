# xattn — cross-attention readout JEPA on CIFAR-10 (40% sparse pool)

The cross-attention readout recipe. Self-contained: ships with the
`omnibird/` package bundled in this folder so no external repo install
is needed.

## Architecture

```
sample → Tokenizer(signal_proj + pos_proj∘γ_Fourier)
       → BigBird per-event encoder (sparse attention)
                                     │
                                     ↓
       tokenizer skip-connection: pool_kv = encoder_out + tokenizer_out
                                     │
                                     ↓
       LocalCrossAttention pool at FPS centroid queries
       (scores -= α · ‖q_coord − k_coord‖²)
                                     │
                                     ├─ pool-at-context  ─ predictor input
                                     │
                                     └─ pool-at-target  ─ target_center → LN → h_tgt
                                                                          │
       predictor (PerceiverPredictor): pool-at-context  ────────────►   h_pred
                                                                          │
                                              smooth-L1 loss + EMA target encoder
```

**Why cross-attention pool instead of per-pixel direct gather:**

- pbb_v2 gathers target features directly at target-pixel positions —
  works because each pixel has 24 bits of RGB content (high token
  entropy).
- xattn lets the encoder consume per-pixel events but reads out at
  FPS-derived centroid queries via cross-attention. The pool acts as a
  bottleneck that adapts to lower-entropy modalities (event cameras,
  point clouds) where direct per-pixel gather wouldn't work.

**Key fixes baked into this recipe:**

| Piece | What it does |
|---|---|
| **Tokenizer skip-connection at pool input** | `pool_kv = encoder(events) + tokenizer(s, c)` — guarantees per-event variation regardless of encoder collapse |
| **Symmetric cross-attention pool** on both context and target | The predictor's job becomes "pool-at-target ← pool-at-context", same statistic at different locations (true i-JEPA contract) |
| **`LocalCrossAttention`** with `scores -= α‖q_coord − k_coord‖²` | Spatial-locality bias on the pool, prevents one query from attending everywhere uniformly |
| **`FixedPosEmbedder` shared** | NeRF γ → frozen orthogonal projection, single shared instance everywhere |
| **DINO-style target centering** before per-token LN | The actual anti-collapse mechanism — same as pbb_v2 |

## Files

| File | Purpose |
|---|---|
| `omnibird/` | Bundled package (10 modules, ~2700 lines) containing the encoder, predictor, attention primitives, FPS+masking data pipeline, JEPA loss + EMA, position embedding. The notebook imports from this. |
| `xattn_cifar10.ipynb` | Full training pipeline + linear probe. |

## Running

```bash
jupyter notebook xattn_cifar10.ipynb
```

The notebook adds the local folder to `sys.path` so `from omnibird import ...`
picks up the bundled package, not any system-installed omnibird.

To start fresh, delete the `./checkpoints_omnibird_xattn_cifar10/`
directory or set `RESUME = False` in the relevant cell.

## When to prefer xattn over pbb_v2

- The modality is **genuinely sparse** with low per-event entropy
  (event-camera 1-bit polarity, point clouds with no per-point
  features). Direct gather at target events provides ~0 bits of useful
  signal; cross-attention pool adapts to local density.
- The downstream task needs **per-region readout** at arbitrary
  locations (not just at observed pixels). The pool queries can be any
  FPS sample or grid sample.

On the 40% sparse CIFAR-10 specifically, pbb_v2 is the stronger
baseline. xattn is here as the alternative design point that scales to
sparser modalities.
