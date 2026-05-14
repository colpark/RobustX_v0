# OmniBird

Multimodal extension of [PointBigBird-JEPA](../PointBigBird/) for **event cameras**
in robotics simulation, with a Phase 2 path to fully multimodal training via
**ICMR (Iterative Cross-Modal Refinement)** borrowed from OmniField.

This folder is **self-contained** — it does not modify or depend on `PointBigBird/`.
Code is forked from PBB and adapted for 3-D event coordinates (x, y, t).

## Why

The PBB-JEPA architecture is designed for *sparse* point sets. Event cameras
produce exactly that: an asynchronous stream of (x, y, t, polarity) tuples,
typically 10⁴–10⁶ events per second, with no underlying dense grid. Treating
each event as one token, the architecture is a natural fit — no event-frame
aggregation, no voxel-grid quantization, no information loss from binning.

Robotics-flavored sim datasets like **EventScape** (CARLA driving) provide
synchronized events + RGB + depth + semantic segmentation, which makes them
ideal for both:
1. **Single-modality** pretraining on events (this delivery).
2. **Multimodal** pretraining with cross-modal target signal (Phase 2 via ICMR).

## Dataset choice

We picked **EventScape** as the canonical robotics + simulation + multimodal
event dataset (Gehrig et al., RAL 2021,
[https://rpg.ifi.uzh.ch/RAMNet.html](https://rpg.ifi.uzh.ch/RAMNet.html)):

- CARLA-generated → genuine robotics simulation.
- Events stored as raw `(x, y, t, polarity)` lists per clip — native sparse
  representation preserved (no voxel-grid aggregation).
- Synchronized with RGB frames, depth, semantic segmentation, IMU.
- Public, well-documented.
- Subsets are usable; full dataset is ~50 GB.

For development without the EventScape download we provide
[`datasets/synthetic.py`](datasets/synthetic.py): a procedurally-generated
10-class event dataset with class-specific spatio-temporal trajectories. It
exercises the entire pipeline end-to-end and lets you verify the model trains
and the probe works before downloading the real data.

## Layout

```
OmniBird/
├── README.md
├── omnibird/                    # the package
│   ├── __init__.py              # all exports
│   ├── config.py                # OmniBirdConfig (event-aware defaults)
│   ├── serialization.py         # 2-D + NEW 3-D Morton/Hilbert curves
│   ├── attention.py             # BigBird block-sparse + dense MHA (verbatim from PBB)
│   ├── model.py                 # Tokenizer (generic), EncoderBlock, OmniBirdEncoder, OmniBirdPredictor
│   ├── data.py                  # OmniBirdEventDataset — JEPA masking on event clouds
│   ├── jepa.py                  # EMA, loss (cosine / smooth_l1), diagnostics
│   ├── probe.py                 # LinearProbe + AttnPoolHead + quick_probe
│   ├── icmr.py                  # NEW: Iterative Cross-Modal Refinement (Phase 2)
│   └── utils.py                 # checkpoint helpers
├── datasets/
│   ├── synthetic.py             # 10-class synthetic events for dev (no download)
│   └── eventscape.py            # EventScape loader (per-clip events_*.npy + label_*.txt)
├── notebooks/
│   └── OmniBird_train_synthetic.ipynb   # end-to-end training on synthetic events
└── docs/
    └── omnibird_approach.html   # design document
```

## What changes vs. PointBigBird

| Component         | PointBigBird (CIFAR-10, 2-D pixels)         | OmniBird (events, 3-D)                    |
|-------------------|---------------------------------------------|-------------------------------------------|
| Coordinate dim    | 2 (y, x)                                    | **3 (x, y, t)**                            |
| Signal dim        | 3 (RGB)                                     | **1 (polarity)**                           |
| Serialization     | 2-D Morton + 2-D Hilbert (+ reverses)        | **3-D Morton + 3-D Hilbert** (Gray-encoded) |
| Per-sample budget | 410 px per image                            | **2048 events per clip** (configurable)     |
| Block sampling    | K nearest in (y, x) Euclidean               | **K nearest in (x, y, t) Euclidean**        |
| Tokenizer         | `signal_proj(rgb) + pos_proj(γ(y,x))`       | `signal_proj(pol) + pos_proj(γ(x,y,t))`     |
| BigBird           | block_size=8 (~13 blocks at K_CTX=100)       | block_size=8 (~128 blocks at N=1024)        |
| Anti-degradation  | cosine loss, EMA cap, no centering, probe-best, early-stop | **inherited verbatim**       |

## How to run (synthetic, no download)

```bash
cd OmniBird
# In a Python env with torch + numpy:
python -c "
import sys; sys.path.insert(0, '.')
from omnibird import OmniBirdConfig
from datasets import build_synthetic_loaders
cfg = OmniBirdConfig()
train, train_eval, test = build_synthetic_loaders(cfg, n_train=200, n_test=50)
b = next(iter(train))
print({k: v.shape if hasattr(v, 'shape') else v for k, v in b.items() if k != 'label'})
"
```

Or open `notebooks/OmniBird_train_synthetic.ipynb` and run top-to-bottom.

## How to run (EventScape, real)

1. Download EventScape from [https://rpg.ifi.uzh.ch/RAMNet.html](https://rpg.ifi.uzh.ch/RAMNet.html).
2. Convert each clip to the directory layout expected by
   `datasets/eventscape.py`:
   ```
   root/
     clip_000/
       events_0.npy        # (N_raw, 4):  x_int, y_int, t_us, polarity ∈ {0,1}
       label_0.txt         # integer class label
       rgb_0.png           # optional, for multimodal mode
       events_1.npy
       label_1.txt
       ...
     clip_001/
       ...
   ```
3. Swap the synthetic dataset for `EventScapeDataset(root, mode="events_only")`
   in the training notebook.

## Multimodal Phase 2 — ICMR design

The single-modality pipeline above produces per-token features `g ∈ ℝᴮˣᴷˣᴰ`
for events. For multimodal training:

1. Run **one OmniBirdEncoder per modality** (events, RGB, etc.) — produces
   `g_events, g_rgb, ...` each of shape `(B, K_m, D)`.
2. Apply **`ICMR(n_latents, modalities)`** from `omnibird/icmr.py`: a shared
   learnable latent set L cross-attends to each modality's tokens, with
   N iterations of refinement.
3. The JEPA loss is reformulated as: predict each modality's target features
   from the *shared latents* L. This forces L to encode multimodal-consistent
   information.

The ICMR module supports **fleximodal** masking via per-sample
`modality_present` booleans — at inference, the latents iterate normally,
just skipping cross-attention to absent modalities.

See `docs/omnibird_approach.html` for the full design rationale and
diagrams.

## Status

- ✅ Single-modality event-only OmniBird-JEPA, end-to-end (this delivery)
- ✅ Synthetic 10-class event dataset for dev
- ✅ EventScape loader (real dataset)
- ✅ 3-D Morton / Hilbert serialization with per-sample lookup
- ✅ ICMR module (Phase 2 building block)
- 🟡 Multimodal training notebook (Phase 2 — code in place, notebook pending)
- 🟡 Multimodal benchmark on EventScape (Phase 2)
