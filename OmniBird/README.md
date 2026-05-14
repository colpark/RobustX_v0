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

## How to run (real datasets)

We provide automated download + conversion in `datasets/download.py`.

### Option A — EventScape (CARLA driving simulation, the primary target)

Project page: [https://rpg.ifi.uzh.ch/RAMNet.html](https://rpg.ifi.uzh.ch/RAMNet.html).
Per-Town zips are ~10–50 GB each.

```bash
cd OmniBird

# 1) List available subsets / URLs
python -m datasets.download urls

# 2) Download one subset (smallest = Town01_train, recommended first)
python -m datasets.download eventscape \
    --out ./data/eventscape_raw \
    --subsets Town01_train

# 3) (Optional) Add more subsets — they accumulate in ./data/eventscape_raw
python -m datasets.download eventscape \
    --out ./data/eventscape_raw \
    --subsets Town01_validation Town02_train

# 4) Convert raw EventScape (events.h5 + rgb/*.png + semantic/*.png) into
#    OmniBird's per-clip layout (events_*.npy + label_*.txt + rgb_*.png).
#    Requires h5py + pillow: pip install h5py pillow
python -m datasets.download convert_eventscape \
    --raw ./data/eventscape_raw \
    --out ./data/eventscape_omnibird
```

Then in the training notebook, swap the synthetic dataset for:
```python
from datasets import EventScapeDataset
base_train = EventScapeDataset("./data/eventscape_omnibird", mode="events_only")
```

### Option B — CIFAR10-DVS (small, ~1 GB, fast iteration)

Event-camera recording of CIFAR-10 images, 10-class classification.
Hosted on Figshare:
[https://figshare.com/articles/dataset/CIFAR10-DVS_New/4724671](https://figshare.com/articles/dataset/CIFAR10-DVS_New/4724671).

```bash
cd OmniBird

# 1) Download the raw zips (auto-discovers via Figshare API)
python -m datasets.download cifar10_dvs --out ./data/cifar10_dvs_raw

# 2) Unzip each archive; you'll get class-named folders containing AEDAT files
#    (this varies by Figshare version; expected layout below)
#    raw/
#      airplane/*.aedat
#      automobile/*.aedat
#      ...

# 3) Convert AEDAT → OmniBird per-clip layout
python -m datasets.download convert_cifar10_dvs \
    --raw ./data/cifar10_dvs_raw \
    --out ./data/cifar10_dvs_omnibird
```

Then use the same `EventScapeDataset` loader on the converted directory:
```python
base_train = EventScapeDataset("./data/cifar10_dvs_omnibird", mode="events_only")
```
(The loader's format is dataset-agnostic; the converter writes the same layout.)

### Expected per-clip layout (after conversion)

```
data/<dataset>_omnibird/
  clip_00000/
    events_0.npy        # (N_raw, 4): x_int, y_int, t_us, polarity ∈ {0, 1}
    label_0.txt         # integer class label
    rgb_0.png           # optional (EventScape only) — paired RGB frame for multimodal
    events_1.npy
    label_1.txt
    ...
  clip_00001/
    ...
```

Each `clip_NNNNN` contains one or more event windows. `EventScapeDataset`
iterates over every `(events_*.npy, label_*.txt)` pair across all clips.

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
