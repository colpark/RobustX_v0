# Correspondence Bench — A Difficulty-Controlled Multimodal Benchmark

A synthetic benchmark designed to **measure whether a self-supervised
learning recipe can discover fine-grained cross-modal correspondences**
— and whether better correspondence learning translates into better
downstream classification, segmentation, and tracking.

The benchmark ships two dataset versions that share the same generative
core but differ in whether the scene has motion:

| | Dataset | What you get | Best for |
|---|---|---|---|
| **A** | **Linked Primitives (static)** | One pair of RGB images: two camera views of a 3D primitive scene at one instant | Classification, segmentation, cross-view dense correspondence — the canonical static-multimodal task |
| **B** | **Linked Primitives Video (spatiotemporal)** | One pair of short videos: two camera views of the same primitive scene over time, plus a GIF | Video classification, spatiotemporal segmentation, tracking, motion-conditioned correspondence, optical flow |

Both share the same difficulty operating points (`easy / basic / hard /
extreme / adversarial`) and the same per-primitive ID space, so an SSL
method can be evaluated on both with comparable settings.

---

## 1. Why this benchmark exists

The standard multimodal SSL evaluation toolkit has gaps:

1. **Global statistics suffice** on many image-pair datasets — you can
   reach high accuracy without doing any fine correspondence.
2. **No ground-truth correspondences** in real datasets, so we evaluate
   via downstream proxies instead of probing the SSL representation
   directly.
3. **Difficulty isn't parameterized** — you can't dial it up to test
   where a method breaks.
4. **Multi-scale + multi-temporal structure isn't controlled** —
   primitives can't be made to span 10× in spatial scale or 10× in
   temporal scale on demand.

This benchmark fixes all four:

1. **Label depends on the latent scene, not the observations.** The
   class label is computed from primitive-set latents, not from rendered
   pixels. You can't solve it by global statistics — you must recover a
   sufficient statistic of the latents, which requires correspondence.
2. **Ground-truth correspondences are returned with every sample.**
   Every primitive has a stable integer ID present in every view and
   every frame. We ship helpers that return matched-pair lists at all
   three granularities (§ 3).
3. **A vector of independent difficulty knobs.** Five named operating
   points span EASY → ADVERSARIAL; you can pin everything except one
   knob for single-axis ablations.
4. **Scale and motion are explicit knobs.** Spatial `scale_range` and
   temporal `motion_amplitude` are independently dialed.

---

## 2. The generative-model contract

Both datasets follow the same schema:

```
Latent scene  z = {entity_1, ..., entity_N}, each with possibly a trajectory
                                                      ↓
Modality A render  X_A = g_A(z)               Modality B render  X_B = g_B(z)
                                                      ↓
                       Label  y = φ(z)
                       Correspondences  C = {(i_X, j_Y) : matches}
```

Crucially:
- `g_A` and `g_B` are **deterministic** given `z` (and per-modality
  nuisance: jitter, noise, style transform).
- `y = φ(z)` is computable **only from the latents**. To predict `y`, a
  model must recover (a sufficient statistic of) `z`, which forces it
  to solve the cross-modal alignment.
- The correspondence ground truth `C` is **returned with every sample**.

---

## 3. The three kinds of correspondence (B only — A has just the first)

Because every primitive carries a stable `pid` everywhere it's visible:

1. **Cross-view at same time**  (pid, view=A, τ) ↔ (pid, view=B, τ)
   — same primitive seen from two cameras. The *only* kind in dataset A.
   - Helper: `cross_view_pairs_at_time(video, t_idx)` for dataset B;
     `correspondence_pairs(rA, rB)` for dataset A.

2. **Cross-time within a view**  (pid, view=A, τ₁) ↔ (pid, view=A, τ₂)
   — same primitive at two different times in the same camera. The
   ground truth for **tracking** and **optical flow** within one
   modality.
   - Helper: `cross_time_pairs_within_view(video, view, t1, t2)`.
   - Bonus: `trajectories_for_view(video, view)` returns the full (T, 2)
     keypoint path for every pid in one view.

3. **Cross-view AND cross-time**  (pid, view=A, τ₁) ↔ (pid, view=B, τ₂)
   — the union; same pid wherever it appears. The most flexible ground
   truth, and the one a fully general spatiotemporal-multimodal SSL
   recipe should be able to recover.
   - Use the same `pid` membership across both views' `ids` arrays;
     intersect by pid as needed.

---

## 4. Dataset A — Linked Primitives (static)

**File:** `linked_primitives.py`

### Generative process

Each scene contains N "linked" primitives placed in 3D space:

| Attribute | Range / domain |
|---|---|
| `shape_id` | one of `circle, square, triangle, plus, star, diamond, hexagon, cross, pentagon, octagon` |
| `color_idx` | one of K configurable RGB palette entries |
| `pos_3d` | uniform in [-0.9, 0.9]³ |
| `size` | log-uniform in `scale_range` (e.g. (0.02, 0.25)) |

Rendered from two camera viewpoints (rotations of the 3D scene about
the y-axis by ±`view_disparity_deg / 2`, with a small random shared
x-tilt). Painter's algorithm with depth sort and occlusion.

### Style asymmetry for view B

| Style | Description |
|---|---|
| `rgb` | identical rendering style to view A |
| `grayscale_B` | view B rendered then desaturated to greyscale |
| `edges_B` | view B rendered then run through a Sobel edge detector and inverted |

Style asymmetry forces the model to learn modality-invariant
representations rather than pixel-matching.

### What you get per render

```python
out = generator.render(scene, view="A")
out["rgb"]   # (H, W, 3) uint8 image
out["seg"]   # (H, W) int32 — per-pixel primitive ID; -1 background
out["kpts"]  # (N_total, 2) float32 — projected centers
out["vis"]   # (N_total,) bool — visible in this view?
out["ids"]   # (N_total,) int32 — primitive IDs in canonical order
```

### Built-in labels (`compute_label`)

| `kind` | What it computes |
|---|---|
| `count_modulo_K` | number of linked primitives mod K |
| `has_pair` | 1 iff there's a pair with same shape but different colours |
| `n_distinct_pairs` | count of distinct (shape, colour) tuples |

### Downstream tasks (use Dataset A for these)

| Task | Input | Output | Ground truth from |
|---|---|---|---|
| **Image-pair classification** | (img_A, img_B) | one class label | `compute_label(scene)` |
| **Per-modality segmentation** | img_A (or img_B) | per-pixel pid in {-1, 0..N-1} | `out["seg"]` |
| **Cross-view keypoint retrieval** | features at primitive centres in A and B | for each kpt in A: top-k match in B | `correspondence_pairs(rA, rB)` |
| **Dense cross-modal alignment** | (img_A, img_B) | for each (x_A, y_A): its (x_B, y_B) | from `seg_A == pid ↔ seg_B == pid` |

All four tasks have closed-form ground truth.

---

## 5. Dataset B — Linked Primitives Video (spatiotemporal)

**File:** `linked_primitives_video.py`

### Generative process — adds trajectories

Every primitive from Dataset A now also carries:

| Extra attribute | Domain |
|---|---|
| `trajectory` | one of `static`, `linear`, `sinusoidal`, `circular` (per-primitive sampled per `motion_type` policy) |
| `lifetime`   | a [τ_birth, τ_death] subset of [0, 1]; primitive is visible only in this window |

A scene also has a `cross_modal_time_offset` — view B's effective time
is τ_A + offset. At higher difficulties this offset is nonzero, modelling
the realistic case where two sensors aren't perfectly time-synced.

### Trajectory kinds

| Kind | Closed form |
|---|---|
| `static` | `pos(τ) = pos_0` |
| `linear` | `pos(τ) = pos_0 + v · τ` |
| `sinusoidal` | `pos(τ) = pos_0 + amp · sin(2π f τ + phase)` (per axis) |
| `circular` | `pos(τ) = pos_0 + r·(cos(ωτ), sin(ωτ), 0)` |

The operating point's `motion_type` field picks the policy ("static_or_slow_linear", "linear_or_slow_sin", "mixed", or any fixed kind), and `motion_amplitude` scales how aggressive the motion is.

### What you get per render

```python
video = generator.render_video_pair(scene)
video["times"]                    # (T,) τ values sampled in [0, 1]
video["view_A"]["rgb"]            # (T, H, W, 3) uint8 video
video["view_A"]["seg"]            # (T, H, W) int32 per-pixel pid; -1 bg
video["view_A"]["kpts"]           # (T, N, 2) float32 — projected centers
video["view_A"]["vis"]            # (T, N) bool — visibility per (frame, primitive)
video["view_A"]["ids"]            # (N,) int32 — primitive IDs (stable across frames)
video["view_B"]                   # same structure
```

### GIF export

```python
generator.save_gif(video["view_A"]["rgb"], "view_A.gif", fps=scene.fps)
generator.save_gif(video["view_B"]["rgb"], "view_B.gif", fps=scene.fps)
```

### Built-in labels (`compute_label`)

In addition to the three static labels from Dataset A, four new
spatiotemporal labels:

| `kind` | What it computes |
|---|---|
| `has_motion_pattern` | 1 iff at least one primitive has a `circular` or `sinusoidal` trajectory |
| `n_distinct_motion_kinds` | count of distinct trajectory kinds (`static`/`linear`/`sin`/`circular`) |
| `has_fast_motion` | 1 iff any primitive has frequency > 2.0 cycles per video — tests whether the model detects high-frequency dynamics anywhere in the scene |
| `freq_band_count` | count of distinct frequency bands {static, slow, medium, fast} present in the scene — explicit test of multi-Hz simultaneous detection |

### Downstream tasks (use Dataset B for these)

Dataset B subsumes A's tasks (just take frame 0) and adds:

| Task | Input | Output | Ground truth from |
|---|---|---|---|
| **Video-pair classification** | (video_A, video_B) | one class label | `compute_label(scene)` |
| **Spatiotemporal segmentation** | video_A (or B) | per-pixel pid per frame; same pid across frames forms a *tube* | `view_A["seg"]` |
| **Within-view tracking** | video_A | for each (pid, frame_0) a trajectory of (x, y) in subsequent frames | `cross_time_pairs_within_view` or `trajectories_for_view` |
| **Cross-view keypoint retrieval over time** | per-frame features | for each (pid, frame_t) in A: match in B at the same time | `cross_view_pairs_at_time(video, t)` |
| **Cross-view + cross-time matching** | per-frame features | for each (pid, view_A, τ₁) → (view_B, τ₂) | pid intersection across all `ids` |
| **Dense optical flow within view** | (frame_t, frame_{t+1}) of view_A | per-pixel motion vector | finite-difference of per-pid kpts in `view_A["kpts"]` |
| **Motion-conditioned classification** | full video pair | the new labels `has_motion_pattern` / `n_distinct_motion_kinds` | `compute_label(scene, kind=...)` |

All seven tasks have closed-form ground truth from the latent scene.

---

## 6. Operating points — the "arbitrary complexity" axis

Both datasets ship with **five named operating points** with parallel
naming so an SSL method can be evaluated on equivalent settings of both.

### Static (Dataset A)

| Knob | EASY | BASIC | HARD | EXTREME | ADVERSARIAL |
|---|---|---|---|---|---|
| n_linked | 4 | 16 | 64 | 128 | 128 |
| n_shapes | 2 | 4 | 8 | 10 | 10 |
| view disparity | 30° | 60° | 120° | 170° | 170° |
| distractors per modality | 0 | 2 | 16 | 48 | 64 |
| scale range | (0.10, 0.15) | (0.06, 0.16) | (0.03, 0.20) | (0.02, 0.25) | (0.02, 0.25) |
| style on B | rgb | rgb | grayscale | edges | edges |
| noise σ | 0 | 0.02 | 0.05 | 0.08 | 0.10 |
| adversarial confusables | — | — | — | — | yes |

### Spatiotemporal (Dataset B — adds temporal knobs)

Inherits the spatial knobs above, plus:

| Knob | EASY | BASIC | HARD | EXTREME | ADVERSARIAL |
|---|---|---|---|---|---|
| motion type policy | static_or_slow_linear | linear_or_slow_sin | mixed | mixed | mixed |
| motion amplitude | 0.0 | 0.20 | 0.35 | 0.50 | 0.60 |
| frequency range (cycles/video) | (0.25, 0.75) | (0.5, 1.5) | (0.5, 3.0) | (0.5, 5.0) | (0.5, 5.0) |
| n_frames | 8 | 12 | 16 | 24 | 24 |
| fps | 8 | 8 | 8 | 12 | 12 |
| lifetime jitter | 0.0 | 0.0 | 0.15 | 0.30 | 0.30 |
| cross-modal time offset | 0.0 | 0.0 | 0.01 | 0.02 | 0.03 |

### Hz-focused operating points (Dataset B only)

Per-primitive frequency is sampled log-uniformly from `frequency_range`,
so a wide range puts both slow and fast primitives in the **same scene**.

| Operating point | frequency range | What it tests |
|---|---|---|
| `slow_only` | (0.25, 1.0) | Detection of slow dynamics in isolation |
| `fast_only` | (3.0, 8.0) | Detection of fast dynamics in isolation |
| `mixed_hz` | (0.25, 6.0) | **Multi-Hz simultaneously** — each scene has slow + fast primitives co-existing |
| `multiscale_hz` | (0.25, 8.0) on `hard` spatial backbone | Multi-Hz × multi-shape × style gap |

For one-knob ablations, pass a `dict` instead of a name:
```python
custom = dict(OPERATING_POINTS["basic"])
custom["motion_amplitude"] = 0.8       # vary just this
gen = LinkedPrimitivesVideoGenerator(operating_point=custom)
```

---

## 7. Evaluation protocol — three metrics, evaluated per operating point

A multimodal SSL method should be measured on **three orthogonal axes**:

### (i) Direct correspondence-retrieval accuracy

Encode every primitive feature in both modalities. For each primitive
in modality A, retrieve top-k nearest neighbours in modality B by cosine
similarity. Report top-1 / top-5 accuracy at finding the corresponding
primitive.

The **clean isolated test** of correspondence learning. Random features
give 1/N accuracy.

### (ii) Linear-probe classification on φ(z)

Standard SSL probe. Train a linear classifier on SSL features → label.
Because φ depends on the latents not the rendered pixels, this can't be
cheated by global statistics *unless* the operating point is so easy
that one-modality features alone suffice.

### (iii) Cross-modal dense segmentation (and, for B, cross-time tracking)

For each pixel / event in modality A belonging to primitive `p`, predict
the corresponding pixel / event in modality B that belongs to the same
primitive. mIoU / per-primitive top-1 accuracy.

For dataset B, the same metric extends to tracking: for each `(pid,
frame_t)` predict its `(x, y)` at `frame_{t+k}`. RMSE in pixel
coordinates is the standard tracking metric.

### Reading the three together

| (i) Retrieval | (ii) Probe | (iii) Dense correspondence | Interpretation |
|---|---|---|---|
| high | high | high | The model has learned correspondences cleanly. |
| low | high | low | Shortcut: global statistics solve the label. Increase difficulty. |
| high | low | high | Correspondences known, head insufficient. Try a stronger probe. |
| low | low | low | SSL is not getting traction. Try a different recipe. |

---

## 8. Files in this folder

| File | Purpose |
|---|---|
| `linked_primitives.py` | Dataset A generator: `Primitive`, `Scene`, `LinkedPrimitivesGenerator`, `correspondence_pairs`. Pure numpy + PIL. |
| `linked_primitives_video.py` | Dataset B generator: `Trajectory`, `STPrimitive`, `STScene`, `LinkedPrimitivesVideoGenerator`, three correspondence helpers, `save_gif`. Imports and reuses Dataset A's rendering primitives. |
| `multiview_primitives.py` | Dataset C generator: 3-camera variant with narrow FOV. `MultiViewLinkedPrimitivesGenerator`, `cross_view_pairs_triple`, `coverage_summary`. Primitives invisible in every camera are dropped at sampling time so labels are always recoverable. |
| `augmenters.py` | Observation-channel augmenters (noise, sparsity, occlusion, image-FOV crop). Composable via `AugmenterPipeline`. **Noise is no longer baked into the renderers** — it's an independent axis controlled here. |
| `generate_dataset.py` | Standalone CLI + library for generating NPZ datasets. Combines any (dataset, operating_point, augmenter_pipeline) tuple. |
| `correspondence_viz.ipynb` | Walkthrough for Datasets A and B: scenes + correspondences at every operating point, GIF export, all downstream tasks demoed. |
| `sparse_viz.ipynb` | Walkthrough for the sparse / occluded / limited-FOV variants (Dataset C + augmenters). |
| `_build_*.py` | Generator scripts for the notebooks. |

All generators and augmenters are **fully self-contained**: numpy + PIL
only, no torch or other ML-framework dependency.

## 9. Augmenters — observation-channel modifications

Noise, sparsity, occlusion, and image-FOV cropping are **independent of
the operating-point difficulty axis**. They're implemented as augmenters
that post-process rendered outputs:

```python
from augmenters import (
    GaussianNoiseAugmenter, RandomSubsampleAugmenter,
    CenterOcclusionAugmenter, LimitedFOVAugmenter, AugmenterPipeline,
)
pipeline = AugmenterPipeline([
    RandomSubsampleAugmenter(0.4),    # drop 60% of pixels
    CenterOcclusionAugmenter(0.3),    # 30% × 30% center mask
    GaussianNoiseAugmenter(0.05),     # then noise
])
observed = pipeline(rendered_output, rng=42)
```

This lets you compose any of `(operating_point) × (augmenter_pipeline)`
without coupling. Each augmenter has its own docstring detailing
behaviour and parameter ranges; see `augmenters.py`.

## 10. Dataset C — Multi-View Limited FOV

**File:** `multiview_primitives.py`

A third dataset designed for **multi-view scene-understanding** SSL.
N (=3 by default) cameras with **narrow FOV** (focal=4.0, vs 2.0 in A/B)
tile the scene at angles +13° / 0° / −13°. Each camera sees roughly
half of what a single wide-FOV camera would see; their union covers
approximately the same angular extent.

### Generative process — visibility guarantee

The whole point of limited FOV is that each individual view shows only
PART of the scene. For downstream "scene understanding" to be tractable,
**every primitive that contributes to the label must be observable
somewhere**. The generator enforces this:

1. Sample 3× the requested number of candidate primitives.
2. Drop any candidate whose projection center is outside every camera's
   FOV.
3. Truncate the surviving set to the requested `n_linked`.

Primitives in `scene.unobservable` (those dropped at step 2) are
returned for diagnostics but do not contribute to labels or
correspondence ground truth.

A `coverage_summary(renders, n_linked)` helper reports actual rendered
visibility per view; typical numbers across difficulty:

| Operating point | n_linked | post-render coverage (visible in ≥1 view) |
|---|---|---|
| easy | 8 | 1.00 |
| basic | 24 | 0.98 |
| hard | 80 | 0.92 |
| extreme | 100 | 0.80 |
| adversarial | 100 | 0.78 |

(Coverage <1.0 at extreme is due to painter's-algorithm occlusion — a
front primitive can write its `pid` over a back one's pixels. This is
a realistic difficulty knob, not a bug.)

### Downstream tasks (use Dataset C for these)

| Task | Input | Output | Ground truth from |
|---|---|---|---|
| **C-1 Multi-view classification** | features from all N views | one class label | `compute_label(scene)` over filtered linked set |
| **C-2 Per-view segmentation** | one view's image | per-pixel pid in that view | `renders[v]["seg"]` |
| **C-3 Cross-view pairing (all triples)** | per-view features | for each (i, j) pair of views: matched primitive kpts | `cross_view_pairs_triple(renders, i, j)` |
| **C-4 Coverage / view-completion** | features from a subset of views | predict properties of an unseen view (e.g. mask cam 0, predict from cams 1+2) | `coverage_summary` + pid intersection |

Built-in labels for C-1:
- `count_modulo_K`, `has_pair`, `n_distinct_shapes` (same shapes as A)
- `spans_all_views` — 1 iff any primitive is observable in **every** view
  (a different statistic — tests whether the model can find primitives
  that survive the most restrictive view intersection)

## 11. Generating data with `generate_dataset.py`

CLI usage:

```bash
# Static dataset, basic difficulty, with 5% Gaussian noise:
python generate_dataset.py --dataset static \
    --operating-point basic --n 1000 \
    --noise-sigma 0.05 \
    --out ./data/basic_n1000_noise05

# Video, mixed_hz, sparse 40% keep:
python generate_dataset.py --dataset video \
    --operating-point mixed_hz --n 200 \
    --subsample 0.4 \
    --out ./data/mixed_hz_sparse40

# Multi-view, hard difficulty, with center occlusion:
python generate_dataset.py --dataset multiview \
    --operating-point hard --n 500 \
    --occlude 0.3 \
    --out ./data/mv_hard_occluded
```

Library usage:

```python
from generate_dataset import generate
from augmenters import GaussianNoiseAugmenter, RandomSubsampleAugmenter

summary = generate(
    dataset="multiview",
    operating_point="basic",
    n_scenes=100,
    augmenters=[GaussianNoiseAugmenter(0.05), RandomSubsampleAugmenter(0.5)],
    out_dir="./data/mv_basic_sparse",
)
```

Each scene is saved as an NPZ with all view RGBs, segmentation maps,
keypoints, visibility flags, primitive IDs, and the label. A
`manifest.json` summarises the run.

---

## 9. Quick API tour

### Dataset A (static)

```python
from linked_primitives import LinkedPrimitivesGenerator, correspondence_pairs

gen = LinkedPrimitivesGenerator(operating_point="basic", image_size=128)
scene = gen.sample_scene(seed=0)
view_A = gen.render(scene, view="A")
view_B = gen.render(scene, view="B")
pairs = correspondence_pairs(view_A, view_B)          # (M, 2, 2)
label = gen.compute_label(scene, kind="count_modulo_K", K=4)
```

### Dataset B (spatiotemporal)

```python
from linked_primitives_video import (
    LinkedPrimitivesVideoGenerator,
    cross_view_pairs_at_time, cross_time_pairs_within_view, trajectories_for_view,
)

gen = LinkedPrimitivesVideoGenerator(operating_point="hard", image_size=128)
scene = gen.sample_scene(seed=0)
video = gen.render_video_pair(scene)
# Save GIFs
gen.save_gif(video["view_A"]["rgb"], "view_A.gif", fps=scene.fps)
gen.save_gif(video["view_B"]["rgb"], "view_B.gif", fps=scene.fps)
# Correspondences
pairs_xv  = cross_view_pairs_at_time(video, t_idx=0)               # (M, 2, 2)
pairs_xt  = cross_time_pairs_within_view(video, "A", 0, 5)          # (M, 2, 2)
trajs     = trajectories_for_view(video, "A")                       # {pid: (T, 2)}
label = gen.compute_label(scene, kind="has_motion_pattern")
```

---

## 10. Extending the benchmark

- **New label functions**: add a branch to `compute_label`. The contract
  is that the label is a function of the latent scene only.
- **New trajectory kinds**: add a branch to `_sample_trajectory` and a
  matching branch to `Trajectory.pos_at`. Any closed-form
  `f(τ): [0,1] → ℝ³` works.
- **New styles**: add a style tag to the render switch.
- **More than 2 modalities**: both files have clean A/B separation;
  adding a third view is mostly mechanical (sample a new camera matrix
  + a new render call).
- **New shapes**: append to `SHAPES` and add a `_draw_shape` branch.
- **New operating points**: add a dict to `OPERATING_POINTS`.

The evaluation protocol is dataset-agnostic — it operates on whatever
`(modality_features, correspondences, labels)` tuple the dataset
returns.

---

## 11. The bigger plan this benchmark serves

Test the central hypothesis of the sparse-input SSL line of work:

> Multi-scale, position-aware aggregators (RoPE / HRR / spectral) should
> outperform PointNet-style max-pool aggregators when the downstream
> task requires fine-grained spatial *or spatiotemporal* correspondence —
> and this advantage should grow with task difficulty.

On CIFAR-10 the three aggregators converged because the dataset didn't
punish lossy aggregation. On this benchmark the difficulty knobs
explicitly amplify the information loss: adversarial near-duplicate
primitives, fast motion that smears appearance across frames,
cross-modal time offsets that require alignment.

Expected curves: flat at EASY (all methods tie), spreading at HARD,
wide gap at EXTREME/ADVERSARIAL. Dataset B specifically tests whether
the aggregator advantage extends to motion — RoPE/HRR's spectral
decomposition should remain stable under controlled motion, while
PointNet's max-pool may struggle to track shifting features.
