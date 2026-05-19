# Correspondence Bench — A Difficulty-Controlled Multimodal Benchmark

A synthetic benchmark designed to **measure whether a self-supervised
learning recipe can discover fine-grained cross-modal correspondences**
— and whether better correspondence learning translates into better
downstream classification / segmentation.

The benchmark is composed of two complementary datasets:

| | Dataset | Modalities | Best for |
|---|---|---|---|
| **A** | **Linked Primitives** | Two rendered 2D images (different camera views) of the same 3D primitive scene | Visual debugging; classification + segmentation; the human-interpretable "main figure" of a paper |
| **C** | **Synthetic Event Streams** | Two sparse point sets in space-time, with modality-specific attributes | Sparse-input SSL recipes (events, point clouds); matches the FPS+KNN pipeline directly |

Both datasets share the same **design philosophy** and the same **named
difficulty operating points** (`easy / basic / hard / extreme /
adversarial`). The intent is that the **same SSL method** can be
evaluated on both datasets at the same operating point and the results
are directly comparable.

---

## 1. Why this benchmark exists

Existing multimodal SSL benchmarks are not designed to **isolate
correspondence learning**. They tend to fail one of the following:

1. **Global statistics suffice.** Many image-pair datasets can be solved
   by learning a global representation (e.g. average colour, total
   brightness). The model never has to do fine-grained correspondence,
   so we can't tell whether it has the ability.
2. **No ground-truth correspondences.** Real datasets don't ship with
   pixel-level cross-modal alignment, so we evaluate via downstream
   proxy tasks and can't probe the correspondence quality directly.
3. **Difficulty is not parameterized.** Real datasets are "hard" or
   "easy" as a whole; you can't dial them up to test where a method
   breaks.
4. **Multi-scale structure is not controlled.** Real data has whatever
   scale distribution it has. We can't test scale-equivariance with
   experiments where scale is the only thing that varies.

This benchmark fixes all four:

1. **Label depends on the latent scene, not the observations.** The
   class label is computed from the *primitive-set latents*, not from
   the rendered image. So you can't solve it by global statistics —
   you have to recover (a sufficient statistic of) the latents, which
   requires cross-modal correspondence.
2. **Ground-truth correspondences are returned with every sample.**
   Each primitive carries a stable integer ID present in both modality
   renders, and we ship a helper that returns the `(M, 2)` matched
   pair list directly.
3. **A vector of independent difficulty knobs.** Five named operating
   points span EASY → ADVERSARIAL, plus you can pin everything except
   one knob to do single-axis ablations.
4. **Scale variance is an explicit knob.** `scale_range = (0.02, 0.25)`
   means primitives' physical sizes span > 12× in the same scene,
   forcing the model to be scale-aware.

---

## 2. The generative-model contract

Both datasets follow the same schema:

```
Latent scene  z = {entity_1, entity_2, ..., entity_N}
                                                      ↓
Modality A observation  X_A = g_A(z)              Modality B observation  X_B = g_B(z)
                                                      ↓
                       Label  y = φ(z)
                       Correspondences  C = {(i_A, i_B) : both modalities see entity_i}
```

Crucially:
- `g_A` and `g_B` are **deterministic** given `z` (and a random
  per-modality nuisance: jitter, noise, style).
- `y = φ(z)` is computable **only from the latents**. A model that
  predicts `y` must recover a sufficient statistic of `z`, which
  forces it to solve the cross-modal alignment problem.
- The correspondence ground truth `C` is **returned with every
  sample**, so you can evaluate it directly (not just via downstream
  proxies).

---

## 3. Dataset A — Linked Primitives

### Generative process

A scene contains **N "linked" primitives** placed in 3D space:

| Attribute | Range / domain |
|---|---|
| `shape_id` | one of `circle, square, triangle, plus, star, diamond, hexagon, cross, pentagon, octagon` |
| `color_idx` | one of K configurable RGB palette entries |
| `pos_3d` | uniform in [-0.9, 0.9]^3 |
| `size` | log-uniform in `scale_range` (e.g. (0.02, 0.25)) |

Each primitive carries a globally unique `pid` (integer 0..N-1 for
linked primitives, then N..N+M_A-1 for view-A distractors, etc.). The
`pid` is the correspondence label.

The scene is rendered from two camera viewpoints (rotations of the
3D scene around the y-axis by ±`view_disparity_deg / 2`, with a small
random x-tilt shared between views). Rendering uses a simple pinhole
projection plus painter's-algorithm depth sort.

### What each render returns

```python
out = generator.render(scene, view="A")
out["rgb"]   # (H, W, 3) uint8 image
out["seg"]   # (H, W) int32 — per-pixel primitive ID, -1 background
out["kpts"]  # (N_total, 2) float32 — 2D projected centers
out["vis"]   # (N_total,) bool — primitive visible in this view?
out["ids"]   # (N_total,) int32 — primitive IDs in canonical order
```

Correspondence ground truth:
```python
pairs = correspondence_pairs(out_A, out_B)
# pairs: (M, 2, 2) — M matched primitives, each with (x, y) in A and in B
```

### Styles available for view B

| Style | Description |
|---|---|
| `rgb` | identical rendering style to view A |
| `grayscale_B` | view B is rendered then desaturated to greyscale |
| `edges_B` | view B is rendered then run through a Sobel edge detector and inverted, leaving line drawings on white |

Style asymmetry forces the model to learn modality-invariant
representations, not pixel-matching.

### Built-in labels (`compute_label`)

| `kind` | What it computes |
|---|---|
| `count_modulo_K` | number of linked primitives mod K |
| `has_pair` | 1 iff there's a pair of linked primitives with the same shape but different colours |
| `n_distinct_pairs` | count of distinct (shape, colour) tuples in the linked set |

None of these can be solved by global image statistics alone — they
require recovering the per-primitive identity in both modalities.

---

## 4. Dataset C — Synthetic Event Streams

### Generative process

A scene contains **N linked events** in latent space-time:

| Latent | Domain |
|---|---|
| `feat_class_i` | uniform over F feature classes |
| `f_per_event_i` | `prototype[feat_class_i] + small noise` (ℝ⁴) |
| `p_latent_i` | uniform on [-0.9, 0.9]² |
| `τ_i` | uniform on [0, 1] |

A scene-level **cross-modal transform** `T_B` is sampled (one of
identity, rotation, affine, nonlinear-radial-warp). Modality B's
spatial positions are obtained by applying `T_B` to the latents. The
model must learn to *invert* this transform implicitly when learning
correspondences.

Per-modality renders apply:
- **Position jitter** δ_i^A, δ_i^B
- **Time jitter** ε_i^A, ε_i^B
- **Modality-specific attribute encoders** g_A, g_B — separate linear
  projections (one of them with a nonlinear tanh squashing) from
  `f_per_event` into possibly different feature dimensions D_A, D_B.
  An `attr_corr ∈ [0, 1]` knob controls how aligned the two encoders
  are: at `corr=1` they're the same encoder; at `corr=0` they're
  independent random.

Around the linked events, M_A and M_B **distractor events** are sampled
uniformly in space-time with random attributes. Their `src` field is
set to `-1`. After concatenating linked + distractors, each modality is
**independently random-permuted** so that event order carries no
correspondence information.

### What each scene returns

```python
scene = generator.sample_scene(seed=0)
scene.A_pos    # (K_A, 2)  spatial positions
scene.A_time   # (K_A,)    timestamps
scene.A_attrs  # (K_A, D_A) attribute features
scene.A_src    # (K_A,)    source ID; ≥0 means linked, -1 means distractor
scene.B_pos, scene.B_time, scene.B_attrs, scene.B_src  # parallel
scene.transform   # dict describing T_B (so you can probe whether SSL recovers it)
scene.label
```

Correspondence ground truth:
```python
pairs = correspondence_indices(scene)
# pairs: (M, 2) integer indices into A and B (`A_pos[pairs[m, 0]]` ↔ `B_pos[pairs[m, 1]]`)
```

### Built-in labels (`label_kind`)

| `kind` | What it computes |
|---|---|
| `count_modulo_K` | N mod K |
| `majority_feature` | the most common feature class among linked events |
| `n_distinct_features` | how many distinct feature classes appear |
| `transform_class` | coarse type of the cross-modal transform (id/rot/aff/nl) |

---

## 5. Operating points — the "arbitrary complexity" axis

Both datasets ship with **five named operating points**:

| Knob | EASY | BASIC | HARD | EXTREME | ADVERSARIAL |
|---|---|---|---|---|---|
| n_linked | 4 / 8 | 16 / 32 | 64 / 128 | 128 / 256 | 128 / 256 |
| n_shapes (A) / n_features (C) | 2 | 4 | 8 | 10 | 10 |
| view disparity (A) / transform (C) | 30° / identity | 60° / rotation | 120° / affine | 170° / nonlinear | 170° / nonlinear |
| distractors per modality | 0 | 2–4 | 16–32 | 48–128 | 64–192 |
| scale range (A) / attr_corr (C) | small | small | medium | full | full |
| style gap (A) | rgb=rgb | rgb=rgb | rgb vs grayscale | rgb vs edges | rgb vs edges |
| noise σ (A) / position jitter (C) | 0 | 0.02 | 0.05 | 0.08 | 0.10 |
| adversarial confusables (A) | — | — | — | — | yes |

For one-knob ablations, build a custom dict instead of passing a name:

```python
custom = dict(OPERATING_POINTS["basic"])
custom["n_linked"] = 128       # vary just this knob
gen = LinkedPrimitivesGenerator(operating_point=custom)
```

---

## 6. Evaluation protocol — the three metrics

A multimodal SSL method should be measured on **three orthogonal axes**,
not one:

### (i) Direct correspondence-retrieval accuracy

Encode every primitive (linked + distractor) in both modalities. For
each primitive's feature in modality A, retrieve the top-k nearest
neighbours in modality B (by cosine similarity or Euclidean distance).
Report top-1 / top-5 accuracy at finding the correct corresponding
primitive.

This is the **clean isolated test** of correspondence learning.
Random features = 1/N accuracy. Perfect features = 100%. The middle is
the interesting regime.

### (ii) Linear-probe classification on φ(z)

Standard SSL probe. Train a linear classifier on the SSL features to
predict the latent-derived label. Because `φ` depends on the latents
not the image, this can't be cheated with global statistics — *unless*
the operating point is so easy that one-modality features alone suffice
(which is why operating-point sweeps are essential).

### (iii) Cross-modal dense segmentation

For each pixel (linked-primitives) or event (event streams) in modality
A that belongs to primitive `p`, predict the corresponding pixel/event
in modality B that belongs to the same primitive. Treat this as a
classification problem over the N primitives in the scene. Report
mIoU / per-primitive top-1 accuracy.

This is the **fine-grained correspondence quality** metric. Most direct
test of "does the SSL representation know which-thing-is-which".

### Reading the three together

| (i) Retrieval | (ii) Probe | (iii) Cross-modal seg | Interpretation |
|---|---|---|---|
| high | high | high | The model has learned correspondences cleanly. |
| low | high | low | Shortcut — global statistics solve the label. **Increase difficulty.** |
| high | low | high | Correspondences known but linear probe insufficient. Try a stronger head. |
| low | low | low | SSL is not getting traction. Try a different operating point or recipe. |

---

## 7. Files in this folder

| File | Purpose |
|---|---|
| `linked_primitives.py` | Dataset A generator: `Primitive`, `Scene`, `LinkedPrimitivesGenerator`, `correspondence_pairs` helper. Pure-Python (numpy + PIL). |
| `synth_event_streams.py` | Dataset C generator: `EventStreamScene`, `SyntheticEventStreamsGenerator`, `correspondence_indices` helper. Pure numpy. |
| `correspondence_viz.ipynb` | Walkthrough notebook: instantiates both generators at all five operating points, plots renders + correspondences + label distributions. |
| `_build_correspondence_viz.py` | Generator script for the notebook. |
| `README.md` | This file. |

The generators are **fully self-contained**. No torch dependency. They
can be used standalone to generate datasets that any framework
(PyTorch, JAX, anything) can consume.

---

## 8. Quick API tour

```python
from linked_primitives import LinkedPrimitivesGenerator, correspondence_pairs

gen = LinkedPrimitivesGenerator(operating_point="basic", image_size=128)
scene = gen.sample_scene(seed=0)
view_A = gen.render(scene, view="A")
view_B = gen.render(scene, view="B")
pairs = correspondence_pairs(view_A, view_B)           # (M, 2, 2): M matched pairs
label = gen.compute_label(scene, kind="count_modulo_K", K=4)
```

```python
from synth_event_streams import SyntheticEventStreamsGenerator, correspondence_indices

gen = SyntheticEventStreamsGenerator(operating_point="hard")
scene = gen.sample_scene(seed=0)
# scene.A_pos: (K_A, 2), scene.A_attrs: (K_A, D_A), ...
pairs = correspondence_indices(scene)                   # (M, 2) idx pairs
label = scene.label                                      # already computed
```

---

## 9. Extending the benchmark

The intent is for this to be **extensible**. The clean extension points:

- **New label functions.** Add a branch to `compute_label`. The contract
  is that the label must be a function of the latent scene only.
- **New cross-modal transforms.** Add a kind to `_build_transform` /
  `_apply_transform` in `synth_event_streams.py`. Any function
  `T: ℝ² → ℝ²` works.
- **New styles.** Add a style tag to the render switch in
  `linked_primitives.py`. Currently `rgb / grayscale / edges`.
- **More modalities.** Both files have clean A/B separation; adding a
  third modality is mostly mechanical (sample a new render / encoder).
- **New shapes.** Append to `SHAPES` and add a `_draw_shape` branch.
- **New operating points.** Add a dict to `OPERATING_POINTS`. Naming
  convention: one of `easy/basic/hard/extreme/adversarial` or a
  descriptive custom name.

The downstream SSL evaluation (the three metrics in §6) is **dataset-
agnostic** — it operates on whatever (modality_A_features,
modality_B_features, correspondences, labels) tuple your dataset
returns. Adding new datasets does not require changing the evaluation.

---

## 10. The bigger plan this benchmark serves

This dataset is intended to test the central hypothesis of the
sparse-input SSL line of work:

> Multi-scale, position-aware aggregators (RoPE / HRR / spectral) should
> outperform PointNet-style max-pool aggregators when the downstream
> task requires fine-grained spatial correspondence — and this
> advantage should grow with task difficulty.

On CIFAR-10 with a 6-layer encoder, the three aggregators converge to
the same accuracy because the dataset doesn't punish lossy aggregation.
On this benchmark, **the difficulty knobs explicitly amplify the
information loss that an inadequate aggregator suffers**. Two adversarial
near-duplicate primitives can be told apart only if the aggregator
preserves enough spatial-spectral detail to distinguish them at the
patch level.

If the hypothesis holds, the difficulty × accuracy curve should:
- be flat (all methods tie) at EASY,
- start spreading at HARD,
- show a wide gap at EXTREME and ADVERSARIAL,
- with RoPE/HRR pulling ahead of PointNet specifically when the per-
  patch event entropy is the bottleneck.

That curve is the headline figure.
