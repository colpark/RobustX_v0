"""Build correspondence_bench/sparse_viz.ipynb — visualizations of sparse /
occluded / limited-FOV scenarios. Three flavours:
  (1) Random pixel subsample (sparse-modality simulation)
  (2) Center occlusion (foreground-object obstruction)
  (3) Multi-view limited FOV (3 cameras, each narrow, jointly covering)
"""
import json, os
NB = "/Users/davidpark/Documents/Claude/NFJEPA/OmniBird/correspondence_bench/sparse_viz.ipynb"

cells = []
def md(s):   cells.append({"cell_type":"markdown","metadata":{},"source":s.splitlines(keepends=True)})
def code(s): cells.append({"cell_type":"code","metadata":{},"execution_count":None,"outputs":[],"source":s.splitlines(keepends=True)})


# =============================================================================
md(r"""# Sparse / Occluded / Limited-FOV Variants

A companion notebook to `correspondence_viz.ipynb`. The base benchmark
gives you two full-view observations of every scene. This notebook
demonstrates **three sparse-observation variants**, each implemented as
a separate augmenter or a separate generator.

| Variant | Implementation | What it simulates |
|---|---|---|
| **1. Random subsample** | `RandomSubsampleAugmenter` | Sparse-sampling modalities (point clouds, event cameras, sparse pixel pools) |
| **2. Center occlusion** | `CenterOcclusionAugmenter` | A foreground object blocking part of the camera, or intentional center masking for inpainting-style SSL |
| **3. Multi-view limited FOV** | `MultiViewLinkedPrimitivesGenerator` (3 cameras, narrow FOV each) | Multi-sensor scene-understanding where each sensor sees only part of the scene |

## Design principle: noise/sparsity is SEPARATE from difficulty

The base operating points (`easy / basic / hard / ...`) control the
**latent scene**: how many primitives, how multi-scale, how irregular,
how view-disparate. They do **not** control the observation channel.

Observation-channel corruption — noise, sparse subsampling, occlusion,
field-of-view restriction — is layered on top via the augmenter API in
`augmenters.py`. This means you can:

- Run any operating point with any noise level (orthogonal axes).
- Sweep one without touching the other.
- Compose multiple augmenters (e.g. subsample + occlude + noise).

## Visibility guarantee for limited FOV

For multi-view limited FOV (variant 3), each individual camera sees
only ~50% of the scene. The downstream task ("scene understanding") is
only solvable if **every primitive is visible in at least one view**.
We enforce this at scene-sampling time: primitives invisible in all
three views are dropped from the latent scene. They never contribute
to the label or the correspondence ground truth.

This means the downstream task is **always recoverable in principle**
from joint reasoning across the three views.

---

**Outline**

- §1 Setup
- §2 Variant 1 — random subsample (Dataset A)
- §3 Variant 1 — same idea on Dataset B videos
- §4 Variant 2 — center occlusion (Dataset A)
- §5 Variant 2 — same on Dataset B (consistent vs frame-independent occlusion)
- §6 Variant 3 — 3-view limited FOV: scene + per-view renders + coverage
- §7 Variant 3 — scene-understanding downstream tasks (C-1..C-4)
- §8 Composed pipeline — subsample + occlude + noise
""")


# =============================================================================
md("## §1. Setup")
code(r"""import os, sys
sys.path.insert(0, os.path.abspath('.'))
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import ConnectionPatch, Rectangle
from IPython.display import Image as IPImage, display

from linked_primitives import LinkedPrimitivesGenerator, correspondence_pairs
from linked_primitives_video import LinkedPrimitivesVideoGenerator
from multiview_primitives import (
    MultiViewLinkedPrimitivesGenerator,
    cross_view_pairs_triple, coverage_summary,
    OPERATING_POINTS as MV_POINTS,
)
from augmenters import (
    IdentityAugmenter, GaussianNoiseAugmenter, SaltPepperNoiseAugmenter,
    RandomSubsampleAugmenter, CenterOcclusionAugmenter,
    LimitedFOVAugmenter, AugmenterPipeline,
)
np.random.seed(0)
print("multiview operating points:", list(MV_POINTS.keys()))
""")


# =============================================================================
md(r"""## §2. Variant 1 — random subsample on Dataset A

`RandomSubsampleAugmenter(keep_fraction)` drops a random subset of
pixels per image. Same operating point across columns; only the
`keep_fraction` changes.
""")
code(r"""gen = LinkedPrimitivesGenerator(operating_point="basic", image_size=128, base_seed=0)
scene = gen.sample_scene(seed=7)
clean_A = gen.render(scene, view="A")
fractions = [1.0, 0.6, 0.3, 0.1]
fig, axes = plt.subplots(1, len(fractions), figsize=(4 * len(fractions), 4))
for ax, frac in zip(axes, fractions):
    aug = RandomSubsampleAugmenter(keep_fraction=frac)
    out = aug(clean_A, rng=7)
    ax.imshow(out["rgb"]); ax.axis("off")
    ax.set_title(f"keep = {frac:.2f}  (drop {100*(1-frac):.0f}%)")
plt.suptitle("RandomSubsampleAugmenter — Dataset A view", y=1.02)
plt.tight_layout(); plt.show()
""")


# =============================================================================
md(r"""## §3. Variant 1 — random subsample on Dataset B videos

For video, you can apply the augmenter **independently per frame**
(temporally-independent corruption — sparseness fluctuates over time)
or with a **fixed RNG** (temporally-consistent corruption — the same
pixels are dropped at every frame). We show both.
""")
code(r"""gen = LinkedPrimitivesVideoGenerator(operating_point="basic", image_size=128, base_seed=3)
scene = gen.sample_scene(seed=3)
video = gen.render_video_pair(scene)
T = video["view_A"]["rgb"].shape[0]
aug = RandomSubsampleAugmenter(keep_fraction=0.3)

# Temporally-independent (per-frame rng)
fig, axes = plt.subplots(2, 4, figsize=(14, 7))
indices = np.linspace(0, T - 1, 4).astype(int)
for col, t in enumerate(indices):
    one = {k: video["view_A"][k][t] if k != "ids" else video["view_A"][k]
           for k in ("rgb", "seg", "kpts", "vis", "ids")}
    indep   = aug(one, rng=t * 17)
    consist = aug(one, rng=42)    # same rng → same drop mask every frame
    axes[0, col].imshow(indep["rgb"]); axes[0, col].axis("off"); axes[0, col].set_title(f"τ={t/T:.2f}  (per-frame rng)")
    axes[1, col].imshow(consist["rgb"]); axes[1, col].axis("off"); axes[1, col].set_title(f"τ={t/T:.2f}  (fixed rng — same pixels)")
plt.tight_layout(); plt.show()
""")


# =============================================================================
md(r"""## §4. Variant 2 — center occlusion on Dataset A

`CenterOcclusionAugmenter(occlusion_fraction)` masks a centered square
of side `occlusion_fraction × min(H, W)`. The seg map is also masked
(seg = -1 inside the occlusion) so the per-pixel correspondence ground
truth remains accurate.
""")
code(r"""gen = LinkedPrimitivesGenerator(operating_point="basic", image_size=128, base_seed=0)
scene = gen.sample_scene(seed=12)
clean_A = gen.render(scene, view="A"); clean_B = gen.render(scene, view="B")
fracs = [0.0, 0.2, 0.4, 0.6]
fig, axes = plt.subplots(2, len(fracs), figsize=(4 * len(fracs), 8))
for col, f in enumerate(fracs):
    aug = CenterOcclusionAugmenter(occlusion_fraction=f)
    occ_A = aug(clean_A); occ_B = aug(clean_B)
    axes[0, col].imshow(occ_A["rgb"]); axes[0, col].axis("off")
    axes[0, col].set_title(f"view A — occlude {int(100*f)}%")
    axes[1, col].imshow(occ_B["rgb"]); axes[1, col].axis("off")
    axes[1, col].set_title(f"view B — occlude {int(100*f)}%")
plt.suptitle("CenterOcclusionAugmenter — Dataset A", y=1.02)
plt.tight_layout(); plt.show()
""")


# =============================================================================
md(r"""## §5. Variant 2 on Dataset B — temporally-fixed occlusion

For occlusion the natural choice is usually **temporally-fixed** — the
occluding object doesn't suddenly move every frame. So we apply the
same augmenter (no randomness in CenterOcclusionAugmenter anyway) and
get a temporally-stable occluded patch.
""")
code(r"""gen = LinkedPrimitivesVideoGenerator(operating_point="basic", image_size=128, base_seed=4)
scene = gen.sample_scene(seed=4)
video = gen.render_video_pair(scene)
T = video["view_A"]["rgb"].shape[0]
aug = CenterOcclusionAugmenter(occlusion_fraction=0.35)

indices = np.linspace(0, T - 1, 4).astype(int)
fig, axes = plt.subplots(1, len(indices), figsize=(4 * len(indices), 4))
for ax, t in zip(axes, indices):
    one = {k: video["view_A"][k][t] if k != "ids" else video["view_A"][k]
           for k in ("rgb", "seg", "kpts", "vis", "ids")}
    occ = aug(one)
    ax.imshow(occ["rgb"]); ax.axis("off"); ax.set_title(f"τ={t/T:.2f}")
plt.suptitle("Center occlusion is temporally fixed — primitives visible at edges only", y=1.02)
plt.tight_layout(); plt.show()
""")


# =============================================================================
md(r"""## §6. Variant 3 — 3-view limited FOV (`multiview_primitives.py`)

The third camera-config has **N = 3 cameras** at +13° / 0° / -13°, each
with a NARROW field of view (focal = 4.0 — double the standard).
Each camera sees ~half the scene; their union covers ~all of it.

The generator **filters primitives invisible in every view** at sampling
time, so every primitive that contributes to the label is observable
somewhere. The `coverage_summary` helper quantifies this.

Below: three camera renders side by side per operating point, plus a
coverage stat printed underneath.
""")
code(r"""for name in ["basic", "hard", "extreme"]:
    gen = MultiViewLinkedPrimitivesGenerator(operating_point=name, image_size=128, base_seed=42)
    scene = gen.sample_scene(seed=42)
    renders = gen.render(scene)
    fig, axes = plt.subplots(1, len(renders), figsize=(4 * len(renders), 4))
    for col, (ang, r) in enumerate(zip(scene.knobs["view_angles_deg"], renders)):
        axes[col].imshow(r["rgb"]); axes[col].axis("off")
        axes[col].set_title(f"camera {col}  angle = {ang:+.0f}°\n{int(r['vis'].sum())} visible primitives", fontsize=9)
    plt.suptitle(f"{name.upper()} — 3 cameras with narrow FOV  "
                  f"(n_linked = {len(scene.linked)}, unobservable dropped = {len(scene.unobservable)})", y=1.05)
    plt.tight_layout(); plt.show()
    cov = coverage_summary(renders, n_linked=len(scene.linked))
    print(f"  coverage stats (linked primitives):")
    print(f"    visible in each view : {cov['visible_in_each']}")
    print(f"    visible in at least 1: {cov['visible_in_any']} / {len(scene.linked)}  (= {100*cov['coverage_frac']:.1f}%)")
    print(f"    visible in all 3     : {cov['visible_in_all']}")
""")


# =============================================================================
md(r"""## §7. Variant 3 — scene-understanding downstream tasks

The 3-view variant has four canonical tasks (C-1 through C-4) all of
which require joint reasoning across views.

### C-1: Multi-view classification (the headline task)

**Input:** features extracted from all N views.
**Output:** a single scene label.
**Ground truth:** `compute_label(scene)` over the FILTERED linked set,
which is guaranteed observable somewhere.

We show three scenes from `basic` with all three views and their labels.
""")
code(r"""gen = MultiViewLinkedPrimitivesGenerator(operating_point="basic", image_size=96, base_seed=7)
for seed in [1, 5, 11]:
    scene = gen.sample_scene(seed=seed)
    renders = gen.render(scene)
    fig, axes = plt.subplots(1, len(renders), figsize=(3 * len(renders), 3.2))
    for col, r in enumerate(renders):
        axes[col].imshow(r["rgb"]); axes[col].axis("off")
        axes[col].set_title(f"cam {col}", fontsize=9)
    y_count = gen.compute_label(scene, kind="count_modulo_K", K=4)
    y_pair  = gen.compute_label(scene, kind="has_pair")
    y_shape = gen.compute_label(scene, kind="n_distinct_shapes", K=5)
    y_span  = gen.compute_label(scene, kind="spans_all_views")
    plt.suptitle(f"seed={seed}   "
                  f"count_mod4={y_count}  has_pair={y_pair}  "
                  f"n_distinct_shapes={y_shape}  spans_all_views={y_span}",
                  y=1.05, fontsize=10)
    plt.tight_layout(); plt.show()
""")


md(r"""### C-2: Per-view segmentation

Each camera independently produces a `seg` map. Same colormap across
all three views = the same `pid` in different views gets the same
color, making the cross-view correspondence visually obvious.
""")
code(r"""gen = MultiViewLinkedPrimitivesGenerator(operating_point="basic", image_size=128, base_seed=20)
scene = gen.sample_scene(seed=20)
renders = gen.render(scene)
fig, axes = plt.subplots(2, len(renders), figsize=(4 * len(renders), 8))
for col, r in enumerate(renders):
    axes[0, col].imshow(r["rgb"]); axes[0, col].axis("off")
    axes[0, col].set_title(f"cam {col} — RGB", fontsize=9)
    axes[1, col].imshow(r["seg"], cmap="tab20"); axes[1, col].axis("off")
    axes[1, col].set_title(f"cam {col} — seg (same colormap)", fontsize=9)
plt.tight_layout(); plt.show()
""")


md(r"""### C-3: Cross-view pairing for all three (i, j) pairs

For every pair of cameras (0,1), (0,2), (1,2), draw lines connecting
matched primitive centres. This is the cross-view correspondence ground
truth that the model needs to recover.
""")
code(r"""gen = MultiViewLinkedPrimitivesGenerator(operating_point="basic", image_size=128, base_seed=7)
scene = gen.sample_scene(seed=7)
renders = gen.render(scene)
pairs_idx = [(0, 1), (0, 2), (1, 2)]
fig, axes = plt.subplots(len(pairs_idx), 2, figsize=(10, 5 * len(pairs_idx)))
for row, (i, j) in enumerate(pairs_idx):
    pairs = cross_view_pairs_triple(renders, i, j)
    axI = axes[row, 0]; axJ = axes[row, 1]
    axI.imshow(renders[i]["rgb"]); axI.set_title(f"cam {i}"); axI.axis("off")
    axJ.imshow(renders[j]["rgb"]); axJ.set_title(f"cam {j}  ({pairs.shape[0]} matches)"); axJ.axis("off")
    for k in range(pairs.shape[0]):
        (xA, yA), (xB, yB) = pairs[k]
        con = ConnectionPatch(xyA=(xA, yA), coordsA=axI.transData,
                               xyB=(xB, yB), coordsB=axJ.transData,
                               color=plt.cm.tab20(k % 20), lw=0.8, alpha=0.7)
        fig.add_artist(con)
plt.suptitle("C-3 — Cross-view pairing across all camera pairs", y=1.01)
plt.tight_layout(); plt.show()
""")


md(r"""### C-4: View-coverage diagnostic

Visualize, for each primitive, **how many views it appears in**. A
primitive seen by all 3 cameras is in the "easy" middle region; one
seen by only 1 camera is at the edge of the coverage and is harder to
characterize.
""")
code(r"""gen = MultiViewLinkedPrimitivesGenerator(operating_point="hard", image_size=128, base_seed=15)
scene = gen.sample_scene(seed=15)
renders = gen.render(scene)
n_linked = len(scene.linked)
# Count how many views see each linked pid
seen = np.zeros((n_linked,), dtype=int)
for r in renders:
    for i, pid in enumerate(r["ids"]):
        if pid < n_linked and r["vis"][i]:
            seen[pid] += 1
print(f"linked = {n_linked}, "
      f"#visible-in-1 = {int((seen == 1).sum())}, "
      f"#visible-in-2 = {int((seen == 2).sum())}, "
      f"#visible-in-3 = {int((seen == 3).sum())}")

# Overlay the count on the center camera's image
fig, ax = plt.subplots(figsize=(8, 8))
ax.imshow(renders[1]["rgb"])
COLORS = {1: 'red', 2: 'orange', 3: 'green'}
for pid in range(n_linked):
    # If this pid is in the center camera's render, get its kpt there
    if pid in renders[1]["ids"]:
        idx = int(np.where(renders[1]["ids"] == pid)[0][0])
        if renders[1]["vis"][idx]:
            x, y = renders[1]["kpts"][idx]
            ax.plot(x, y, 'o', mfc=COLORS.get(int(seen[pid]), 'gray'),
                     mec='black', ms=10, mew=0.6)
ax.set_title("Coverage by view count  (red = 1 view,  orange = 2 views,  green = 3 views)")
ax.axis("off")
plt.tight_layout(); plt.show()
""")


# =============================================================================
md(r"""## §8. Composed pipeline — subsample + occlude + noise

`AugmenterPipeline` chains augmenters. Order matters: here we first
drop pixels, then occlude the center, then add Gaussian noise on top.
The result mimics a realistic "noisy sensor with occlusion" scenario.
""")
code(r"""gen = LinkedPrimitivesGenerator(operating_point="basic", image_size=128, base_seed=33)
scene = gen.sample_scene(seed=33)
clean = gen.render(scene, view="A")

pipelines = [
    ("clean", AugmenterPipeline([IdentityAugmenter()])),
    ("noise only", AugmenterPipeline([GaussianNoiseAugmenter(0.08)])),
    ("subsample only", AugmenterPipeline([RandomSubsampleAugmenter(0.4)])),
    ("occlude only", AugmenterPipeline([CenterOcclusionAugmenter(0.3)])),
    ("subsample+occlude+noise",
     AugmenterPipeline([
        RandomSubsampleAugmenter(0.4),
        CenterOcclusionAugmenter(0.3),
        GaussianNoiseAugmenter(0.05),
    ])),
]
fig, axes = plt.subplots(1, len(pipelines), figsize=(4 * len(pipelines), 4))
for ax, (name, pipe) in zip(axes, pipelines):
    out = pipe(clean, rng=33)
    ax.imshow(out["rgb"]); ax.axis("off"); ax.set_title(name, fontsize=9)
plt.suptitle("Pipeline composition — observation channel is separate from latent difficulty", y=1.02)
plt.tight_layout(); plt.show()
""")


# =============================================================================
md(r"""## §9. Where to go from here

- Use `generate_dataset.py` to write large NPZ datasets at any
  combination of operating point + augmenter pipeline.
- The three sparse variants compose freely with the temporal version
  (Dataset B) — see §3 and §5 above.
- For the SSL benchmark sweep, fix the operating point and sweep the
  augmenter knobs (or vice versa) to test the orthogonality of latent
  difficulty and observation noise.
""")


nb = {"cells": cells, "metadata": {
    "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
    "language_info": {"name": "python", "version": "3.11"},
}, "nbformat": 4, "nbformat_minor": 5}
os.makedirs(os.path.dirname(NB), exist_ok=True)
with open(NB, "w") as f:
    json.dump(nb, f, indent=1)

import ast
errs = 0
for i, c in enumerate(cells):
    if c["cell_type"] == "code":
        try: ast.parse("".join(c["source"]))
        except SyntaxError as e:
            errs += 1; print(f"  cell {i}: {e}")
print(f"Wrote {NB}")
print(f"  cells: {len(cells)}    syntax errors: {errs}")
