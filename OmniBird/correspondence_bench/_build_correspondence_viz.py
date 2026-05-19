"""Build correspondence_bench/correspondence_viz.ipynb — visualization +
walkthrough of the two correspondence datasets at all five operating
points. Dataset A = static linked primitives. Dataset B = spatiotemporal
(video) linked primitives with GIF export and three correspondence
flavours."""
import json, os
NB = "/Users/davidpark/Documents/Claude/NFJEPA/OmniBird/correspondence_bench/correspondence_viz.ipynb"

cells = []
def md(s):   cells.append({"cell_type":"markdown","metadata":{},"source":s.splitlines(keepends=True)})
def code(s): cells.append({"cell_type":"code","metadata":{},"execution_count":None,"outputs":[],"source":s.splitlines(keepends=True)})


# =============================================================================
md(r"""# Correspondence Bench — Visualization & API walkthrough

This notebook is the inspectable face of the benchmark. It:

1. Loads both generators (static + spatiotemporal).
2. Samples scenes at each of the five named operating points.
3. Renders / plots them with **correspondence overlays** at all
   granularities the dataset supports.
4. **Exports GIFs** for the spatiotemporal dataset so collaborators can
   visually inspect the motion.
5. Sanity-checks the label distribution and the correspondence ground
   truth.

For the design philosophy, the full API, and the **downstream-task
catalogue per dataset**, see **`README.md`**.

**Outline:**
- §1 Setup
- §2 Dataset A — Linked Primitives (static): scenes at every difficulty
- §3 A correspondence overlays (lines connecting matched primitives across views)
- §4 A segmentation masks
- §5 Scale-variance histograms
- §6 Dataset B — Linked Primitives Video: scenes at every difficulty
- §7 B — GIF export per view
- §8 B — Cross-view correspondences (same time, two cameras)
- §9 B — Cross-time correspondences within a view (= tracking ground truth)
- §10 B — Trajectory paths visualized in pixel space
- §11 Label distribution sanity checks
""")


# =============================================================================
md("## 1. Setup")
code(r"""import os, sys
sys.path.insert(0, os.path.abspath('.'))
import math
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import ConnectionPatch
from IPython.display import Image as IPImage, display

from linked_primitives import (
    LinkedPrimitivesGenerator, correspondence_pairs,
    OPERATING_POINTS as A_POINTS, SHAPES,
)
from linked_primitives_video import (
    LinkedPrimitivesVideoGenerator,
    cross_view_pairs_at_time, cross_time_pairs_within_view, trajectories_for_view,
    OPERATING_POINTS as B_POINTS,
)

np.random.seed(0)
DIFFS = ["easy", "basic", "hard", "extreme", "adversarial"]
GIF_DIR = "./gifs"
os.makedirs(GIF_DIR, exist_ok=True)
print(f"Operating points (A static):        {list(A_POINTS.keys())}")
print(f"Operating points (B spatiotemporal): {list(B_POINTS.keys())}")
print(f"Shapes available: {SHAPES}")
""")


# =============================================================================
md(r"""## 2. Dataset A — Linked Primitives (static)

One sample scene at each difficulty. Top row: view A. Bottom row: view B
(with whatever style transform is configured at that difficulty).
""")
code(r"""IMG = 128
fig, axes = plt.subplots(2, len(DIFFS), figsize=(4 * len(DIFFS), 8))
for col, name in enumerate(DIFFS):
    gen = LinkedPrimitivesGenerator(operating_point=name, image_size=IMG, base_seed=42)
    scene = gen.sample_scene(seed=42)
    out_A = gen.render(scene, view="A")
    out_B = gen.render(scene, view="B")
    axes[0, col].imshow(out_A["rgb"])
    axes[0, col].set_title(f"{name.upper()}\nview A  (N={len(scene.linked)} linked)", fontsize=10)
    axes[0, col].axis("off")
    axes[1, col].imshow(out_B["rgb"])
    axes[1, col].set_title(f"view B  (style={scene.style_B})", fontsize=10)
    axes[1, col].axis("off")
plt.suptitle("Linked Primitives (static) — one scene per operating point", y=1.02)
plt.tight_layout(); plt.show()
""")


# =============================================================================
md(r"""## 3. A — correspondence overlays

Lines connecting matched primitive centres across the two camera views.
Ground truth from `correspondence_pairs(view_A, view_B)`.
""")
code(r"""for name in ["easy", "basic", "hard"]:
    gen = LinkedPrimitivesGenerator(operating_point=name, image_size=IMG, base_seed=42)
    scene = gen.sample_scene(seed=42)
    out_A = gen.render(scene, view="A")
    out_B = gen.render(scene, view="B")
    pairs = correspondence_pairs(out_A, out_B)

    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    axes[0].imshow(out_A["rgb"]); axes[0].set_title(f"{name.upper()} — view A"); axes[0].axis("off")
    axes[1].imshow(out_B["rgb"]); axes[1].set_title(f"view B  ({pairs.shape[0]} matches)"); axes[1].axis("off")
    for k in range(pairs.shape[0]):
        (xA, yA), (xB, yB) = pairs[k]
        con = ConnectionPatch(xyA=(xA, yA), coordsA=axes[0].transData,
                               xyB=(xB, yB), coordsB=axes[1].transData,
                               color=plt.cm.tab20(k % 20), lw=0.8, alpha=0.7)
        fig.add_artist(con)
        axes[0].plot(xA, yA, 'o', mfc='white', mec='black', ms=4, mew=0.5)
        axes[1].plot(xB, yB, 'o', mfc='white', mec='black', ms=4, mew=0.5)
    plt.tight_layout(); plt.show()
""")


# =============================================================================
md(r"""## 4. A — segmentation masks

Each primitive carries a unique integer ID. The seg map is just that ID
painted into each pixel the primitive occupies. Background = -1.
""")
code(r"""gen = LinkedPrimitivesGenerator(operating_point="basic", image_size=128, base_seed=7)
scene = gen.sample_scene(seed=7)
out_A = gen.render(scene, view="A")
out_B = gen.render(scene, view="B")

fig, axes = plt.subplots(2, 2, figsize=(10, 10))
axes[0, 0].imshow(out_A["rgb"]); axes[0, 0].set_title("view A — RGB"); axes[0, 0].axis("off")
axes[0, 1].imshow(out_A["seg"], cmap="tab20"); axes[0, 1].set_title("view A — primitive IDs"); axes[0, 1].axis("off")
axes[1, 0].imshow(out_B["rgb"]); axes[1, 0].set_title("view B — RGB"); axes[1, 0].axis("off")
axes[1, 1].imshow(out_B["seg"], cmap="tab20"); axes[1, 1].set_title("view B — primitive IDs (same colormap)"); axes[1, 1].axis("off")
plt.tight_layout(); plt.show()

ids_in_both = set(np.unique(out_A["seg"])) & set(np.unique(out_B["seg"])) - {-1}
print(f"primitives visible in BOTH views: {len(ids_in_both)} / {len(scene.linked)} linked")
""")


# =============================================================================
md(r"""## 5. Scale-variance check — multi-scale by construction

Histogram of primitive sizes per operating point. HARD and above span >4×
in the same scene.
""")
code(r"""fig, axes = plt.subplots(1, 5, figsize=(20, 4))
for ax, name in zip(axes, DIFFS):
    gen = LinkedPrimitivesGenerator(operating_point=name, image_size=128, base_seed=11)
    sizes = []
    for s in range(50):
        scene = gen.sample_scene(seed=s)
        sizes += [p.size for p in scene.linked]
    sizes = np.array(sizes)
    ax.hist(sizes, bins=30, color='C0', alpha=0.7)
    s_lo, s_hi = A_POINTS[name]["scale_range"]
    ax.axvline(s_lo, color='k', ls=':', lw=1, label=f"range [{s_lo}, {s_hi}]")
    ax.axvline(s_hi, color='k', ls=':', lw=1)
    ax.set_title(f"{name.upper()}\nratio = {s_hi/s_lo:.1f}x", fontsize=10)
    ax.set_xlabel("primitive size")
    if ax is axes[0]: ax.set_ylabel("count")
    ax.legend(fontsize=7)
plt.suptitle("Scale distribution per operating point — log-uniform on configured range", y=1.02)
plt.tight_layout(); plt.show()
""")


# =============================================================================
md(r"""## 6. Dataset B — Linked Primitives Video (spatiotemporal)

Each scene is now a *video* per view. We show frame 0 and frame T/2 side
by side per difficulty (top: view A, bottom: view B). Motion is visible
between the two frames.
""")
code(r"""IMG = 128
fig, axes = plt.subplots(4, len(DIFFS), figsize=(4 * len(DIFFS), 14))
for col, name in enumerate(DIFFS):
    gen = LinkedPrimitivesVideoGenerator(operating_point=name, image_size=IMG, base_seed=42)
    scene = gen.sample_scene(seed=42)
    video = gen.render_video_pair(scene)
    T = video["view_A"]["rgb"].shape[0]
    f0, fm = 0, T // 2
    axes[0, col].imshow(video["view_A"]["rgb"][f0])
    axes[0, col].set_title(f"{name.upper()}\nview A  τ=0", fontsize=9); axes[0, col].axis("off")
    axes[1, col].imshow(video["view_A"]["rgb"][fm])
    axes[1, col].set_title(f"view A  τ≈0.5", fontsize=9); axes[1, col].axis("off")
    axes[2, col].imshow(video["view_B"]["rgb"][f0])
    axes[2, col].set_title(f"view B  τ=0  (style={scene.style_B})", fontsize=9); axes[2, col].axis("off")
    axes[3, col].imshow(video["view_B"]["rgb"][fm])
    axes[3, col].set_title(f"view B  τ≈0.5", fontsize=9); axes[3, col].axis("off")
plt.suptitle("Linked Primitives Video — first vs midpoint frame, per operating point", y=1.01)
plt.tight_layout(); plt.show()
""")


# =============================================================================
md(r"""## 7. B — GIF export

`save_gif` writes a (T, H, W, 3) frame stack to disk as an animated GIF.
We render `basic` and `hard` and inline the GIFs.
""")
code(r"""for name in ["basic", "hard"]:
    gen = LinkedPrimitivesVideoGenerator(operating_point=name, image_size=128, base_seed=3)
    scene = gen.sample_scene(seed=3)
    video = gen.render_video_pair(scene)
    for view in ("A", "B"):
        path = os.path.join(GIF_DIR, f"{name}_view_{view}.gif")
        gen.save_gif(video[f"view_{view}"]["rgb"], path, fps=scene.fps)
        print(f"  wrote {path}  ({video[f'view_{view}']['rgb'].shape}, fps={scene.fps})")
    # Display the basic ones inline
    if name == "basic":
        for view in ("A", "B"):
            print(f"\n{name.upper()} — view {view}:")
            display(IPImage(filename=os.path.join(GIF_DIR, f"{name}_view_{view}.gif")))
""")


# =============================================================================
md(r"""## 8. B — Cross-view correspondences (same time, two cameras)

For a single frame `t_idx`, draw lines between matched primitive centres
across views. This is correspondence type **(1)** from the README.
""")
code(r"""gen = LinkedPrimitivesVideoGenerator(operating_point="basic", image_size=128, base_seed=3)
scene = gen.sample_scene(seed=3)
video = gen.render_video_pair(scene)
T = video["view_A"]["rgb"].shape[0]
frames_to_show = [0, T // 2, T - 1]
fig, axes = plt.subplots(len(frames_to_show), 2, figsize=(10, 5 * len(frames_to_show)))
for row, t in enumerate(frames_to_show):
    pairs = cross_view_pairs_at_time(video, t)
    axA = axes[row, 0]; axB = axes[row, 1]
    axA.imshow(video["view_A"]["rgb"][t]); axA.set_title(f"view A  frame {t}"); axA.axis("off")
    axB.imshow(video["view_B"]["rgb"][t]); axB.set_title(f"view B  frame {t}  ({pairs.shape[0]} matches)"); axB.axis("off")
    for k in range(pairs.shape[0]):
        (xA, yA), (xB, yB) = pairs[k]
        con = ConnectionPatch(xyA=(xA, yA), coordsA=axA.transData,
                               xyB=(xB, yB), coordsB=axB.transData,
                               color=plt.cm.tab20(k % 20), lw=0.8, alpha=0.7)
        fig.add_artist(con)
        axA.plot(xA, yA, 'o', mfc='white', mec='black', ms=3, mew=0.4)
        axB.plot(xB, yB, 'o', mfc='white', mec='black', ms=3, mew=0.4)
plt.tight_layout(); plt.show()
""")


# =============================================================================
md(r"""## 9. B — Cross-time correspondences within a view (tracking ground truth)

For a single view, draw lines between matched primitive centres across
two time indices. This is correspondence type **(2)** from the README —
the ground truth used to evaluate tracking and optical-flow estimation.
""")
code(r"""gen = LinkedPrimitivesVideoGenerator(operating_point="hard", image_size=128, base_seed=5)
scene = gen.sample_scene(seed=5)
video = gen.render_video_pair(scene)
T = video["view_A"]["rgb"].shape[0]
t1, t2 = 0, T - 1

pairs = cross_time_pairs_within_view(video, "A", t1, t2)
fig, axes = plt.subplots(1, 2, figsize=(10, 5))
axes[0].imshow(video["view_A"]["rgb"][t1]); axes[0].set_title(f"view A  τ={t1}")
axes[0].axis("off")
axes[1].imshow(video["view_A"]["rgb"][t2]); axes[1].set_title(f"view A  τ={t2}  ({pairs.shape[0]} tracked)")
axes[1].axis("off")
for k in range(pairs.shape[0]):
    (x1, y1), (x2, y2) = pairs[k]
    con = ConnectionPatch(xyA=(x1, y1), coordsA=axes[0].transData,
                           xyB=(x2, y2), coordsB=axes[1].transData,
                           color=plt.cm.tab20(k % 20), lw=0.8, alpha=0.6)
    fig.add_artist(con)
    axes[0].plot(x1, y1, 'o', mfc='white', mec='black', ms=3, mew=0.4)
    axes[1].plot(x2, y2, 'o', mfc='white', mec='black', ms=3, mew=0.4)
plt.tight_layout(); plt.show()
""")


# =============================================================================
md(r"""## 10. B — Trajectory paths in pixel space

Plot every primitive's full pixel-trajectory in one view as a line.
`trajectories_for_view` returns `{pid: (T, 2)}` and we overlay the lines
on the first frame.
""")
code(r"""gen = LinkedPrimitivesVideoGenerator(operating_point="hard", image_size=128, base_seed=5)
scene = gen.sample_scene(seed=5)
video = gen.render_video_pair(scene)
trajs = trajectories_for_view(video, "A")

fig, ax = plt.subplots(figsize=(7, 7))
ax.imshow(video["view_A"]["rgb"][0])
for pid, path in trajs.items():
    if pid < 0: continue
    mask = ~np.isnan(path[:, 0])
    if mask.sum() < 2: continue
    ax.plot(path[mask, 0], path[mask, 1],
             '-', color=plt.cm.tab20(int(pid) % 20), lw=1.4, alpha=0.85)
    ax.plot(path[mask, 0][0], path[mask, 1][0],
             'o', color=plt.cm.tab20(int(pid) % 20), ms=5, mec='black', mew=0.4)
ax.set_title(f"view A — per-primitive pixel trajectories  ({len(trajs)} pids, T={video['view_A']['rgb'].shape[0]})")
ax.axis("off")
plt.tight_layout(); plt.show()
""")


# =============================================================================
md(r"""## 11. Label distribution sanity check

Sample many scenes per operating point and confirm the labels are
reasonably balanced.
""")
code(r"""N_PER = 300
fig, axes = plt.subplots(2, len(DIFFS), figsize=(4 * len(DIFFS), 7))
for col, name in enumerate(DIFFS):
    # Dataset A — count_modulo_4
    gen_A = LinkedPrimitivesGenerator(operating_point=name, image_size=64, base_seed=0)
    labels_A = []
    for s in range(N_PER):
        scene = gen_A.sample_scene(seed=s)
        labels_A.append(gen_A.compute_label(scene, kind="count_modulo_K", K=4))
    axes[0, col].hist(labels_A, bins=np.arange(5) - 0.5, color='C0', rwidth=0.8)
    axes[0, col].set_title(f"A: {name}\ncount mod 4", fontsize=9)
    axes[0, col].set_xticks(range(4))
    if col == 0: axes[0, col].set_ylabel(f"count over {N_PER} scenes")

    # Dataset B — has_motion_pattern
    gen_B = LinkedPrimitivesVideoGenerator(operating_point=name, image_size=64, base_seed=0)
    labels_B = []
    for s in range(N_PER):
        scene = gen_B.sample_scene(seed=s)
        labels_B.append(gen_B.compute_label(scene, kind="has_motion_pattern"))
    axes[1, col].hist(labels_B, bins=np.arange(3) - 0.5, color='C3', rwidth=0.8)
    axes[1, col].set_title(f"B: {name}\nhas_motion_pattern", fontsize=9)
    axes[1, col].set_xticks([0, 1])
    if col == 0: axes[1, col].set_ylabel(f"count over {N_PER} scenes")
plt.tight_layout(); plt.show()
""")


# =============================================================================
md(r"""## 12. Recap & next steps

Validated:

- **Dataset A** renders both views, returns segmentation masks +
  correspondence keypoints, balances labels.
- **Dataset B** renders video pairs, exports GIFs, returns all three
  flavours of correspondence ground truth, and tracks every primitive's
  pixel trajectory across frames.

Next steps in the project:

1. **Supervised oracle baseline.** Train a small model with full latent
   access (`compute_label` ground truth) to confirm each task is
   solvable — sets the ceiling for SSL evaluation.
2. **SSL recipe scaffolding.** Wire dataset B into the FPS+KNN-patch
   pipeline used elsewhere in this repo (replace CIFAR-10 events with
   per-frame primitive centroids).
3. **Three-metric evaluation sweep.** For each operating point and each
   SSL recipe, report (i) cross-modal retrieval, (ii) linear probe on
   `compute_label`, (iii) cross-time tracking RMSE / dense
   cross-modal-correspondence mIoU.
4. **Plot the difficulty × accuracy curve** for each SSL method. This is
   the headline figure that distinguishes methods which scale gracefully
   to fine-grained correspondence from those that plateau.
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
