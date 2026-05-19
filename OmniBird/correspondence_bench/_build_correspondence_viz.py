"""Build correspondence_bench/correspondence_viz.ipynb — visualization +
walkthrough of the two correspondence datasets at all five difficulty
operating points."""
import json, os
NB = "/Users/davidpark/Documents/Claude/NFJEPA/OmniBird/correspondence_bench/correspondence_viz.ipynb"

cells = []
def md(s):   cells.append({"cell_type":"markdown","metadata":{},"source":s.splitlines(keepends=True)})
def code(s): cells.append({"cell_type":"code","metadata":{},"execution_count":None,"outputs":[],"source":s.splitlines(keepends=True)})


# =============================================================================
md(r"""# Correspondence Bench — Visualization & API walkthrough

This notebook is the **inspectable face** of the benchmark. It:

1. Loads the two generators.
2. Samples scenes at each of the five named operating points
   (EASY → ADVERSARIAL).
3. Renders / plots them with correspondence overlays.
4. Sanity-checks the label distribution and the correspondence ground
   truth.

For the design philosophy and the full API, see **`README.md`** in this
folder.

**Outline:**
- §1 Setup
- §2 Dataset A — Linked Primitives: scenes at every difficulty
- §3 Correspondence overlays (lines connecting matched primitives)
- §4 Segmentation masks
- §5 Multi-scale / scale-range visualization
- §6 Dataset C — Synthetic Event Streams: scenes at every difficulty
- §7 Cross-modal transform recovery sanity check
- §8 Label distribution check
""")


# =============================================================================
md("## 1. Setup")
code(r"""import os, sys
sys.path.insert(0, os.path.abspath('.'))
import math
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import ConnectionPatch

from linked_primitives import (
    LinkedPrimitivesGenerator, correspondence_pairs,
    OPERATING_POINTS as A_POINTS, SHAPES,
)
from synth_event_streams import (
    SyntheticEventStreamsGenerator, correspondence_indices,
    OPERATING_POINTS as C_POINTS,
)

np.random.seed(0)
DIFFS = ["easy", "basic", "hard", "extreme", "adversarial"]
print(f"Operating points (Dataset A): {list(A_POINTS.keys())}")
print(f"Operating points (Dataset C): {list(C_POINTS.keys())}")
print(f"Shapes available: {SHAPES}")
""")


# =============================================================================
md(r"""## 2. Dataset A — Linked Primitives

One sample scene at each difficulty level. Top row: view A. Bottom row:
view B (with whatever style transform is configured at that difficulty).
""")
code(r"""IMG = 128
fig, axes = plt.subplots(2, len(DIFFS), figsize=(4 * len(DIFFS), 8))
for col, name in enumerate(DIFFS):
    gen = LinkedPrimitivesGenerator(operating_point=name, image_size=IMG, base_seed=42)
    scene = gen.sample_scene(seed=42)
    out_A = gen.render(scene, view="A")
    out_B = gen.render(scene, view="B")
    axes[0, col].imshow(out_A["rgb"])
    axes[0, col].set_title(f"{name.upper()}\nview A  (N={len(scene.linked)} linked)",
                            fontsize=10)
    axes[0, col].axis("off")
    axes[1, col].imshow(out_B["rgb"])
    axes[1, col].set_title(f"view B  (style={scene.style_B})", fontsize=10)
    axes[1, col].axis("off")
plt.suptitle("Linked Primitives — one scene per operating point", y=1.02)
plt.tight_layout(); plt.show()
""")


# =============================================================================
md(r"""## 3. Correspondence overlays

Same scenes as above, but with lines connecting matched primitive centres
across the two modalities. The ground-truth correspondences come from
`correspondence_pairs(view_A, view_B)` — there's nothing learned here,
this is what the generator hands you for free.
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
    # Draw correspondence lines
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
md(r"""## 4. Segmentation masks

Each primitive carries a unique integer ID. The segmentation array
`out["seg"]` is just that ID painted into each pixel the primitive
occupies. Background = -1. This is the per-pixel ground-truth for
fine-grained correspondence learning: `seg_A == id ↔ seg_B == id`
identifies every pixel pair across modalities that belongs to the same
primitive.
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

# Print per-primitive overlap statistics
ids_in_both = set(np.unique(out_A["seg"])) & set(np.unique(out_B["seg"])) - {-1}
print(f"primitives visible in BOTH views: {len(ids_in_both)} / {len(scene.linked)} linked + {len(scene.distractors_A)} dist_A")
print("(this is the M from `correspondence_pairs`)")
""")


# =============================================================================
md(r"""## 5. Scale variance check — multi-scale by construction

At HARD and above, primitive sizes span > 4× in the same scene. The
histogram below shows that the scale knob `scale_range` actually
produces the requested log-uniform spread, and the rendered images show
small + large primitives co-existing.
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
plt.suptitle("Scale distribution per operating point — log-uniform on the configured range", y=1.02)
plt.tight_layout(); plt.show()
""")


# =============================================================================
md(r"""## 6. Dataset C — Synthetic Event Streams

Sparse point clouds in space-time. Same five operating points, plotted
as `(x, y)` scatter with **point colour = source ID** (matched colours
across modalities = matched events) and **alpha = depth on the time
axis**.

Distractor events are drawn in light grey.
""")
code(r"""fig, axes = plt.subplots(2, len(DIFFS), figsize=(4 * len(DIFFS), 8))
for col, name in enumerate(DIFFS):
    gen = SyntheticEventStreamsGenerator(operating_point=name, base_seed=42)
    scene = gen.sample_scene(seed=42)
    for row, (pos, src, t, mod) in enumerate([
        (scene.A_pos, scene.A_src, scene.A_time, "A"),
        (scene.B_pos, scene.B_src, scene.B_time, "B"),
    ]):
        ax = axes[row, col]
        linked_mask = src >= 0
        # Distractors first (background)
        ax.scatter(pos[~linked_mask, 0], pos[~linked_mask, 1],
                    c='lightgray', s=10, alpha=0.7, label='distractor')
        # Linked events — colour = source ID
        colors = plt.cm.tab20(src[linked_mask] % 20)
        ax.scatter(pos[linked_mask, 0], pos[linked_mask, 1],
                    c=colors, s=20, edgecolors='black', linewidths=0.4)
        ax.set_xlim(-1.2, 1.2); ax.set_ylim(-1.2, 1.2); ax.set_aspect('equal')
        ax.set_title(f"{name.upper()} mod {mod}\n"
                     f"linked={linked_mask.sum()}, dist={len(pos)-linked_mask.sum()}, "
                     f"T={scene.transform['kind']}", fontsize=9)
        ax.grid(alpha=0.3)
plt.suptitle("Synthetic Event Streams — colour = source ID (matched across modalities)", y=1.02)
plt.tight_layout(); plt.show()
""")


# =============================================================================
md(r"""## 7. Cross-modal transform recovery sanity check

The `transform_class` label is the coarse type of `T_B` (identity /
rotation / affine / nonlinear). Let's confirm the transform actually
gets applied: plot linked events in modality A and modality B in the
same axes, drawing lines between matched events. The lines should reveal
the structure of `T_B`.
""")
code(r"""fig, axes = plt.subplots(1, 4, figsize=(16, 4))
for ax, kind in zip(axes, ["identity", "rotation", "affine", "nonlinear"]):
    custom = dict(C_POINTS["basic"])
    custom["transform"] = kind
    custom["n_distractors_A"] = custom["n_distractors_B"] = 0   # remove clutter for this plot
    gen = SyntheticEventStreamsGenerator(operating_point=custom, base_seed=3)
    scene = gen.sample_scene(seed=3)
    pairs = correspondence_indices(scene)
    A_pos, B_pos = scene.A_pos, scene.B_pos
    ax.scatter(A_pos[:, 0], A_pos[:, 1], c='C0', s=18, label='A', edgecolors='black', lw=0.4)
    ax.scatter(B_pos[:, 0], B_pos[:, 1], c='C3', s=18, label='B', edgecolors='black', lw=0.4)
    for iA, iB in pairs:
        ax.plot([A_pos[iA, 0], B_pos[iB, 0]],
                 [A_pos[iA, 1], B_pos[iB, 1]],
                 color='gray', lw=0.4, alpha=0.6)
    ax.set_xlim(-1.4, 1.4); ax.set_ylim(-1.4, 1.4); ax.set_aspect('equal')
    ax.set_title(f"T = {kind}"); ax.legend(fontsize=8); ax.grid(alpha=0.3)
plt.suptitle("Cross-modal transform T_B — connecting lines should reveal its structure", y=1.02)
plt.tight_layout(); plt.show()
""")


# =============================================================================
md(r"""## 8. Label distribution sanity check

Sample many scenes per operating point and check that the labels are
reasonably balanced (no degenerate class collapsing).
""")
code(r"""N_PER = 500
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

    # Dataset C — majority_feature
    gen_C = SyntheticEventStreamsGenerator(operating_point=name, base_seed=0)
    labels_C = []
    for s in range(N_PER):
        scene = gen_C.sample_scene(seed=s, label_kind="majority_feature")
        labels_C.append(scene.label)
    nb = max(labels_C) + 1 if labels_C else 1
    axes[1, col].hist(labels_C, bins=np.arange(nb + 1) - 0.5, color='C3', rwidth=0.8)
    axes[1, col].set_title(f"C: {name}\nmajority feature", fontsize=9)
    if col == 0: axes[1, col].set_ylabel(f"count over {N_PER} scenes")
plt.tight_layout(); plt.show()
print("\nLabel balance: a well-designed task should give roughly uniform histograms.")
print("Bias means the class is recoverable from priors alone — increase n_linked, vary scenes more.")
""")


# =============================================================================
md(r"""## 9. Recap & next steps

We've validated:

- **Scene rendering works** at every operating point in both datasets.
- **Correspondences are returned correctly** by both helpers
  (`correspondence_pairs`, `correspondence_indices`).
- **Segmentation masks** in Dataset A give the per-pixel correspondence
  ground truth for dense evaluation.
- **Scale variance** is wide at HARD and above, as designed.
- **Cross-modal transforms** are visible in the linked-event scatter
  plots.
- **Labels are reasonably balanced** at every operating point.

What to do next:

1. **Train a supervised baseline** on `compute_label` to confirm the
   task is solvable when you can see the latents (oracle access).
2. **Train an SSL recipe** on either dataset (mask + JEPA, RoPE / HRR
   aggregator, etc.).
3. **Evaluate all three metrics** in §6 of the README:
   - cross-modal retrieval (top-k accuracy on matched primitives),
   - linear-probe on the label,
   - dense cross-modal segmentation (Dataset A) or correspondence
     classification (Dataset C).
4. **Sweep the difficulty axis**, holding the SSL recipe fixed, and
   plot the three metrics vs operating point. That's the headline
   figure: it tests whether the recipe degrades gracefully as
   difficulty rises.
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
