"""Build correspondence_bench/correspondence_viz.ipynb — comprehensive
walkthrough: intro + spread/Hz visualizations + every downstream task
demoed for both datasets."""
import json, os
NB = "/Users/davidpark/Documents/Claude/NFJEPA/OmniBird/correspondence_bench/correspondence_viz.ipynb"

cells = []
def md(s):   cells.append({"cell_type":"markdown","metadata":{},"source":s.splitlines(keepends=True)})
def code(s): cells.append({"cell_type":"code","metadata":{},"execution_count":None,"outputs":[],"source":s.splitlines(keepends=True)})


# =============================================================================
md(r"""# Correspondence Bench — Visualization & Walkthrough

A difficulty-controlled multimodal-correspondence benchmark for SSL
evaluation. Two datasets share a generative-model core and a parallel
naming for difficulty knobs:

| | Dataset | What it returns | When you'd use it |
|---|---|---|---|
| **A** | **Linked Primitives** (`linked_primitives.py`) | one *image pair*: two camera views of a 3D primitive scene at a single instant | classification, per-image segmentation, cross-view keypoint retrieval, dense cross-modal alignment |
| **B** | **Linked Primitives Video** (`linked_primitives_video.py`) | one *video pair*: two camera views of the same scene over time, plus per-frame seg masks + keypoints, plus an animated GIF | video classification, spatiotemporal segmentation (tubes), within-view tracking, optical flow, cross-view-cross-time matching, motion-conditioned classification |

**Why this exists.** Standard multimodal datasets often let you solve
the task without doing fine correspondence at all (global statistics
suffice). This benchmark is engineered so the **label depends only on
the latent scene** — recovering the label requires recovering a
sufficient statistic of the latents, which requires fine cross-modal
(and, in B, cross-time) correspondence. Each primitive carries a
**stable integer ID** that's visible wherever it appears, so the
correspondence ground truth is returned for free with every sample.

**Difficulty axes (knobs in `OPERATING_POINTS`):**

Both datasets ship five base operating points — `easy / basic / hard /
extreme / adversarial` — spanning a 32× spread in linked-primitive
count and progressively widening style mismatch, distractor density,
noise, scale range, and view disparity. Dataset B adds **four extra
operating points to test frequency (Hz) handling**:

| Operating point | Frequency range (cycles per video) | Tests |
|---|---|---|
| `slow_only` | (0.25, 1.0) | Can the model detect SLOW dynamics? |
| `fast_only` | (3.0, 8.0) | Can it detect FAST dynamics? |
| `mixed_hz` | (0.25, 6.0) — sampled per primitive | Can it handle BOTH simultaneously in one scene? |
| `multiscale_hz` | (0.25, 8.0) on a harder spatial backbone | Multi-Hz + multi-shape + style gap |

Frequencies are sampled **log-uniformly** across the configured range,
so a scene with `mixed_hz` contains both slow and fast primitives at
the same time. This is the core "can the model handle multi-Hz
simultaneously" stress test.

**All seven downstream tasks for B (and four for A) are demonstrated
below** with explicit visualizations and label printouts.

---

**Notebook structure**

| §  | Section |
|----|----|
| §1 | Setup + ops summary |
| §2 | Dataset A scenes (all operating points) — observe spread reaches image edges |
| §3 | A-1: Image-pair classification |
| §4 | A-2: Per-modality segmentation |
| §5 | A-3: Cross-view keypoint retrieval |
| §6 | A-4: Dense cross-modal alignment (per-pixel) |
| §7 | A multi-scale check |
| §8 | Dataset B scenes (all operating points) |
| §9 | B GIF export |
| §10 | **Hz variation: slow_only / fast_only / mixed_hz / multiscale_hz** |
| §11 | B-1: Video-pair classification |
| §12 | B-2: Spatiotemporal segmentation (tubes) |
| §13 | B-3: Within-view tracking |
| §14 | B-4: Cross-view kpt retrieval over time |
| §15 | B-5: Cross-view × cross-time matching |
| §16 | B-6: Dense optical flow within view |
| §17 | B-7: Motion-conditioned classification |
| §18 | Label distribution sanity check |
""")


# =============================================================================
md("## §1. Setup")
code(r"""import os, sys, math
sys.path.insert(0, os.path.abspath('.'))
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
BASE_DIFFS = ["easy", "basic", "hard", "extreme", "adversarial"]
HZ_OPS = ["slow_only", "fast_only", "mixed_hz", "multiscale_hz"]
GIF_DIR = "./gifs"; os.makedirs(GIF_DIR, exist_ok=True)
print(f"Dataset A operating points: {list(A_POINTS.keys())}")
print(f"Dataset B operating points: {list(B_POINTS.keys())}")
print(f"Shapes available: {SHAPES}")
""")


# =============================================================================
md(r"""## §2. Dataset A — scenes at every operating point

After widening the position range, primitives now reach the image edges
instead of clustering near the center.
""")
code(r"""IMG = 128
fig, axes = plt.subplots(2, len(BASE_DIFFS), figsize=(4 * len(BASE_DIFFS), 8))
for col, name in enumerate(BASE_DIFFS):
    gen = LinkedPrimitivesGenerator(operating_point=name, image_size=IMG, base_seed=42)
    scene = gen.sample_scene(seed=42)
    out_A = gen.render(scene, view="A"); out_B = gen.render(scene, view="B")
    axes[0, col].imshow(out_A["rgb"]); axes[0, col].axis("off")
    axes[0, col].set_title(f"{name.upper()}\nview A — N={len(scene.linked)} linked", fontsize=10)
    axes[1, col].imshow(out_B["rgb"]); axes[1, col].axis("off")
    axes[1, col].set_title(f"view B  ({scene.style_B})", fontsize=10)
plt.suptitle("Dataset A — Linked Primitives (static)", y=1.02)
plt.tight_layout(); plt.show()
""")


# =============================================================================
md(r"""## §3. Downstream A-1 — image-pair classification

**Input:** the pair `(img_A, img_B)`.
**Output:** a single class label `y = compute_label(scene, kind=...)`.
**Ground truth:** computed from the latent scene description, not the
rendered pixels — so it cannot be solved by global image statistics.
""")
code(r"""gen = LinkedPrimitivesGenerator(operating_point="basic", image_size=128, base_seed=0)
fig, axes = plt.subplots(3, 2, figsize=(8, 12))
for row, seed in enumerate([0, 17, 42]):
    scene = gen.sample_scene(seed=seed)
    out_A = gen.render(scene, view="A"); out_B = gen.render(scene, view="B")
    y_count = gen.compute_label(scene, kind="count_modulo_K", K=4)
    y_pair  = gen.compute_label(scene, kind="has_pair")
    y_dist  = gen.compute_label(scene, kind="n_distinct_pairs", K=4)
    axes[row, 0].imshow(out_A["rgb"]); axes[row, 0].axis("off")
    axes[row, 0].set_title(f"seed={seed}  view A", fontsize=9)
    axes[row, 1].imshow(out_B["rgb"]); axes[row, 1].axis("off")
    axes[row, 1].set_title(
        f"view B  |  labels:\n  count_mod4={y_count}, has_pair={y_pair}, n_distinct={y_dist}",
        fontsize=8,
    )
plt.tight_layout(); plt.show()
""")


# =============================================================================
md(r"""## §4. Downstream A-2 — per-modality segmentation

**Input:** one image (either modality).
**Output:** a `(H, W)` map of primitive IDs (`-1` = background).
**Ground truth:** `out["seg"]` from the render.
""")
code(r"""gen = LinkedPrimitivesGenerator(operating_point="basic", image_size=128, base_seed=7)
scene = gen.sample_scene(seed=7)
out_A = gen.render(scene, view="A"); out_B = gen.render(scene, view="B")
fig, axes = plt.subplots(2, 2, figsize=(10, 10))
axes[0, 0].imshow(out_A["rgb"]); axes[0, 0].set_title("view A — RGB"); axes[0, 0].axis("off")
axes[0, 1].imshow(out_A["seg"], cmap="tab20"); axes[0, 1].set_title("view A — primitive IDs"); axes[0, 1].axis("off")
axes[1, 0].imshow(out_B["rgb"]); axes[1, 0].set_title("view B — RGB"); axes[1, 0].axis("off")
axes[1, 1].imshow(out_B["seg"], cmap="tab20"); axes[1, 1].set_title("view B — primitive IDs (same colormap)"); axes[1, 1].axis("off")
plt.tight_layout(); plt.show()
""")


# =============================================================================
md(r"""## §5. Downstream A-3 — cross-view keypoint retrieval

**Input:** features at primitive centers in modality A.
**Output:** for each kpt in A, the matching kpt in B (top-k retrieval).
**Ground truth:** `correspondence_pairs(view_A, view_B)` returns
`(M, 2, 2)` where each row is `((x_A, y_A), (x_B, y_B))`.
""")
code(r"""for name in ["easy", "basic", "hard"]:
    gen = LinkedPrimitivesGenerator(operating_point=name, image_size=128, base_seed=42)
    scene = gen.sample_scene(seed=42)
    out_A = gen.render(scene, view="A"); out_B = gen.render(scene, view="B")
    pairs = correspondence_pairs(out_A, out_B)
    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    axes[0].imshow(out_A["rgb"]); axes[0].set_title(f"{name.upper()}  view A"); axes[0].axis("off")
    axes[1].imshow(out_B["rgb"]); axes[1].set_title(f"view B  ({pairs.shape[0]} GT matches)"); axes[1].axis("off")
    for k in range(pairs.shape[0]):
        (xA, yA), (xB, yB) = pairs[k]
        con = ConnectionPatch(xyA=(xA, yA), coordsA=axes[0].transData,
                               xyB=(xB, yB), coordsB=axes[1].transData,
                               color=plt.cm.tab20(k % 20), lw=0.8, alpha=0.7)
        fig.add_artist(con)
        axes[0].plot(xA, yA, 'o', mfc='white', mec='black', ms=4, mew=0.4)
        axes[1].plot(xB, yB, 'o', mfc='white', mec='black', ms=4, mew=0.4)
    plt.tight_layout(); plt.show()
""")


# =============================================================================
md(r"""## §6. Downstream A-4 — dense cross-modal alignment

**Input:** pixel `(x_A, y_A)` in modality A.
**Output:** the matching pixel `(x_B, y_B)` in modality B that belongs
to the same primitive (or "no match" if not visible in B).
**Ground truth:** `seg_A[y_A, x_A] == pid  ⇒  any pixel where seg_B == pid`.

The viz: pick 6 random labelled pixels in A, draw an arrow to a single
representative pixel of the matching primitive in B.
""")
code(r"""gen = LinkedPrimitivesGenerator(operating_point="basic", image_size=128, base_seed=33)
scene = gen.sample_scene(seed=33)
out_A = gen.render(scene, view="A"); out_B = gen.render(scene, view="B")
seg_A, seg_B = out_A["seg"], out_B["seg"]
rng = np.random.RandomState(5)

# Find 6 random foreground pixels in A whose pid is also in B
candidates = np.argwhere(seg_A >= 0)
rng.shuffle(candidates)
chosen = []
for (y, x) in candidates:
    pid = int(seg_A[y, x])
    if pid in seg_B and len(chosen) < 6:
        chosen.append((y, x, pid))

fig, axes = plt.subplots(1, 2, figsize=(10, 5))
axes[0].imshow(out_A["rgb"]); axes[0].set_title("view A  — sampled pixels"); axes[0].axis("off")
axes[1].imshow(out_B["rgb"]); axes[1].set_title("view B  — matched pixels"); axes[1].axis("off")
for k, (y, x, pid) in enumerate(chosen):
    # Representative match in B = mean of pixels where seg_B == pid
    ys, xs = np.where(seg_B == pid)
    yB, xB = float(ys.mean()), float(xs.mean())
    c = plt.cm.tab10(k)
    axes[0].plot(x, y, 'o', mfc=c, mec='black', ms=10, mew=1.0)
    axes[1].plot(xB, yB, 'o', mfc=c, mec='black', ms=10, mew=1.0)
    con = ConnectionPatch(xyA=(x, y), coordsA=axes[0].transData,
                           xyB=(xB, yB), coordsB=axes[1].transData,
                           color=c, lw=1.0, alpha=0.7)
    fig.add_artist(con)
plt.tight_layout(); plt.show()
""")


# =============================================================================
md(r"""## §7. Multi-scale check — primitive size distribution

`scale_range` is log-uniform per primitive. At `hard` and above, scenes
contain primitives spanning >4× in size — explicitly testing
scale-equivariance of the encoder.
""")
code(r"""fig, axes = plt.subplots(1, 5, figsize=(20, 4))
for ax, name in zip(axes, BASE_DIFFS):
    gen = LinkedPrimitivesGenerator(operating_point=name, image_size=128, base_seed=11)
    sizes = []
    for s in range(50):
        scene = gen.sample_scene(seed=s)
        sizes += [p.size for p in scene.linked]
    sizes = np.array(sizes)
    ax.hist(sizes, bins=30, color='C0', alpha=0.7)
    s_lo, s_hi = A_POINTS[name]["scale_range"]
    ax.axvline(s_lo, color='k', ls=':', lw=1); ax.axvline(s_hi, color='k', ls=':', lw=1)
    ax.set_title(f"{name.upper()}  ratio={s_hi/s_lo:.1f}×", fontsize=10)
    ax.set_xlabel("primitive size")
    if ax is axes[0]: ax.set_ylabel("count")
plt.suptitle("Size distribution per operating point — log-uniform on configured range", y=1.02)
plt.tight_layout(); plt.show()
""")


# =============================================================================
md(r"""## §8. Dataset B — scenes at every base operating point

First and midpoint frames per difficulty, top: view A, bottom: view B.
Motion is visible between the two frames.
""")
code(r"""fig, axes = plt.subplots(4, len(BASE_DIFFS), figsize=(4 * len(BASE_DIFFS), 14))
for col, name in enumerate(BASE_DIFFS):
    gen = LinkedPrimitivesVideoGenerator(operating_point=name, image_size=128, base_seed=42)
    scene = gen.sample_scene(seed=42)
    video = gen.render_video_pair(scene)
    T = video["view_A"]["rgb"].shape[0]; f0, fm = 0, T // 2
    axes[0, col].imshow(video["view_A"]["rgb"][f0]); axes[0, col].set_title(f"{name.upper()}\nA  τ=0", fontsize=9); axes[0, col].axis("off")
    axes[1, col].imshow(video["view_A"]["rgb"][fm]); axes[1, col].set_title(f"A  τ≈0.5", fontsize=9); axes[1, col].axis("off")
    axes[2, col].imshow(video["view_B"]["rgb"][f0]); axes[2, col].set_title(f"B  τ=0  ({scene.style_B})", fontsize=9); axes[2, col].axis("off")
    axes[3, col].imshow(video["view_B"]["rgb"][fm]); axes[3, col].set_title(f"B  τ≈0.5", fontsize=9); axes[3, col].axis("off")
plt.suptitle("Dataset B — first vs midpoint frame, per operating point", y=1.01)
plt.tight_layout(); plt.show()
""")


# =============================================================================
md(r"""## §9. B — GIF export

`save_gif` writes a `(T, H, W, 3)` frame stack as an animated GIF. We
render `basic` and `hard` and inline the basic GIFs.
""")
code(r"""for name in ["basic", "hard"]:
    gen = LinkedPrimitivesVideoGenerator(operating_point=name, image_size=128, base_seed=3)
    scene = gen.sample_scene(seed=3)
    video = gen.render_video_pair(scene)
    for view in ("A", "B"):
        path = os.path.join(GIF_DIR, f"{name}_view_{view}.gif")
        gen.save_gif(video[f"view_{view}"]["rgb"], path, fps=scene.fps)
        print(f"  wrote {path}  (T={video[f'view_{view}']['rgb'].shape[0]}, fps={scene.fps})")
    if name == "basic":
        for view in ("A", "B"):
            print(f"\n{name.upper()} — view {view}:")
            display(IPImage(filename=os.path.join(GIF_DIR, f"{name}_view_{view}.gif")))
""")


# =============================================================================
md(r"""## §10. Hz variation — slow / fast / mixed / multiscale

Four Hz-focused operating points test different dynamic regimes:

- **`slow_only`** — all primitives at low Hz (one cycle or fewer per video)
- **`fast_only`** — all primitives at high Hz (3–8 cycles per video)
- **`mixed_hz`** — wide log-uniform range; each scene contains both slow
  and fast primitives simultaneously
- **`multiscale_hz`** — wide range on a harder spatial backbone

Below, for each Hz operating point we show:
1. The first three frames (view A) — fast scenes will look more chaotic.
2. **Per-primitive pixel trajectories**, colored by frequency (blue=slow,
   red=fast). This is the diagnostic that reveals what the model has to
   handle.
""")
code(r"""fig, axes = plt.subplots(2, len(HZ_OPS), figsize=(5 * len(HZ_OPS), 9))
for col, name in enumerate(HZ_OPS):
    gen = LinkedPrimitivesVideoGenerator(operating_point=name, image_size=128, base_seed=2)
    scene = gen.sample_scene(seed=2)
    video = gen.render_video_pair(scene)
    T = video["view_A"]["rgb"].shape[0]

    # Top row: 3-frame strip
    axes[0, col].imshow(np.concatenate([
        video["view_A"]["rgb"][0],
        video["view_A"]["rgb"][T // 2],
        video["view_A"]["rgb"][-1],
    ], axis=1))
    f_lo, f_hi = B_POINTS[name]["frequency_range"]
    axes[0, col].set_title(f"{name.upper()}\n"
                            f"freq range = [{f_lo:.2f}, {f_hi:.2f}] cycles/video  "
                            f"(τ=0, mid, end)", fontsize=10)
    axes[0, col].axis("off")

    # Bottom row: pixel trajectories colored by freq
    ax = axes[1, col]; ax.imshow(video["view_A"]["rgb"][0])
    trajs = trajectories_for_view(video, "A")
    # Build a fresh color map from freqs
    freqs_per_pid = {p.pid: abs(p.trajectory.params.get("freq", 0.0)) for p in scene.linked}
    fmax = max(freqs_per_pid.values()) if freqs_per_pid else 1.0
    fmax = max(fmax, 0.1)
    for pid, path in trajs.items():
        if pid < 0 or pid not in freqs_per_pid: continue
        mask = ~np.isnan(path[:, 0])
        if mask.sum() < 2: continue
        f = freqs_per_pid[pid]
        color = plt.cm.coolwarm(f / fmax)
        ax.plot(path[mask, 0], path[mask, 1], '-', color=color, lw=1.2, alpha=0.85)
    ax.set_title(f"pixel trajectories  (color: blue=slow, red=fast)", fontsize=10)
    ax.axis("off")
plt.suptitle("Hz operating points — slow_only, fast_only, mixed_hz, multiscale_hz", y=1.02)
plt.tight_layout(); plt.show()
""")


# =============================================================================
md(r"""### Per-primitive frequencies in a `mixed_hz` scene

Prints the actual frequency assigned to each linked primitive. Confirms
that within one scene we get both slow (<1 cycle/video) and fast
(>3 cycles/video) primitives.
""")
code(r"""gen = LinkedPrimitivesVideoGenerator(operating_point="mixed_hz", image_size=64, base_seed=0)
scene = gen.sample_scene(seed=0)
print(f"mixed_hz scene  —  {len(scene.linked)} linked primitives")
print(f"{'pid':>4s} {'kind':>12s} {'|freq|':>10s} {'band':>6s}")
for p in scene.linked:
    f = abs(p.trajectory.params.get("freq", 0.0))
    if f == 0.0:    band = "[0]"
    elif f <= 1.0:  band = "[slow]"
    elif f <= 3.0:  band = "[med]"
    else:           band = "[fast]"
    print(f"{p.pid:>4d} {p.trajectory.kind:>12s} {f:>10.3f} {band:>6s}")
""")


# =============================================================================
md(r"""## §11. Downstream B-1 — video-pair classification

**Input:** `(video_A, video_B)`.
**Output:** scene-level label.
**Ground truth:** computed from latents via `gen.compute_label(scene, kind=...)`.

Below we sample three `mixed_hz` scenes and print all available label
kinds for each. Note the new motion-conditioned labels (`has_motion_pattern`,
`n_distinct_motion_kinds`, `has_fast_motion`, `freq_band_count`).
""")
code(r"""gen = LinkedPrimitivesVideoGenerator(operating_point="mixed_hz", image_size=96, base_seed=11)
fig, axes = plt.subplots(3, 4, figsize=(14, 9))
for row, seed in enumerate([1, 7, 42]):
    scene = gen.sample_scene(seed=seed)
    video = gen.render_video_pair(scene)
    T = video["view_A"]["rgb"].shape[0]
    for col, t in enumerate([0, T // 3, 2 * T // 3, T - 1]):
        axes[row, col].imshow(video["view_A"]["rgb"][t]); axes[row, col].axis("off")
        axes[row, col].set_title(f"seed={seed}  τ={t/T:.2f}", fontsize=8)
    ymotion = gen.compute_label(scene, kind="has_motion_pattern")
    yfast   = gen.compute_label(scene, kind="has_fast_motion")
    nbands  = gen.compute_label(scene, kind="freq_band_count", K=5)
    nkinds  = gen.compute_label(scene, kind="n_distinct_motion_kinds", K=5)
    print(f"seed={seed:>3d}  has_motion={ymotion}  has_fast={yfast}  freq_bands={nbands}  motion_kinds={nkinds}")
plt.tight_layout(); plt.show()
""")


# =============================================================================
md(r"""## §12. Downstream B-2 — spatiotemporal segmentation (tubes)

**Input:** video_A (or B).
**Output:** for each (frame, pixel), the primitive ID. Same `pid` across
frames forms a *tube* through space-time.
**Ground truth:** `video["view_A"]["seg"]` — shape `(T, H, W)` int32.

Visualization: show the per-frame seg maps for one view of one scene.
""")
code(r"""gen = LinkedPrimitivesVideoGenerator(operating_point="basic", image_size=96, base_seed=12)
scene = gen.sample_scene(seed=12)
video = gen.render_video_pair(scene)
T = video["view_A"]["rgb"].shape[0]
n_show = min(6, T)
indices = np.linspace(0, T - 1, n_show).astype(int)
fig, axes = plt.subplots(2, n_show, figsize=(2.4 * n_show, 5))
for col, t in enumerate(indices):
    axes[0, col].imshow(video["view_A"]["rgb"][t]); axes[0, col].axis("off")
    axes[0, col].set_title(f"τ={t/T:.2f}", fontsize=9)
    axes[1, col].imshow(video["view_A"]["seg"][t], cmap="tab20", vmin=-1)
    axes[1, col].axis("off")
plt.suptitle("Spatiotemporal segmentation — pid persists across frames (= tube)", y=1.02)
plt.tight_layout(); plt.show()
""")


# =============================================================================
md(r"""## §13. Downstream B-3 — within-view tracking

**Input:** video_A.
**Output:** for each primitive seen at frame 0, its `(x, y)` at every
subsequent frame.
**Ground truth:** `cross_time_pairs_within_view(video, "A", t1, t2)`
returns `(M, 2, 2)` pairs of `(kpt@t1, kpt@t2)`.
""")
code(r"""gen = LinkedPrimitivesVideoGenerator(operating_point="hard", image_size=128, base_seed=5)
scene = gen.sample_scene(seed=5)
video = gen.render_video_pair(scene)
T = video["view_A"]["rgb"].shape[0]
t1, t2 = 0, T - 1
pairs = cross_time_pairs_within_view(video, "A", t1, t2)
fig, axes = plt.subplots(1, 2, figsize=(10, 5))
axes[0].imshow(video["view_A"]["rgb"][t1]); axes[0].set_title(f"view A  τ={t1}"); axes[0].axis("off")
axes[1].imshow(video["view_A"]["rgb"][t2]); axes[1].set_title(f"view A  τ={t2}  ({pairs.shape[0]} tracked)"); axes[1].axis("off")
for k in range(pairs.shape[0]):
    (x1, y1), (x2, y2) = pairs[k]
    con = ConnectionPatch(xyA=(x1, y1), coordsA=axes[0].transData,
                           xyB=(x2, y2), coordsB=axes[1].transData,
                           color=plt.cm.tab20(k % 20), lw=0.8, alpha=0.6)
    fig.add_artist(con)
    axes[0].plot(x1, y1, 'o', mfc='white', mec='black', ms=3, mew=0.3)
    axes[1].plot(x2, y2, 'o', mfc='white', mec='black', ms=3, mew=0.3)
plt.tight_layout(); plt.show()
""")


# =============================================================================
md(r"""## §14. Downstream B-4 — cross-view keypoint retrieval over time

**Input:** per-frame features in both views.
**Output:** for each `(pid, t)` in A, the matching kpt in B at the same
frame.
**Ground truth:** `cross_view_pairs_at_time(video, t_idx)` at every t.
""")
code(r"""gen = LinkedPrimitivesVideoGenerator(operating_point="basic", image_size=128, base_seed=3)
scene = gen.sample_scene(seed=3)
video = gen.render_video_pair(scene)
T = video["view_A"]["rgb"].shape[0]
indices = [0, T // 2, T - 1]
fig, axes = plt.subplots(len(indices), 2, figsize=(10, 5 * len(indices)))
for row, t in enumerate(indices):
    pairs = cross_view_pairs_at_time(video, t)
    axA = axes[row, 0]; axB = axes[row, 1]
    axA.imshow(video["view_A"]["rgb"][t]); axA.set_title(f"view A  τ={t}"); axA.axis("off")
    axB.imshow(video["view_B"]["rgb"][t]); axB.set_title(f"view B  τ={t}  ({pairs.shape[0]} matches)"); axB.axis("off")
    for k in range(pairs.shape[0]):
        (xA, yA), (xB, yB) = pairs[k]
        con = ConnectionPatch(xyA=(xA, yA), coordsA=axA.transData,
                               xyB=(xB, yB), coordsB=axB.transData,
                               color=plt.cm.tab20(k % 20), lw=0.7, alpha=0.6)
        fig.add_artist(con)
plt.tight_layout(); plt.show()
""")


# =============================================================================
md(r"""## §15. Downstream B-5 — cross-view × cross-time matching

**Input:** features at `(view_A, frame_t1)` and `(view_B, frame_t2)`.
**Output:** for each primitive at the first, find it in the second.
**Ground truth:** just intersect pid sets — the same primitive has the
same `pid` everywhere.

This subsumes B-3 (set view_A = view_B) and B-4 (set t1 = t2). It's the
strictest test: the model must be invariant to *both* viewpoint and time
simultaneously.
""")
code(r"""gen = LinkedPrimitivesVideoGenerator(operating_point="basic", image_size=128, base_seed=3)
scene = gen.sample_scene(seed=3)
video = gen.render_video_pair(scene)
T = video["view_A"]["rgb"].shape[0]
t1, t2 = 0, T - 1
A = video["view_A"]; B = video["view_B"]
ids_A = set(int(i) for i, v in zip(A["ids"], A["vis"][t1]) if v)
ids_B = set(int(i) for i, v in zip(B["ids"], B["vis"][t2]) if v)
common = sorted(ids_A & ids_B)

fig, axes = plt.subplots(1, 2, figsize=(10, 5))
axes[0].imshow(A["rgb"][t1]); axes[0].set_title(f"view A  τ={t1}"); axes[0].axis("off")
axes[1].imshow(B["rgb"][t2]); axes[1].set_title(f"view B  τ={t2}  ({len(common)} matches across view+time)"); axes[1].axis("off")
for k, pid in enumerate(common):
    iA = int(np.where(A["ids"] == pid)[0][0])
    iB = int(np.where(B["ids"] == pid)[0][0])
    xA, yA = A["kpts"][t1, iA]; xB, yB = B["kpts"][t2, iB]
    con = ConnectionPatch(xyA=(xA, yA), coordsA=axes[0].transData,
                           xyB=(xB, yB), coordsB=axes[1].transData,
                           color=plt.cm.tab20(k % 20), lw=0.7, alpha=0.6)
    fig.add_artist(con)
plt.tight_layout(); plt.show()
""")


# =============================================================================
md(r"""## §16. Downstream B-6 — dense optical flow within view

**Input:** consecutive frames `(frame_t, frame_{t+1})` of one view.
**Output:** for each foreground pixel in `frame_t`, its motion vector
`Δ(x, y) → frame_{t+1}`.
**Ground truth:** finite-difference of per-pid keypoint trajectories.
Below we visualize the flow as arrows at primitive centers.
""")
code(r"""gen = LinkedPrimitivesVideoGenerator(operating_point="fast_only", image_size=128, base_seed=7)
scene = gen.sample_scene(seed=7)
video = gen.render_video_pair(scene)
T = video["view_A"]["rgb"].shape[0]; t = T // 4
trajs = trajectories_for_view(video, "A")

fig, ax = plt.subplots(figsize=(7, 7))
ax.imshow(video["view_A"]["rgb"][t])
for pid, path in trajs.items():
    if pid < 0 or t >= T - 1: continue
    p1 = path[t]; p2 = path[t + 1]
    if np.isnan(p1[0]) or np.isnan(p2[0]): continue
    dx, dy = float(p2[0] - p1[0]), float(p2[1] - p1[1])
    if dx ** 2 + dy ** 2 < 0.5: continue   # skip near-zero motion
    ax.arrow(p1[0], p1[1], dx, dy,
             color='red', head_width=2.5, head_length=2.5,
             length_includes_head=True, alpha=0.8, lw=1.0)
ax.set_title(f"Optical flow from frame {t} → {t+1}  ({scene.knobs['_name']})")
ax.axis("off")
plt.tight_layout(); plt.show()
""")


# =============================================================================
md(r"""## §17. Downstream B-7 — motion-conditioned classification

**Input:** video pair.
**Output:** label that **depends on motion**, not just appearance:
- `has_motion_pattern` — 1 iff any primitive has sinusoidal / circular trajectory
- `has_fast_motion` — 1 iff any primitive has freq > 2.0 cycles/video
- `n_distinct_motion_kinds` — count of unique trajectory kinds in the scene
- `freq_band_count` — count of distinct frequency bands {static, slow, med, fast}

**Ground truth:** `gen.compute_label(scene, kind=...)`.
""")
code(r"""print(f"\nLabel distributions over 200 scenes per Hz operating point:\n")
print(f"{'op':>15s}  {'has_fast':>10s}  {'#freq_bands [counts per band]':>40s}")
for name in HZ_OPS:
    gen = LinkedPrimitivesVideoGenerator(operating_point=name, image_size=48)
    fast_counts = [0, 0]
    band_counts = [0] * 5
    for s in range(200):
        sc = gen.sample_scene(seed=s)
        fast_counts[gen.compute_label(sc, kind="has_fast_motion")] += 1
        band_counts[gen.compute_label(sc, kind="freq_band_count", K=5)] += 1
    print(f"{name:>15s}  {fast_counts!s:>10s}  {band_counts!s:>40s}")
""")


# =============================================================================
md(r"""## §18. Label distribution sanity check

For all operating points in both datasets, confirm that `count_modulo_4`
(A) and `freq_band_count` (B) are reasonably distributed across classes.
""")
code(r"""N_PER = 200
fig, axes = plt.subplots(2, max(len(BASE_DIFFS), len(HZ_OPS)),
                          figsize=(4 * max(len(BASE_DIFFS), len(HZ_OPS)), 7))
# Row 0: A datasets — count_modulo_4
for col, name in enumerate(BASE_DIFFS):
    gen = LinkedPrimitivesGenerator(operating_point=name, image_size=48, base_seed=0)
    labels = [gen.compute_label(gen.sample_scene(seed=s), kind="count_modulo_K", K=4)
              for s in range(N_PER)]
    axes[0, col].hist(labels, bins=np.arange(5) - 0.5, color='C0', rwidth=0.8)
    axes[0, col].set_title(f"A: {name}\ncount mod 4", fontsize=9)
    axes[0, col].set_xticks(range(4))
for col in range(len(BASE_DIFFS), axes.shape[1]):
    axes[0, col].axis("off")

# Row 1: B datasets — freq_band_count for the Hz ops
for col, name in enumerate(HZ_OPS):
    gen = LinkedPrimitivesVideoGenerator(operating_point=name, image_size=48, base_seed=0)
    labels = [gen.compute_label(gen.sample_scene(seed=s), kind="freq_band_count", K=5)
              for s in range(N_PER)]
    axes[1, col].hist(labels, bins=np.arange(6) - 0.5, color='C3', rwidth=0.8)
    axes[1, col].set_title(f"B: {name}\nfreq_band_count", fontsize=9)
    axes[1, col].set_xticks(range(5))
for col in range(len(HZ_OPS), axes.shape[1]):
    axes[1, col].axis("off")
plt.suptitle("Label balance per operating point", y=1.02)
plt.tight_layout(); plt.show()
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
