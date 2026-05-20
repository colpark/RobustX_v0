"""Build correspondence_bench/data_schemes.ipynb — a single-figure overview
of the five sensor modalities the benchmark supports, rendered from the
same underlying scene so the diversity of sensor signatures is visible
side-by-side.

The five modalities, in order:
    1. Dense RGB          — standard wide-FOV camera
    2. Infrared           — temperature heatmap (inferno)
    3. Depth camera       — viridis colormap on per-pixel depth
    4. EBC (event camera) — pixel polarity from frame-to-frame diff
                            (red = positive event, blue = negative)
    5. LiDAR              — sparse depth returns on dark background

Each tile gets a thick black border to read as a schematic / figure-grade
illustration.
"""
import json, os
NB = "/Users/davidpark/Documents/Claude/NFJEPA/OmniBird/correspondence_bench/data_schemes.ipynb"

cells = []
def md(s):   cells.append({"cell_type":"markdown","metadata":{},"source":s.splitlines(keepends=True)})
def code(s): cells.append({"cell_type":"code","metadata":{},"execution_count":None,"outputs":[],"source":s.splitlines(keepends=True)})


# =============================================================================
md(r"""# Data Schemes — five sensor modalities, one scene

A single-row schematic showing the five sensor modalities supported by
`correspondence_bench`, each rendered from the **same underlying scene**.
The figure exists to make the diversity of sensor signatures concrete
in one image — what each kind of data actually LOOKS LIKE.

| # | Modality | Density | Signal carried per pixel | Where it comes from |
|---|---|---|---|---|
| 1 | **Dense RGB**           | dense   | colour + intensity                | `LinkedPrimitivesGenerator` (standard render) |
| 2 | **Infrared**            | dense   | pseudo-temperature → inferno cmap | `InfraredModality` |
| 3 | **Depth camera**        | dense   | z-distance → viridis cmap         | `DepthCameraModality` |
| 4 | **EBC** (event camera)  | **sparse** | binary polarity per pixel (per Δt) | simulated here from a frame-to-frame diff of `LinkedPrimitivesVideoGenerator` |
| 5 | **LiDAR**               | **sparse** | depth at a random ~15% of returns | `LiDARModality` |

The five tiles are illustrated in a single row with thick black borders
— intended as a slide / paper figure.
""")


# =============================================================================
md("## §1. Setup")
code(r"""import os, sys, math
sys.path.insert(0, os.path.abspath('.'))
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

from linked_primitives import (
    LinkedPrimitivesGenerator, _project, FOCAL_DEFAULT, CAMERA_BACK_DEFAULT,
)
from linked_primitives_video import LinkedPrimitivesVideoGenerator
from modalities import InfraredModality, DepthCameraModality, LiDARModality

np.random.seed(0)
""")


# =============================================================================
md(r"""## §2. Helpers

Two small helpers:

* `enrich_with_depth_meta` — adds the `depth` and `primitives_meta` fields
  to a static render so the modality classes (which expect them) can run.
* `simulate_ebc` — generates an event-camera image from two RGB frames:
  red for positive polarity (intensity went up), blue for negative,
  black where no event fired.
""")
code(r"""# enrich_with_depth_meta — adds depth + primitives_meta to a static render so
# the modality classes (which need both) can be invoked on it.
def enrich_with_depth_meta(scene, render_out, view="A"):
    view_mat = scene.view_A if view == "A" else scene.view_B
    prims = scene.linked + (scene.distractors_A if view == "A" else scene.distractors_B)
    depth = np.full(render_out["seg"].shape, np.nan, dtype=np.float32)
    for p in prims:
        screen, z = _project(p.pos_3d, view_mat,
                              focal=FOCAL_DEFAULT, camera_back=CAMERA_BACK_DEFAULT)
        if screen is None: continue
        depth[render_out["seg"] == p.pid] = z
    out = {**render_out, "depth": depth,
           "primitives_meta": {p.pid: (p.shape_id, p.color_idx) for p in prims}}
    return out


# simulate_ebc — event-based-camera output from two RGB frames.
#   positive polarity (intensity increased > threshold) → red
#   negative polarity (intensity decreased > threshold) → blue
#   otherwise → dark background.
# Visualizes the standard accumulated-events-over-Δt representation.
def simulate_ebc(rgb_a, rgb_b, threshold=0.04,
                  background=(8, 8, 12),
                  pos_color=(245, 60, 60),
                  neg_color=(60, 120, 245)):
    ga = rgb_a.astype(np.float32).mean(-1) / 255.0
    gb = rgb_b.astype(np.float32).mean(-1) / 255.0
    diff = gb - ga
    pos = diff >  threshold
    neg = diff < -threshold
    H, W = ga.shape
    out = np.zeros((H, W, 3), dtype=np.uint8)
    out[..., 0] = background[0]; out[..., 1] = background[1]; out[..., 2] = background[2]
    out[pos] = pos_color
    out[neg] = neg_color
    return out
""")


# =============================================================================
md(r"""## §3. Generate one scene → render five sensors

We sample one scene with `LinkedPrimitivesVideoGenerator` (so we have
the two frames needed for the EBC simulation). The first frame is used
for all the dense / sparse modalities; the EBC tile uses the diff
between frames 0 and 4.
""")
code(r"""IMG = 144
gen_video = LinkedPrimitivesVideoGenerator(operating_point="basic",
                                            image_size=IMG, base_seed=7)
scene_v = gen_video.sample_scene(seed=7)
video = gen_video.render_video_pair(scene_v)

# For modalities that need the static-renderer plumbing (depth, IR, LiDAR),
# build a per-frame render dict and attach depth+meta. We reuse the video
# generator's first-frame output and synthesise depth from its seg.
def video_frame_with_depth(scene_v, video, view, frame_idx):
    out = {
        "rgb":  video[f"view_{view}"]["rgb"][frame_idx],
        "seg":  video[f"view_{view}"]["seg"][frame_idx],
        "kpts": video[f"view_{view}"]["kpts"][frame_idx],
        "vis":  video[f"view_{view}"]["vis"][frame_idx],
        "ids":  video[f"view_{view}"]["ids"],
    }
    view_mat = scene_v.view_A if view == "A" else scene_v.view_B
    prims = scene_v.linked + (scene_v.distractors_A if view == "A" else scene_v.distractors_B)
    depth = np.full(out["seg"].shape, np.nan, dtype=np.float32)
    for p in prims:
        # For animated primitives, use the position at the right τ
        tau = float(frame_idx / max(video["view_A"]["rgb"].shape[0] - 1, 1))
        pos_3d = p.trajectory.pos_at(tau)
        screen, z = _project(pos_3d, view_mat,
                              focal=FOCAL_DEFAULT, camera_back=CAMERA_BACK_DEFAULT)
        if screen is None: continue
        depth[out["seg"] == p.pid] = z
    out["depth"] = depth
    out["primitives_meta"] = {p.pid: (p.shape_id, p.color_idx) for p in prims}
    return out


# Build one enriched render at frame 0 (used by IR / depth / LiDAR tiles)
enriched = video_frame_with_depth(scene_v, video, view="A", frame_idx=0)

# Sensor-specific outputs
dense_rgb = enriched["rgb"]                                 # 1. Dense RGB
ir_out    = InfraredModality()(enriched)["rgb"]              # 2. Infrared
depth_out = DepthCameraModality()(enriched)["rgb"]           # 3. Depth
ebc_out   = simulate_ebc(                                    # 4. EBC
    video["view_A"]["rgb"][0],
    video["view_A"]["rgb"][4],
)
lidar_out = LiDARModality(keep_fraction=0.18)(enriched, rng=7)["rgb"]   # 5. LiDAR

print(f"All five sensor outputs ready, image size = {IMG}x{IMG}.")
""")


# =============================================================================
md(r"""## §4. The figure

A single row, five tiles, thick black borders. Each tile shows the same
underlying scene through a different sensor. The point: the same world
produces five very different observations — and an SSL recipe for
sensor fusion has to find the latent scene shared across all of them.
""")
code(r"""TILES = [
    ("Dense RGB",     dense_rgb,  "wide-FOV camera, color + intensity"),
    ("Infrared",      ir_out,     "temperature → inferno colormap"),
    ("Depth",         depth_out,  "z-distance → viridis colormap"),
    ("EBC",           ebc_out,    "polarity events  (red = +, blue = −)"),
    ("LiDAR",         lidar_out,  "sparse depth returns (~15%)"),
]

fig, axes = plt.subplots(1, len(TILES), figsize=(4.0 * len(TILES), 4.6))
for ax, (name, img, descr) in zip(axes, TILES):
    ax.imshow(img)
    ax.set_xticks([]); ax.set_yticks([])
    # Thick black border via axes spines
    for side in ("top", "right", "bottom", "left"):
        ax.spines[side].set_visible(True)
        ax.spines[side].set_linewidth(3.5)
        ax.spines[side].set_edgecolor("black")
    ax.set_title(name, fontsize=14, weight="bold", pad=10)
    ax.set_xlabel(descr, fontsize=10, labelpad=8)

plt.suptitle("Five sensor modalities, one underlying scene", fontsize=15, y=1.02)
plt.tight_layout()
plt.savefig("data_schemes.png", dpi=150, bbox_inches="tight")
plt.show()
print("Saved → data_schemes.png")
""")


# =============================================================================
md(r"""## §5. Reading the figure

What each tile reveals:

* **Dense RGB** — colours and shapes are visible; standard vision sensor.
  Carries the most per-pixel information (colour + intensity) but loses
  depth and material identity.
* **Infrared** — colour is replaced by a temperature heatmap. Different
  primitives may share the same temperature, so IR alone cannot always
  tell them apart. Carries weak material identity, no geometric depth.
* **Depth camera** — colour and material are gone; every pixel encodes
  z-distance from the sensor. Carries strong geometric information and
  zero appearance information.
* **EBC (event-based camera)** — black almost everywhere; only PIXELS
  WHERE INTENSITY CHANGED between two timesteps carry an event. Per
  pixel, 1 bit of polarity (+ or −). Carries motion + edges but no
  static appearance or absolute intensity.
* **LiDAR** — sparse depth points. Per-point depth, no colour, very
  sparse coverage. Carries the same kind of info as depth-camera but
  at a sparser sampling rate.

Sensor fusion is about reconciling these into a single latent scene
representation. The benchmark is engineered so that **no single
modality** carries enough information to solve the downstream task on
its own.
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
