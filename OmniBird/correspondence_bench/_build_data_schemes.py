"""Build correspondence_bench/data_schemes.ipynb — a schematic figure of
the stereotypical SAMPLING PATTERNS of five real-world sensor modalities.

The figure shows WHERE each sensor takes samples in the image plane,
abstracted away from any scene content. Each tile is a square frame
with thick black borders containing black dots at the sensor's
characteristic measurement locations:

  1. RGB camera     — dense regular grid (every pixel a sample)
  2. IR thermal     — dense regular grid at coarser resolution
                      (typical IR sensors are ~160x120 vs RGB at ~1080p)
  3. Depth camera   — dense grid with characteristic "depth holes"
                      (random missing pixels from reflective surfaces,
                      out-of-range objects, depth discontinuities)
  4. EBC            — sparse asynchronous events
                      (only triggered on intensity changes,
                      output is unstructured x-y points)
  5. LiDAR (spinning) — horizontal scan rings
                      (a rotating sensor produces N rings of points,
                      each ring a row of horizontal samples)

No scene content; the figure is purely about WHERE each sensor measures.
"""
import json, os
NB = "/Users/davidpark/Documents/Claude/NFJEPA/OmniBird/correspondence_bench/data_schemes.ipynb"

cells = []
def md(s):   cells.append({"cell_type":"markdown","metadata":{},"source":s.splitlines(keepends=True)})
def code(s): cells.append({"cell_type":"code","metadata":{},"execution_count":None,"outputs":[],"source":s.splitlines(keepends=True)})


# =============================================================================
md(r"""# Sensor Data Schemes — sampling patterns of five real-world modalities

A schematic figure intended for slides or papers. **No scene content** —
each tile shows the characteristic SAMPLING PATTERN of one sensor type:
WHERE the sensor takes measurements in the image plane, abstracted from
any image.

| # | Sensor | Pattern in one sentence |
|---|---|---|
| 1 | **RGB camera** | dense regular grid; every pixel is sampled |
| 2 | **IR thermal** | dense regular grid at coarser resolution (typically ~160x120 vs RGB 1080p) |
| 3 | **Depth camera** | dense grid with characteristic "depth holes" — random missing pixels from reflective surfaces and depth discontinuities |
| 4 | **EBC** (event-based camera) | sparse asynchronous events, triggered only by per-pixel intensity changes — output is an unstructured cloud of `(x, y, t, polarity)` |
| 5 | **LiDAR** (rotating) | horizontal scan rings — a rotating sensor produces N rows of points, each at a fixed vertical angle |

Each tile is rendered as black dots on a white background inside a
thick-black-border square. The figure is generated entirely from a few
lines of numpy — no scene generator is involved.
""")


# =============================================================================
md("## §1. Setup")
code(r"""import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
np.random.seed(0)
""")


# =============================================================================
md(r"""## §2. Sampling-pattern generators

Each function returns `(N, 2)` array of `(x, y)` sample locations in the
unit square `[0, 1]^2`. The five sensors:
""")
code(r"""# Dense regular grid — every pixel sampled (RGB camera).
def pattern_rgb_camera(grid=28):
    xs, ys = np.meshgrid(np.linspace(0.05, 0.95, grid),
                          np.linspace(0.05, 0.95, grid))
    return np.stack([xs.ravel(), ys.ravel()], axis=-1)


# Coarser regular grid (IR thermal sensors are typically much lower-res
# than RGB: ~160x120 or 80x60 vs 1080p).
def pattern_ir_thermal(grid=12):
    xs, ys = np.meshgrid(np.linspace(0.05, 0.95, grid),
                          np.linspace(0.05, 0.95, grid))
    return np.stack([xs.ravel(), ys.ravel()], axis=-1)


# Dense grid with characteristic depth holes (Kinect, RealSense, ToF).
# Real depth sensors drop pixels at: reflective/dark surfaces, depth
# discontinuities, out-of-range distances. Structured-light systems also
# show occasional vertical 'occlusion shadows' from the baseline; we add
# a few of those.
def pattern_depth_camera(grid=24, hole_fraction=0.18, rng=None):
    if rng is None: rng = np.random.RandomState(1)
    xs, ys = np.meshgrid(np.linspace(0.05, 0.95, grid),
                          np.linspace(0.05, 0.95, grid))
    pts = np.stack([xs.ravel(), ys.ravel()], axis=-1)
    keep = rng.uniform(size=len(pts)) > hole_fraction
    for stripe_x in rng.uniform(0.2, 0.8, size=2):
        in_stripe = np.abs(pts[:, 0] - stripe_x) < 0.04
        keep &= ~in_stripe
    return pts[keep]


# Sparse async events along noisy curves (DVS / ATIS event cameras fire
# only on intensity-change crossings, so events cluster along moving
# edges). Without scene content we sample around a few wandering curves
# with Gaussian thickness — the visual signature of an event-camera trace.
def pattern_ebc_events(n_events=350, rng=None):
    if rng is None: rng = np.random.RandomState(2)
    pts = []
    n_curves = rng.randint(3, 5)
    for _ in range(n_curves):
        t = np.linspace(0, 1, 80)
        ax = rng.uniform(-0.4, 0.4); bx = rng.uniform(-0.2, 0.2)
        ay = rng.uniform(-0.4, 0.4); by = rng.uniform(-0.2, 0.2)
        cx = rng.uniform(0.2, 0.8); cy = rng.uniform(0.2, 0.8)
        x = cx + ax * np.sin(2 * np.pi * t) + bx * t
        y = cy + ay * np.cos(2 * np.pi * t) + by * t
        n_per_curve = n_events // n_curves
        noise = rng.normal(0, 0.018, size=(n_per_curve, 2))
        idx = rng.choice(len(t), size=n_per_curve)
        pts.append(np.stack([x[idx], y[idx]], axis=-1) + noise)
    out = np.concatenate(pts, axis=0)
    out = np.clip(out, 0.03, 0.97)
    return out


# Horizontal scan lines from a rotating LiDAR (Velodyne-style). Each ring
# is a row of dots at a fixed vertical angle. Real sensors have 16/32/64/
# 128 rings; we use evenly-spaced rings for clarity. The small horizontal
# jitter mimics angular-encoder noise.
def pattern_lidar_scan_lines(n_rings=10, n_per_ring=44, rng=None):
    if rng is None: rng = np.random.RandomState(3)
    pts = []
    ys = np.linspace(0.08, 0.92, n_rings)
    for y in ys:
        x = np.linspace(0.04, 0.96, n_per_ring)
        x = x + rng.normal(0, 0.003, n_per_ring)
        pts.append(np.stack([x, np.full_like(x, y)], axis=-1))
    return np.concatenate(pts, axis=0)
""")


# =============================================================================
md(r"""## §3. The figure

Five tiles in a row. Each is a unit square with a thick black border
containing black sample dots; nothing else. Below each tile: the sensor
name and the sample count.
""")
code(r"""rgb_pts   = pattern_rgb_camera(grid=28)
ir_pts    = pattern_ir_thermal(grid=12)
depth_pts = pattern_depth_camera(grid=24, hole_fraction=0.18)
ebc_pts   = pattern_ebc_events(n_events=350)
lidar_pts = pattern_lidar_scan_lines(n_rings=10, n_per_ring=44)

TILES = [
    ("RGB camera",   "dense regular grid",            rgb_pts,   1.6),
    ("IR thermal",   "coarser dense grid",            ir_pts,    8.0),
    ("Depth camera", "dense + depth holes",           depth_pts, 2.5),
    ("EBC",          "sparse async events",           ebc_pts,   4.5),
    ("LiDAR",        "horizontal scan rings",         lidar_pts, 3.5),
]

fig, axes = plt.subplots(1, len(TILES), figsize=(3.6 * len(TILES), 4.4))
for ax, (name, descr, pts, marker_size) in zip(axes, TILES):
    ax.scatter(pts[:, 0], pts[:, 1], s=marker_size,
                c='black', marker='o', alpha=0.95, linewidths=0)
    ax.set_xlim(0.0, 1.0); ax.set_ylim(0.0, 1.0)
    ax.set_aspect('equal')
    ax.invert_yaxis()       # match image convention (origin top-left)
    ax.set_xticks([]); ax.set_yticks([])
    for side in ("top", "right", "bottom", "left"):
        ax.spines[side].set_visible(True)
        ax.spines[side].set_linewidth(3.5)
        ax.spines[side].set_edgecolor('black')
    ax.set_title(name, fontsize=15, weight='bold', pad=12)
    ax.set_xlabel(f"{descr}\nN = {len(pts):,} samples", fontsize=10, labelpad=10)

plt.suptitle("Stereotypical sensor sampling patterns", fontsize=16, y=1.02)
plt.tight_layout()
plt.savefig("sensor_sampling_patterns.png", dpi=150, bbox_inches="tight")
plt.show()
print("Saved → sensor_sampling_patterns.png")
""")


# =============================================================================
md(r"""## §4. Reading the figure

Each tile encodes one fact about its sensor:

- **RGB camera** — uniform dense grid. Maximum spatial coverage. The
  reference point: every other modality is some kind of subsampling,
  redistribution, or alternative measurement of this base layout.
- **IR thermal** — same uniform layout, but at a much coarser
  resolution. Thermal sensors trade resolution for the ability to
  measure long-wavelength emission, so spatial density drops.
- **Depth camera** — almost as dense as RGB, but with characteristic
  per-pixel dropouts ("depth holes") from reflective surfaces, depth
  discontinuities, out-of-range objects, and (for structured-light /
  stereo systems) occlusion shadows visible as the vertical stripes.
- **EBC** — total break from the grid. Events are unstructured points
  triggered only by per-pixel intensity changes. No regular layout, no
  guaranteed coverage of static regions. What you see here are events
  clustered along moving edges — the only place a DVS pixel fires.
- **LiDAR** (rotating) — points are organized in horizontal rings, not
  a grid. Each ring is one rotation of the spinning sensor at a fixed
  vertical angle. Sample density is high horizontally within each
  ring, low vertically between rings — exactly the opposite of a
  camera's near-isotropic pixel grid.

The benchmark's `multiview_primitives.py` simulates three of these
(LiDAR, Infrared, Depth) as the three views of the sensor-fusion
variant; the SSL recipe has to learn the common latent scene despite
these very different sampling patterns.
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
