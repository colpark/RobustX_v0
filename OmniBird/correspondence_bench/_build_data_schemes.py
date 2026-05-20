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
md(r"""## §2. Sampling-pattern generators with sensor-specific FOV

Each function returns `(N, 2)` array of `(x, y)` sample locations
**inside a sensor-specific field of view** within the unit square
`[0, 1]^2`. The FOV is itself a signature:

| Sensor | FOV aspect | FOV area | Real-world analog |
|---|---|---|---|
| RGB camera | wide 16:9 | ~85% of frame | typical 60-90° HFOV consumer camera |
| IR thermal | narrow ~4:3 | ~50% of frame | LWIR cameras are commonly telephoto (~30-50° HFOV) |
| Depth camera | standard ~4:3 | ~65% of frame | Kinect / RealSense (~60° HFOV) |
| EBC | wide 16:9 | ~85% of frame | DVS uses the same optics as the RGB sensor it shares |
| LiDAR (rotating) | panoramic ~6:1 | ~30% of frame | 360° horizontal × ~30-40° vertical → a wide thin strip |

The FOV outlines are drawn as dashed gray rectangles so the SHAPE of
each sensor's coverage is visible alongside its sampling density.
""")
code(r"""# ---- FOV bounding boxes per sensor ----
# All are (x_lo, y_lo, x_hi, y_hi) inside the unit square. The FOV
# captures sensor optics: angular extent and aspect ratio.
FOV_RGB    = (0.06, 0.22, 0.94, 0.78)   # 16:9 wide screen, ~85% of frame
FOV_IR     = (0.22, 0.32, 0.78, 0.68)   # narrow 1.5:1, ~50% of frame
FOV_DEPTH  = (0.10, 0.20, 0.90, 0.80)   # standard ~4:3, ~65% of frame
FOV_EBC    = (0.06, 0.22, 0.94, 0.78)   # same as RGB (DVS shares optics)
FOV_LIDAR  = (0.025, 0.42, 0.975, 0.58) # ~6:1 panoramic strip, ~30% of frame


# Dense regular grid sampling inside FOV (RGB camera).
def pattern_rgb_camera(fov=FOV_RGB, grid_x=32, grid_y=18):
    x_lo, y_lo, x_hi, y_hi = fov
    xs, ys = np.meshgrid(np.linspace(x_lo, x_hi, grid_x),
                          np.linspace(y_lo, y_hi, grid_y))
    return np.stack([xs.ravel(), ys.ravel()], axis=-1)


# Coarser regular grid inside narrow FOV (IR thermal).
def pattern_ir_thermal(fov=FOV_IR, grid_x=14, grid_y=10):
    x_lo, y_lo, x_hi, y_hi = fov
    xs, ys = np.meshgrid(np.linspace(x_lo, x_hi, grid_x),
                          np.linspace(y_lo, y_hi, grid_y))
    return np.stack([xs.ravel(), ys.ravel()], axis=-1)


# Dense grid with characteristic depth holes inside standard FOV.
# Real depth sensors (Kinect / RealSense / ToF) drop pixels at:
# reflective surfaces, depth discontinuities, and out-of-range distances.
# Structured-light systems also show vertical 'occlusion shadows' from
# the baseline; we add a couple of those.
def pattern_depth_camera(fov=FOV_DEPTH, grid_x=24, grid_y=18,
                          hole_fraction=0.18, rng=None):
    if rng is None: rng = np.random.RandomState(1)
    x_lo, y_lo, x_hi, y_hi = fov
    xs, ys = np.meshgrid(np.linspace(x_lo, x_hi, grid_x),
                          np.linspace(y_lo, y_hi, grid_y))
    pts = np.stack([xs.ravel(), ys.ravel()], axis=-1)
    keep = rng.uniform(size=len(pts)) > hole_fraction
    for stripe_x in rng.uniform(x_lo + 0.1, x_hi - 0.1, size=2):
        in_stripe = np.abs(pts[:, 0] - stripe_x) < 0.025
        keep &= ~in_stripe
    return pts[keep]


# Sparse async events along noisy curves inside FOV (DVS / ATIS).
# Real event cameras fire only on per-pixel intensity changes, so
# events cluster along moving edges.
def pattern_ebc_events(fov=FOV_EBC, n_events=320, rng=None):
    if rng is None: rng = np.random.RandomState(2)
    x_lo, y_lo, x_hi, y_hi = fov
    w = x_hi - x_lo; h = y_hi - y_lo
    pts = []
    n_curves = rng.randint(3, 5)
    for _ in range(n_curves):
        t = np.linspace(0, 1, 80)
        ax = rng.uniform(-0.35, 0.35) * w; bx = rng.uniform(-0.2, 0.2) * w
        ay = rng.uniform(-0.35, 0.35) * h; by = rng.uniform(-0.2, 0.2) * h
        cx = rng.uniform(x_lo + 0.15 * w, x_hi - 0.15 * w)
        cy = rng.uniform(y_lo + 0.15 * h, y_hi - 0.15 * h)
        x = cx + ax * np.sin(2 * np.pi * t) + bx * t
        y = cy + ay * np.cos(2 * np.pi * t) + by * t
        n_per_curve = n_events // n_curves
        noise = rng.normal(0, 0.014, size=(n_per_curve, 2))
        idx = rng.choice(len(t), size=n_per_curve)
        pts.append(np.stack([x[idx], y[idx]], axis=-1) + noise)
    out = np.concatenate(pts, axis=0)
    # Clip to FOV
    out[:, 0] = np.clip(out[:, 0], x_lo + 0.005, x_hi - 0.005)
    out[:, 1] = np.clip(out[:, 1], y_lo + 0.005, y_hi - 0.005)
    return out


# Horizontal scan lines inside the LiDAR's panoramic strip FOV.
# Velodyne-style: many horizontal rings packed into a narrow vertical
# range (the sensor scans 360° horizontally and ±15° vertically).
def pattern_lidar_scan_lines(fov=FOV_LIDAR, n_rings=8, n_per_ring=64,
                              rng=None):
    if rng is None: rng = np.random.RandomState(3)
    x_lo, y_lo, x_hi, y_hi = fov
    pts = []
    ys = np.linspace(y_lo + 0.005, y_hi - 0.005, n_rings)
    for y in ys:
        x = np.linspace(x_lo + 0.005, x_hi - 0.005, n_per_ring)
        x = x + rng.normal(0, 0.002, n_per_ring)
        pts.append(np.stack([x, np.full_like(x, y)], axis=-1))
    return np.concatenate(pts, axis=0)
""")


# =============================================================================
md(r"""## §3. The figure

Five tiles in a row. Each is a unit square with a thick black border
containing black sample dots; nothing else. Below each tile: the sensor
name and the sample count.
""")
code(r"""rgb_pts   = pattern_rgb_camera()
ir_pts    = pattern_ir_thermal()
depth_pts = pattern_depth_camera()
ebc_pts   = pattern_ebc_events()
lidar_pts = pattern_lidar_scan_lines()

TILES = [
    ("RGB camera",   "dense regular grid",     rgb_pts,   FOV_RGB,   1.4),
    ("IR thermal",   "coarse dense grid",      ir_pts,    FOV_IR,    8.5),
    ("Depth camera", "dense + depth holes",    depth_pts, FOV_DEPTH, 2.6),
    ("EBC",          "sparse async events",    ebc_pts,   FOV_EBC,   4.5),
    ("LiDAR",        "horizontal scan rings",  lidar_pts, FOV_LIDAR, 2.5),
]

fig, axes = plt.subplots(1, len(TILES), figsize=(3.6 * len(TILES), 4.4))
for ax, (name, descr, pts, fov, marker_size) in zip(axes, TILES):
    # 1) Light grey fill behind the FOV to make its SHAPE pop visually
    x_lo, y_lo, x_hi, y_hi = fov
    ax.add_patch(Rectangle((x_lo, y_lo), x_hi - x_lo, y_hi - y_lo,
                            facecolor='#eeeeee', edgecolor='none', zorder=0))
    # 2) Dashed grey outline of the FOV
    ax.add_patch(Rectangle((x_lo, y_lo), x_hi - x_lo, y_hi - y_lo,
                            facecolor='none', edgecolor='#666666',
                            linestyle='--', linewidth=1.4, zorder=2))
    # 3) Black sample dots inside the FOV
    ax.scatter(pts[:, 0], pts[:, 1], s=marker_size,
                c='black', marker='o', alpha=0.95, linewidths=0, zorder=3)
    ax.set_xlim(0.0, 1.0); ax.set_ylim(0.0, 1.0)
    ax.set_aspect('equal')
    ax.invert_yaxis()       # match image convention (origin top-left)
    ax.set_xticks([]); ax.set_yticks([])
    for side in ("top", "right", "bottom", "left"):
        ax.spines[side].set_visible(True)
        ax.spines[side].set_linewidth(3.5)
        ax.spines[side].set_edgecolor('black')
    # FOV aspect for the subtitle
    fov_w = x_hi - x_lo; fov_h = y_hi - y_lo
    aspect_str = f"FOV {fov_w / fov_h:.1f}:1"
    ax.set_title(name, fontsize=15, weight='bold', pad=12)
    ax.set_xlabel(f"{descr}\n{aspect_str}   |   N = {len(pts):,}",
                   fontsize=10, labelpad=10)

plt.suptitle("Stereotypical sensor sampling patterns and fields of view",
              fontsize=16, y=1.02)
plt.tight_layout()
plt.savefig("sensor_sampling_patterns.png", dpi=150, bbox_inches="tight")
plt.show()
print("Saved → sensor_sampling_patterns.png")
""")


# =============================================================================
md(r"""## §4. Reading the figure

Two signatures per tile — **FOV shape** (the dashed grey rectangle)
and **sampling density inside it** (the black dots):

- **RGB camera** — 16:9 wide FOV, uniform dense grid inside it.
  Maximum spatial coverage; the reference point. Every other modality
  is either a different FOV shape, a sub-sampling pattern, or both.
- **IR thermal** — markedly **narrower FOV** (LWIR cameras are
  commonly telephoto: ~30-50° HFOV vs 60-90° for consumer RGB), and
  also lower spatial density inside that FOV. Two compounding losses.
- **Depth camera** — FOV similar to RGB, but with characteristic
  dropouts inside it: random depth holes plus a couple of vertical
  "occlusion shadows" typical of structured-light systems with a
  non-zero baseline.
- **EBC** — same wide FOV as the RGB sensor it shares optics with,
  but no regular sampling: events fire only where per-pixel intensity
  changes, so the dots cluster along (moving) edges. Most of the FOV
  has no data at any given timestep.
- **LiDAR** (rotating) — the most distinctive FOV: a **panoramic
  strip** (~360° horizontal × ±15° vertical = roughly 6:1 aspect)
  containing horizontal scan rings. The FOV shape itself signals
  "rotating sensor"; the within-FOV layout signals "many horizontal
  samples per ring, few rings vertically".

The benchmark's `multiview_primitives.py` simulates LiDAR, Infrared,
and Depth as three different views of the sensor-fusion variant; the
FOV mismatches shown here are part of why **no single modality is
sufficient** to recover the latent scene.
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
