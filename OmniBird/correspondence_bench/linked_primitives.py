"""Linked Primitives — a synthetic multimodal dataset with ground-truth
fine-grained cross-modal correspondences.

GENERATIVE MODEL
----------------
Each scene is a set of N geometric primitives placed in 3D space:

    primitive_i = (shape_i, color_i, size_i, pos_3d_i)
    shape_i ∈ {circle, square, triangle, plus, star, ...}
    color_i ∈ {discrete palette}
    pos_3d_i ∈ [-1, 1]^3
    size_i ∈ [s_min, s_max]

The scene is rendered from two camera viewpoints (view_A, view_B). The
two rendered images form the two modalities. Each primitive carries a
**stable integer ID** that's known in BOTH renderings — this is the
ground-truth correspondence. A model that solves the dataset must
recover, for each primitive in view A, its matching projection in view B.

Optional difficulty-amplifying extras:
  - DISTRACTORS — primitives that appear only in one modality (look real,
    have no cross-modal match). The model must learn to ignore them.
  - STYLE GAP — view B can be rendered in a different style (grayscale,
    edge-only, hue-shifted, ...).
  - SCALE VARIANCE — primitives' physical sizes can span many orders of
    magnitude in the same scene, testing scale-equivariance.
  - OCCLUSION — primitives in front occlude those behind them; only the
    visible portion is in the segmentation mask.

LABEL
-----
The scene label φ(z) is computed from the LATENT scene description, not
the renders, so it requires cross-modal correspondence learning to
recover from observations alone. Built-in label functions:

    "count_modulo_K"   — number of linked primitives mod K
    "has_pair"         — is there a pair (i, j) of linked primitives that
                          satisfies a relation (same shape, different color)?
    "n_distinct_pairs" — count of distinct (shape, color) tuples

OUTPUT
------
For each rendered view we return:
    rgb  : (H, W, 3) uint8  — the image
    seg  : (H, W) int32     — per-pixel primitive ID; -1 for background
    kpts : (N, 2) float32   — projected 2D centers of all primitives
                              (NaN for behind-camera / off-screen)
    vis  : (N,) bool        — whether each primitive is visible in this view

Both views share the same primitive indexing, so seg_A == id <-> seg_B == id
gives the correspondence ground truth directly.
"""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Optional, Union
import math

import numpy as np
from PIL import Image, ImageDraw


# ===========================================================================
# Difficulty operating points
# ===========================================================================

OPERATING_POINTS = {
    # Each is a dict of knobs; sample_scene reads from here.
    "easy": dict(
        n_linked=4, n_shapes=2, n_colors=3,
        view_disparity_deg=30.0,
        n_distractors_A=0, n_distractors_B=0,
        scale_range=(0.10, 0.15),
        style="rgb",
        noise_sigma=0.0,
        adversarial_confusables=False,
    ),
    "basic": dict(
        n_linked=16, n_shapes=4, n_colors=5,
        view_disparity_deg=60.0,
        n_distractors_A=2, n_distractors_B=2,
        scale_range=(0.06, 0.16),
        style="rgb",
        noise_sigma=0.02,
        adversarial_confusables=False,
    ),
    "hard": dict(
        n_linked=64, n_shapes=8, n_colors=8,
        view_disparity_deg=120.0,
        n_distractors_A=16, n_distractors_B=16,
        scale_range=(0.03, 0.20),
        style="grayscale_B",
        noise_sigma=0.05,
        adversarial_confusables=False,
    ),
    "extreme": dict(
        n_linked=128, n_shapes=10, n_colors=10,
        view_disparity_deg=170.0,
        n_distractors_A=48, n_distractors_B=48,
        scale_range=(0.02, 0.25),
        style="edges_B",
        noise_sigma=0.08,
        adversarial_confusables=False,
    ),
    "adversarial": dict(
        n_linked=128, n_shapes=10, n_colors=10,
        view_disparity_deg=170.0,
        n_distractors_A=64, n_distractors_B=64,
        scale_range=(0.02, 0.25),
        style="edges_B",
        noise_sigma=0.10,
        adversarial_confusables=True,   # near-duplicate primitives clustered
    ),
}

SHAPES = ["circle", "square", "triangle", "plus", "star",
          "diamond", "hexagon", "cross", "pentagon", "octagon"]

DEFAULT_PALETTE = np.array([
    [0.85, 0.20, 0.20],   # red
    [0.20, 0.65, 0.30],   # green
    [0.20, 0.40, 0.85],   # blue
    [0.90, 0.80, 0.15],   # yellow
    [0.75, 0.25, 0.75],   # magenta
    [0.20, 0.75, 0.75],   # cyan
    [0.95, 0.55, 0.10],   # orange
    [0.45, 0.30, 0.70],   # purple
    [0.50, 0.50, 0.50],   # gray
    [0.10, 0.10, 0.10],   # near-black
], dtype=np.float32)


# ===========================================================================
# Scene representation
# ===========================================================================

@dataclass
class Primitive:
    shape_id: int                  # index into SHAPES
    color_idx: int                 # index into the palette in use
    color: np.ndarray              # (3,) RGB in [0, 1] (resolved from palette)
    pos_3d: np.ndarray             # (3,) in [-1, 1]
    size: float                    # half-extent in world units
    pid: int = -1                  # globally unique primitive ID (set by generator)


@dataclass
class Scene:
    linked: list[Primitive]                  # primitives that appear in BOTH views
    distractors_A: list[Primitive]           # only rendered in view A
    distractors_B: list[Primitive]           # only rendered in view B
    view_A: np.ndarray = field(default_factory=lambda: np.eye(3))   # (3, 3) rotation
    view_B: np.ndarray = field(default_factory=lambda: np.eye(3))
    style_A: str = "rgb"
    style_B: str = "rgb"
    noise_sigma: float = 0.0
    # Metadata
    knobs: dict = field(default_factory=dict)
    seed: int = 0


# ===========================================================================
# Geometry
# ===========================================================================

def _rotation_y(angle_rad: float) -> np.ndarray:
    c, s = math.cos(angle_rad), math.sin(angle_rad)
    return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]], dtype=np.float32)


def _rotation_x(angle_rad: float) -> np.ndarray:
    c, s = math.cos(angle_rad), math.sin(angle_rad)
    return np.array([[1, 0, 0], [0, c, -s], [0, s, c]], dtype=np.float32)


def _project(pos_3d: np.ndarray, view: np.ndarray,
             focal: float = 1.6, camera_back: float = 3.0):
    """Pinhole projection. Returns ((sx, sy), z) where sx, sy ∈ [-1, 1]-ish
    and z is the depth from the camera; or (None, None) if behind the camera."""
    pt = view @ pos_3d
    z = pt[2] + camera_back
    if z <= 0.15:
        return None, None
    sx = focal * pt[0] / z
    sy = focal * pt[1] / z
    return (float(sx), float(sy)), float(z)


# ===========================================================================
# Generator
# ===========================================================================

class LinkedPrimitivesGenerator:
    """Sample scenes + render them at a chosen difficulty operating point."""

    def __init__(self,
                 operating_point: Union[str, dict] = "basic",
                 image_size: int = 128,
                 background: tuple = (245, 245, 245),
                 palette: np.ndarray = DEFAULT_PALETTE,
                 base_seed: int = 0):
        if isinstance(operating_point, str):
            self.knobs = dict(OPERATING_POINTS[operating_point])
            self.knobs["_name"] = operating_point
        else:
            self.knobs = dict(operating_point)
            self.knobs.setdefault("_name", "custom")
        self.image_size = image_size
        self.background = background
        self.palette = palette
        self.base_seed = base_seed

    # ---- scene sampling -------------------------------------------------

    def sample_scene(self, seed: Optional[int] = None) -> Scene:
        rng = np.random.RandomState(seed if seed is not None else self.base_seed)
        k = self.knobs

        # Pre-pick the subsets of shapes / colors / scale range we'll use
        shape_ids_in_use = list(range(min(k["n_shapes"], len(SHAPES))))
        color_ids_in_use = list(range(min(k["n_colors"], len(self.palette))))

        def random_primitive() -> Primitive:
            shape_id = int(rng.choice(shape_ids_in_use))
            color_idx = int(rng.choice(color_ids_in_use))
            color = self.palette[color_idx]
            # 3D position in the unit cube, but biased away from the camera axis
            pos = rng.uniform(-0.9, 0.9, size=3).astype(np.float32)
            # log-uniform scale within the configured range
            s_lo, s_hi = k["scale_range"]
            size = float(math.exp(rng.uniform(math.log(s_lo), math.log(s_hi))))
            return Primitive(shape_id=shape_id, color_idx=color_idx, color=color,
                              pos_3d=pos, size=size)

        # Sample N linked primitives.
        linked = [random_primitive() for _ in range(k["n_linked"])]

        # Optionally cluster adversarial near-duplicates near each anchor.
        if k.get("adversarial_confusables", False) and len(linked) > 0:
            n_extra = min(len(linked) // 2, 16)
            for _ in range(n_extra):
                src = linked[rng.randint(len(linked))]
                p = Primitive(
                    shape_id=src.shape_id,                            # same shape
                    color_idx=src.color_idx,                          # same color
                    color=src.color.copy(),
                    pos_3d=src.pos_3d + rng.uniform(-0.15, 0.15, 3).astype(np.float32),
                    size=src.size * float(rng.uniform(0.7, 1.3)),
                )
                linked.append(p)

        # Assign globally unique IDs to linked primitives (0 .. n_linked-1)
        for i, p in enumerate(linked):
            p.pid = i

        # Sample distractors per modality (use distinct ID ranges)
        n_link = len(linked)
        distractors_A = [random_primitive() for _ in range(k["n_distractors_A"])]
        for j, p in enumerate(distractors_A):
            p.pid = n_link + j      # IDs immediately after linked
        distractors_B = [random_primitive() for _ in range(k["n_distractors_B"])]
        for j, p in enumerate(distractors_B):
            p.pid = n_link + len(distractors_A) + j

        # Two viewpoints — rotate about y by ±view_disparity/2, small x tilt.
        delta = math.radians(k["view_disparity_deg"]) * 0.5
        tilt = math.radians(float(rng.uniform(-15, 15)))
        view_A = _rotation_x(tilt) @ _rotation_y(-delta)
        view_B = _rotation_x(tilt) @ _rotation_y(+delta)

        style_B = k.get("style", "rgb")
        # Normalize style spec into per-view tags
        style_A = "rgb"
        if style_B == "rgb":          style_B = "rgb"
        elif style_B == "grayscale_B": style_B = "grayscale"
        elif style_B == "edges_B":     style_B = "edges"

        return Scene(
            linked=linked,
            distractors_A=distractors_A,
            distractors_B=distractors_B,
            view_A=view_A, view_B=view_B,
            style_A=style_A, style_B=style_B,
            noise_sigma=k.get("noise_sigma", 0.0),
            knobs=dict(self.knobs),
            seed=int(seed if seed is not None else self.base_seed),
        )

    # ---- rendering ------------------------------------------------------

    def render(self, scene: Scene, view: str = "A") -> dict:
        """Render one modality of a scene.

        Returns a dict with:
          rgb:  (H, W, 3) uint8 image
          seg:  (H, W)   int32 — primitive ID per pixel, -1 for background
          kpts: (N_total, 2) float32 — 2D projected centers (NaN if culled)
          vis:  (N_total,) bool — visibility flag per primitive
          ids:  (N_total,) int32 — primitive IDs, in canonical order

        N_total = len(linked) + len(distractors_for_this_view).
        Linked primitives are first; distractors follow. Order is identical
        in both views (only the distractor SET differs).
        """
        H = W = self.image_size
        rgb_img = Image.new("RGB", (W, H), self.background)
        seg_arr = np.full((H, W), -1, dtype=np.int32)
        seg_img = Image.fromarray(seg_arr, mode="I")
        draw_rgb = ImageDraw.Draw(rgb_img)
        draw_seg = ImageDraw.Draw(seg_img)

        if view == "A":
            view_mat = scene.view_A
            primitives_present = scene.linked + scene.distractors_A
            style = scene.style_A
        elif view == "B":
            view_mat = scene.view_B
            primitives_present = scene.linked + scene.distractors_B
            style = scene.style_B
        else:
            raise ValueError(view)

        # Project all primitives, painter's-algorithm sort back-to-front
        projected = []
        for p in primitives_present:
            screen, z = _project(p.pos_3d, view_mat)
            if screen is None:
                continue
            projected.append((z, p, screen))
        projected.sort(key=lambda t: -t[0])

        for z, p, (sx, sy) in projected:
            # World size → screen size: size * focal / z, then scale to pixels
            r_px = max(2.0, p.size * 1.6 / z * (W / 2))
            cx = W * 0.5 + sx * (W / 2)
            cy = H * 0.5 - sy * (H / 2)
            color = tuple(int(c * 255) for c in p.color)
            self._draw_shape(draw_rgb, draw_seg, p.shape_id,
                              cx, cy, r_px, color, p.pid)

        seg_arr = np.array(seg_img, dtype=np.int32)
        rgb_arr = np.array(rgb_img, dtype=np.uint8)

        # Apply style transform to view B
        if style == "grayscale":
            gray = (0.299 * rgb_arr[..., 0]
                    + 0.587 * rgb_arr[..., 1]
                    + 0.114 * rgb_arr[..., 2]).astype(np.uint8)
            rgb_arr = np.stack([gray, gray, gray], axis=-1)
        elif style == "edges":
            gray = (0.299 * rgb_arr[..., 0]
                    + 0.587 * rgb_arr[..., 1]
                    + 0.114 * rgb_arr[..., 2]).astype(np.float32)
            # Simple Sobel edges
            gx = np.zeros_like(gray); gy = np.zeros_like(gray)
            gx[:, 1:-1] = gray[:, 2:] - gray[:, :-2]
            gy[1:-1, :] = gray[2:, :] - gray[:-2, :]
            mag = np.sqrt(gx ** 2 + gy ** 2)
            mag = (mag / max(mag.max(), 1e-6) * 255).clip(0, 255).astype(np.uint8)
            inv = 255 - mag
            rgb_arr = np.stack([inv, inv, inv], axis=-1)

        # Add Gaussian noise
        if scene.noise_sigma > 0:
            noise = np.random.RandomState(scene.seed + (0 if view == "A" else 1)).normal(
                0, scene.noise_sigma * 255, rgb_arr.shape
            )
            rgb_arr = (rgb_arr.astype(np.float32) + noise).clip(0, 255).astype(np.uint8)

        # Build kpts + vis arrays in canonical order (linked, then distractors)
        kpts = np.full((len(primitives_present), 2), np.nan, dtype=np.float32)
        vis = np.zeros(len(primitives_present), dtype=bool)
        ids = np.array([p.pid for p in primitives_present], dtype=np.int32)
        for i, p in enumerate(primitives_present):
            screen, z = _project(p.pos_3d, view_mat)
            if screen is None:
                continue
            sx, sy = screen
            cx = W * 0.5 + sx * (W / 2)
            cy = H * 0.5 - sy * (H / 2)
            if 0 <= cx < W and 0 <= cy < H:
                kpts[i] = (cx, cy)
                vis[i] = bool((seg_arr == p.pid).any())  # actually visible
        return dict(rgb=rgb_arr, seg=seg_arr, kpts=kpts, vis=vis, ids=ids)

    def _draw_shape(self, draw_rgb, draw_seg, shape_id, cx, cy, r,
                     color, pid):
        shape_name = SHAPES[shape_id]
        if shape_name == "circle":
            bbox = [cx - r, cy - r, cx + r, cy + r]
            draw_rgb.ellipse(bbox, fill=color)
            draw_seg.ellipse(bbox, fill=int(pid))
        elif shape_name == "square":
            bbox = [cx - r, cy - r, cx + r, cy + r]
            draw_rgb.rectangle(bbox, fill=color)
            draw_seg.rectangle(bbox, fill=int(pid))
        elif shape_name == "diamond":
            pts = [(cx, cy - r), (cx + r, cy), (cx, cy + r), (cx - r, cy)]
            draw_rgb.polygon(pts, fill=color)
            draw_seg.polygon(pts, fill=int(pid))
        elif shape_name == "triangle":
            pts = [(cx, cy - r), (cx + r * 0.866, cy + r * 0.5),
                   (cx - r * 0.866, cy + r * 0.5)]
            draw_rgb.polygon(pts, fill=color)
            draw_seg.polygon(pts, fill=int(pid))
        elif shape_name == "plus":
            arm = r * 0.4
            draw_rgb.rectangle([cx - r, cy - arm, cx + r, cy + arm], fill=color)
            draw_rgb.rectangle([cx - arm, cy - r, cx + arm, cy + r], fill=color)
            draw_seg.rectangle([cx - r, cy - arm, cx + r, cy + arm], fill=int(pid))
            draw_seg.rectangle([cx - arm, cy - r, cx + arm, cy + r], fill=int(pid))
        elif shape_name == "cross":
            arm = r * 0.4
            # diagonal arms via two rotated rectangles approximated by polygons
            for ang in (math.pi / 4, -math.pi / 4):
                c, s = math.cos(ang), math.sin(ang)
                local = [(-r, -arm), (r, -arm), (r, arm), (-r, arm)]
                pts = [(cx + lx * c - ly * s, cy + lx * s + ly * c) for lx, ly in local]
                draw_rgb.polygon(pts, fill=color)
                draw_seg.polygon(pts, fill=int(pid))
        elif shape_name in ("star", "pentagon", "hexagon", "octagon"):
            n_sides = {"star": 10, "pentagon": 5, "hexagon": 6, "octagon": 8}[shape_name]
            pts = []
            for i in range(n_sides):
                ang = 2 * math.pi * i / n_sides - math.pi / 2
                rr = r if shape_name != "star" else (r if i % 2 == 0 else r * 0.5)
                pts.append((cx + rr * math.cos(ang), cy + rr * math.sin(ang)))
            draw_rgb.polygon(pts, fill=color)
            draw_seg.polygon(pts, fill=int(pid))
        else:
            # Fallback: circle
            bbox = [cx - r, cy - r, cx + r, cy + r]
            draw_rgb.ellipse(bbox, fill=color)
            draw_seg.ellipse(bbox, fill=int(pid))

    # ---- labels ---------------------------------------------------------

    def compute_label(self, scene: Scene, kind: str = "count_modulo_K",
                       K: int = 4) -> int:
        """Compute a scene label from the LATENT primitive set."""
        linked = scene.linked
        if kind == "count_modulo_K":
            return len(linked) % K
        if kind == "has_pair":
            # 1 iff there's a pair of linked primitives with same shape, different color
            for i in range(len(linked)):
                for j in range(i + 1, len(linked)):
                    a, b = linked[i], linked[j]
                    if a.shape_id == b.shape_id and a.color_idx != b.color_idx:
                        return 1
            return 0
        if kind == "n_distinct_pairs":
            seen = set()
            for p in linked:
                seen.add((p.shape_id, p.color_idx))
            return min(len(seen), K - 1)
        raise ValueError(kind)


# ===========================================================================
# Correspondence ground-truth helpers
# ===========================================================================

def correspondence_pairs(render_A: dict, render_B: dict) -> np.ndarray:
    """Given two renders of the same scene, return the (N_match, 2) array of
    (kpt_in_A, kpt_in_B) pairs that share a primitive ID and are visible in
    both modalities.

    Returns shape (M, 2, 2): M matched pairs, each with kpt in A and kpt in B
    as (x, y) pixel coordinates.
    """
    ids_A = set(int(i) for i, v in zip(render_A["ids"], render_A["vis"]) if v)
    ids_B = set(int(i) for i, v in zip(render_B["ids"], render_B["vis"]) if v)
    common = sorted(ids_A & ids_B)
    out = []
    for pid in common:
        i_A = int(np.where(render_A["ids"] == pid)[0][0])
        i_B = int(np.where(render_B["ids"] == pid)[0][0])
        out.append([render_A["kpts"][i_A], render_B["kpts"][i_B]])
    if not out:
        return np.zeros((0, 2, 2), dtype=np.float32)
    return np.array(out, dtype=np.float32)
