"""Multi-View Linked Primitives — 3-modality variant with narrow,
non-overlapping fields of view that jointly cover the scene.

WHAT THIS IS
============

A third dataset in the correspondence_bench, designed specifically for
**multi-view scene-understanding** experiments. Where Datasets A and B
produce TWO views of every scene, this one produces **N (default 3)
views**, each with a NARROW FIELD OF VIEW (higher focal length) and a
DIFFERENT camera angle, arranged so the union of the three views
roughly covers the same volume the standard rendering does.

The configuration is:

    camera 0 :  rotated  +α°  about y-axis,  focal = focal_narrow
    camera 1 :  rotated   0°  about y-axis,  focal = focal_narrow
    camera 2 :  rotated  -α°  about y-axis,  focal = focal_narrow

with α and focal_narrow chosen so that the union of the three views'
projected ranges spans the same angular extent as the standard
wide-FOV rendering. The DEFAULT (α=13°, focal_narrow=4.0) gives each
camera a ~28° half-angle and the union covers ~±27° — matching the
wide rendering at focal=2.0.

VISIBILITY GUARANTEE
====================

The whole point of limited FOV is that **each camera sees only PART of
the scene** — but the union must see ENOUGH of every primitive that
the downstream task is solvable from joint reasoning across views.

After sampling a scene, the generator can optionally **filter primitives
to those visible in at least one view** (`require_visibility=True`,
default). This means:

  * Primitives visible in ≥1 view: kept; contribute to label / corr GT.
  * Primitives visible in 0 views: dropped from the latent scene
    (they wouldn't be recoverable by any model anyway).

So the downstream "scene understanding" label is always computable in
principle: every primitive that contributes to the label is observable
somewhere.

DOWNSTREAM TASKS
================

The natural tasks for this dataset are **scene-understanding** tasks
that require multi-view aggregation:

  C-1  Multi-view classification        — label = property of the full scene;
                                          computable only by integrating views.
  C-2  Per-view segmentation             — per-pixel pid in each modality.
  C-3  Cross-modal pairing (all triples) — match (cam_i, kpt) ↔ (cam_j, kpt)
                                          for the same pid across views.
  C-4  Coverage / completion             — given features from a subset of
                                          views, predict properties of the
                                          UNSEEN region (e.g. mask one
                                          camera and predict what it would
                                          see from the other two).

Per-modality, per-view segmentation is straightforward because each
camera writes a clean per-pixel pid map. Tasks (C-3) and (C-4) are the
ones that genuinely require cross-modal reasoning.

WHY THIS COMPOSES WITH augmenters.py
====================================

`MultiViewLinkedPrimitivesGenerator.render(scene)` returns a list of N
render-output dicts. Each one can be independently passed through
augmenters from `augmenters.py` (noise, subsample, occlusion, ...).
That gives you scenarios like "view 0 noisy, view 1 occluded, view 2
clean" with no extra code.
"""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Optional, Union, List
import math

import numpy as np
from PIL import Image, ImageDraw

from linked_primitives import (
    Primitive, SHAPES, DEFAULT_PALETTE,
    _project, _rotation_x, _rotation_y,
    FOCAL_DEFAULT, CAMERA_BACK_DEFAULT,
)


# ===========================================================================
# Configuration knobs
# ===========================================================================

DEFAULT_VIEW_ANGLES_DEG = (+13.0, 0.0, -13.0)
DEFAULT_FOCAL_NARROW = 4.0

OPERATING_POINTS = {
    "easy": dict(
        n_linked=8, n_shapes=2, n_colors=3,
        n_distractors_total=0,
        scale_range=(0.10, 0.20),
        view_angles_deg=DEFAULT_VIEW_ANGLES_DEG,
        focal_narrow=DEFAULT_FOCAL_NARROW,
        require_visibility=True,
        adversarial_confusables=False,
    ),
    "basic": dict(
        n_linked=24, n_shapes=4, n_colors=5,
        n_distractors_total=4,
        scale_range=(0.06, 0.20),
        view_angles_deg=DEFAULT_VIEW_ANGLES_DEG,
        focal_narrow=DEFAULT_FOCAL_NARROW,
        require_visibility=True,
        adversarial_confusables=False,
    ),
    "hard": dict(
        n_linked=80, n_shapes=8, n_colors=8,
        n_distractors_total=20,
        scale_range=(0.03, 0.20),
        view_angles_deg=DEFAULT_VIEW_ANGLES_DEG,
        focal_narrow=DEFAULT_FOCAL_NARROW,
        require_visibility=True,
        adversarial_confusables=False,
    ),
    "extreme": dict(
        # Held to 100 so post-render coverage stays > 80% — at the narrow
        # 3-camera FOV, painter's-algorithm occlusion at very high density
        # would push many primitives below the seg-visibility threshold.
        n_linked=100, n_shapes=10, n_colors=10,
        n_distractors_total=48,
        scale_range=(0.03, 0.25),
        view_angles_deg=DEFAULT_VIEW_ANGLES_DEG,
        focal_narrow=DEFAULT_FOCAL_NARROW,
        require_visibility=True,
        adversarial_confusables=False,
    ),
    "adversarial": dict(
        n_linked=100, n_shapes=10, n_colors=10,
        n_distractors_total=64,
        scale_range=(0.03, 0.25),
        view_angles_deg=DEFAULT_VIEW_ANGLES_DEG,
        focal_narrow=DEFAULT_FOCAL_NARROW,
        require_visibility=True,
        adversarial_confusables=True,
    ),
}


# ===========================================================================
# Scene
# ===========================================================================

@dataclass
class MVScene:
    """Scene with N camera matrices and per-camera focal length.

    `linked` is the FILTERED set of primitives — those visible in at
    least one camera (if require_visibility=True at sampling time).
    `unobservable` holds primitives that were sampled but invisible
    everywhere — they are returned for diagnostic purposes only and
    do NOT contribute to labels or correspondences.
    """
    linked: list[Primitive]
    unobservable: list[Primitive]
    distractors_per_view: list[list[Primitive]]   # length N
    view_matrices: list[np.ndarray]                # length N, each (3, 3)
    focal_narrow: float
    knobs: dict = field(default_factory=dict)
    seed: int = 0

    @property
    def n_views(self) -> int:
        return len(self.view_matrices)


# ===========================================================================
# Generator
# ===========================================================================

class MultiViewLinkedPrimitivesGenerator:
    """Render the same 3D primitive scene from N cameras (N≥1).

    Each camera has its own rotation matrix and focal length. The
    default 3-camera config tiles ±13° at focal=4.0 (narrow FOV) so
    that the union of views covers the same angular extent as the
    standard wide-FOV rendering.

    Parameters
    ----------
    operating_point : str or dict
        One of "easy", "basic", "hard", "extreme", "adversarial", or a
        custom dict of knobs.
    image_size : int
        Output H × W.
    background : (R, G, B) uint8 tuple
    palette : (K, 3) float32 array
    base_seed : int
        Default seed used when sample_scene is called without one.
    """
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

    # ---- scene sampling ----

    def sample_scene(self, seed: Optional[int] = None) -> MVScene:
        rng = np.random.RandomState(seed if seed is not None else self.base_seed)
        k = self.knobs

        shape_ids = list(range(min(k["n_shapes"], len(SHAPES))))
        color_ids = list(range(min(k["n_colors"], len(self.palette))))

        def random_primitive(pid: int) -> Primitive:
            shape_id = int(rng.choice(shape_ids))
            color_idx = int(rng.choice(color_ids))
            color = self.palette[color_idx]
            # Sample positions in a slightly wider 3D volume than the 2-view
            # case to give the 3 narrow cameras useful coverage.
            pos = rng.uniform(-1.2, 1.2, size=3).astype(np.float32)
            s_lo, s_hi = k["scale_range"]
            size = float(math.exp(rng.uniform(math.log(s_lo), math.log(s_hi))))
            return Primitive(shape_id=shape_id, color_idx=color_idx,
                              color=color, pos_3d=pos, size=size, pid=pid)

        # OVERSAMPLE candidates so the *visible-in-at-least-one-view* count
        # comes close to the requested n_linked target. With narrow FOVs,
        # ~60% of randomly-placed primitives can fall outside all 3 cameras
        # at the hardest operating points — oversampling by ~3x compensates.
        target_n = k["n_linked"]
        oversample_factor = 3
        candidates = [random_primitive(i) for i in range(target_n * oversample_factor)]

        if k.get("adversarial_confusables", False):
            n_extra = min(target_n // 2, 12)
            n_before = len(candidates)
            for j in range(n_extra):
                src = candidates[rng.randint(n_before)]
                p = random_primitive(n_before + j)
                p.shape_id, p.color_idx, p.color = src.shape_id, src.color_idx, src.color.copy()
                p.pos_3d = (src.pos_3d + rng.uniform(-0.15, 0.15, 3)).astype(np.float32)
                p.size = src.size * float(rng.uniform(0.7, 1.3))
                candidates.append(p)

        # Camera setup
        view_angles_deg = k["view_angles_deg"]
        focal_narrow = float(k["focal_narrow"])
        view_matrices = []
        tilt = math.radians(float(rng.uniform(-10, 10)))   # shared small x-tilt
        for ang_deg in view_angles_deg:
            ang_rad = math.radians(float(ang_deg))
            view_matrices.append(_rotation_x(tilt) @ _rotation_y(ang_rad))

        # Visibility check: drop primitives that no camera sees.
        # _project returns None if behind the camera, but on-screen visibility
        # also requires |sx|, |sy| ≤ 1 (after the focal). We do that check
        # explicitly here using the narrow focal.
        H = W = self.image_size

        def is_visible_in_any_view(p: Primitive) -> bool:
            for V in view_matrices:
                screen, z = _project(p.pos_3d, V,
                                      focal=focal_narrow, camera_back=CAMERA_BACK_DEFAULT)
                if screen is None: continue
                sx, sy = screen
                if -1.0 <= sx <= 1.0 and -1.0 <= sy <= 1.0:
                    return True
            return False

        if k.get("require_visibility", True):
            observable   = [p for p in candidates if is_visible_in_any_view(p)]
            unobservable = [p for p in candidates if not is_visible_in_any_view(p)]
            # Truncate to the requested target so the final `n_linked` matches
            # the operating-point spec (independent of how many candidates
            # the oversampling produced).
            linked = observable[:target_n]
            # Anything past the target counts as unobservable for label purposes
            unobservable = unobservable + observable[target_n:]
        else:
            linked, unobservable = candidates[:target_n], candidates[target_n:]

        # Renumber pids contiguously over the kept set so seg uses a small range
        for i, p in enumerate(linked):
            p.pid = i

        # Distractors per view — primitives that appear only in one view.
        # They get pids starting after the linked set.
        n_link = len(linked)
        n_distr_total = k["n_distractors_total"]
        n_distr_per_view = n_distr_total // len(view_matrices)
        distractors_per_view = []
        next_pid = n_link
        for v_idx in range(len(view_matrices)):
            ds = []
            for _ in range(n_distr_per_view):
                p = random_primitive(next_pid)
                ds.append(p)
                next_pid += 1
            distractors_per_view.append(ds)

        return MVScene(
            linked=linked,
            unobservable=unobservable,
            distractors_per_view=distractors_per_view,
            view_matrices=view_matrices,
            focal_narrow=focal_narrow,
            knobs=dict(self.knobs),
            seed=int(seed if seed is not None else self.base_seed),
        )

    # ---- rendering ----

    def render(self, scene: MVScene) -> list[dict]:
        """Render every camera. Returns a list of N render-output dicts.

        Each dict has the same schema as LinkedPrimitivesGenerator.render:
            rgb  : (H, W, 3) uint8
            seg  : (H, W)    int32; -1 background
            kpts : (N_in_view, 2) float32
            vis  : (N_in_view,)   bool
            ids  : (N_in_view,)   int32 — primitive IDs in canonical order

        Canonical order in each view = linked primitives (n_link of them)
        followed by that view's distractors. So `ids[:n_link]` is the
        same across all views (they share the same linked set), and the
        per-view distractor IDs start at index n_link.
        """
        outs = []
        for v_idx, view_mat in enumerate(scene.view_matrices):
            primitives = scene.linked + scene.distractors_per_view[v_idx]
            outs.append(self._render_one(primitives, view_mat, scene.focal_narrow))
        return outs

    def _render_one(self, primitives, view_mat, focal_narrow):
        H = W = self.image_size
        rgb_img = Image.new("RGB", (W, H), self.background)
        seg_arr_init = np.full((H, W), -1, dtype=np.int32)
        seg_img = Image.fromarray(seg_arr_init, mode="I")
        draw_rgb = ImageDraw.Draw(rgb_img)
        draw_seg = ImageDraw.Draw(seg_img)

        projected = []
        for p in primitives:
            screen, z = _project(p.pos_3d, view_mat,
                                  focal=focal_narrow, camera_back=CAMERA_BACK_DEFAULT)
            if screen is None: continue
            projected.append((z, p, screen))
        projected.sort(key=lambda t: -t[0])

        for z, p, (sx, sy) in projected:
            r_px = max(2.0, p.size * focal_narrow / z * (W / 2))
            cx = W * 0.5 + sx * (W / 2)
            cy = H * 0.5 - sy * (H / 2)
            color = tuple(int(c * 255) for c in p.color)
            self._draw_shape(draw_rgb, draw_seg, p.shape_id, cx, cy, r_px, color, p.pid)

        seg_arr = np.array(seg_img, dtype=np.int32)
        rgb_arr = np.array(rgb_img, dtype=np.uint8)

        ids = np.array([p.pid for p in primitives], dtype=np.int32)
        kpts = np.full((len(primitives), 2), np.nan, dtype=np.float32)
        vis = np.zeros(len(primitives), dtype=bool)
        for i, p in enumerate(primitives):
            screen, z = _project(p.pos_3d, view_mat,
                                  focal=focal_narrow, camera_back=CAMERA_BACK_DEFAULT)
            if screen is None: continue
            sx, sy = screen
            cx = W * 0.5 + sx * (W / 2); cy = H * 0.5 - sy * (H / 2)
            if 0 <= cx < W and 0 <= cy < H:
                kpts[i] = (cx, cy)
                vis[i] = bool((seg_arr == p.pid).any())
        return dict(rgb=rgb_arr, seg=seg_arr, kpts=kpts, vis=vis, ids=ids)

    def _draw_shape(self, draw_rgb, draw_seg, shape_id, cx, cy, r, color, pid):
        # Same dispatch as linked_primitives.py — extracted as a method to
        # avoid an import cycle if we ever change the static-renderer signature.
        shape_name = SHAPES[shape_id]
        if shape_name == "circle":
            bbox = [cx - r, cy - r, cx + r, cy + r]
            draw_rgb.ellipse(bbox, fill=color); draw_seg.ellipse(bbox, fill=int(pid))
        elif shape_name == "square":
            bbox = [cx - r, cy - r, cx + r, cy + r]
            draw_rgb.rectangle(bbox, fill=color); draw_seg.rectangle(bbox, fill=int(pid))
        elif shape_name == "diamond":
            pts = [(cx, cy - r), (cx + r, cy), (cx, cy + r), (cx - r, cy)]
            draw_rgb.polygon(pts, fill=color); draw_seg.polygon(pts, fill=int(pid))
        elif shape_name == "triangle":
            pts = [(cx, cy - r), (cx + r * 0.866, cy + r * 0.5),
                   (cx - r * 0.866, cy + r * 0.5)]
            draw_rgb.polygon(pts, fill=color); draw_seg.polygon(pts, fill=int(pid))
        elif shape_name == "plus":
            arm = r * 0.4
            draw_rgb.rectangle([cx - r, cy - arm, cx + r, cy + arm], fill=color)
            draw_rgb.rectangle([cx - arm, cy - r, cx + arm, cy + r], fill=color)
            draw_seg.rectangle([cx - r, cy - arm, cx + r, cy + arm], fill=int(pid))
            draw_seg.rectangle([cx - arm, cy - r, cx + arm, cy + r], fill=int(pid))
        elif shape_name == "cross":
            arm = r * 0.4
            for ang in (math.pi / 4, -math.pi / 4):
                c, s = math.cos(ang), math.sin(ang)
                local = [(-r, -arm), (r, -arm), (r, arm), (-r, arm)]
                pts = [(cx + lx * c - ly * s, cy + lx * s + ly * c) for lx, ly in local]
                draw_rgb.polygon(pts, fill=color); draw_seg.polygon(pts, fill=int(pid))
        elif shape_name in ("star", "pentagon", "hexagon", "octagon"):
            n_sides = {"star": 10, "pentagon": 5, "hexagon": 6, "octagon": 8}[shape_name]
            pts = []
            for i in range(n_sides):
                ang = 2 * math.pi * i / n_sides - math.pi / 2
                rr = r if shape_name != "star" else (r if i % 2 == 0 else r * 0.5)
                pts.append((cx + rr * math.cos(ang), cy + rr * math.sin(ang)))
            draw_rgb.polygon(pts, fill=color); draw_seg.polygon(pts, fill=int(pid))
        else:
            bbox = [cx - r, cy - r, cx + r, cy + r]
            draw_rgb.ellipse(bbox, fill=color); draw_seg.ellipse(bbox, fill=int(pid))

    # ---- labels ----

    def compute_label(self, scene: MVScene, kind: str = "count_modulo_K",
                       K: int = 4) -> int:
        """Compute the scene-understanding label from the FILTERED linked set.

        Because primitives invisible in every view are dropped at scene-
        sampling time, the label is always recoverable in principle from
        the union of view observations.

        Available kinds:
          - "count_modulo_K"      number of (observable) linked primitives mod K
          - "has_pair"            1 iff any same-shape, different-color pair exists
          - "n_distinct_shapes"   count of distinct shape ids; clipped to K-1
          - "spans_all_views"     1 iff at least one primitive is visible in EVERY view
        """
        linked = scene.linked
        if kind == "count_modulo_K":
            return len(linked) % K
        if kind == "has_pair":
            for i in range(len(linked)):
                for j in range(i + 1, len(linked)):
                    if linked[i].shape_id == linked[j].shape_id \
                       and linked[i].color_idx != linked[j].color_idx:
                        return 1
            return 0
        if kind == "n_distinct_shapes":
            return min(len({p.shape_id for p in linked}), K - 1)
        if kind == "spans_all_views":
            # Visible in EVERY camera
            for p in linked:
                ok = True
                for V in scene.view_matrices:
                    screen, _ = _project(p.pos_3d, V,
                                          focal=scene.focal_narrow,
                                          camera_back=CAMERA_BACK_DEFAULT)
                    if screen is None or not (-1.0 <= screen[0] <= 1.0 and -1.0 <= screen[1] <= 1.0):
                        ok = False; break
                if ok: return 1
            return 0
        raise ValueError(kind)


# ===========================================================================
# Correspondence helpers
# ===========================================================================

def cross_view_pairs_triple(renders: list[dict],
                             i: int, j: int) -> np.ndarray:
    """Pairs of (kpt_i, kpt_j) for primitives visible in both views i and j.

    Returns (M, 2, 2) — M matched pairs, each with (x, y) in view i and view j.
    """
    A = renders[i]; B = renders[j]
    ids_A = set(int(p) for p, v in zip(A["ids"], A["vis"]) if v)
    ids_B = set(int(p) for p, v in zip(B["ids"], B["vis"]) if v)
    common = sorted(ids_A & ids_B)
    if not common: return np.zeros((0, 2, 2), dtype=np.float32)
    out = []
    for pid in common:
        iA = int(np.where(A["ids"] == pid)[0][0])
        iB = int(np.where(B["ids"] == pid)[0][0])
        out.append([A["kpts"][iA], B["kpts"][iB]])
    return np.array(out, dtype=np.float32)


def coverage_summary(renders: list[dict], n_linked: int) -> dict:
    """How well do the N views cover the linked primitives?

    Returns:
      visible_in_each : (n_views,) int — count of linked pids visible in each view
      visible_in_any  : int — count of linked pids visible in at least one view
      visible_in_all  : int — count visible in every view
      coverage_frac   : float — visible_in_any / n_linked
    """
    n_views = len(renders)
    seen = np.zeros((n_linked, n_views), dtype=bool)
    for v_idx, r in enumerate(renders):
        ids = r["ids"]; vis = r["vis"]
        for i, pid in enumerate(ids):
            if pid < n_linked and vis[i]:
                seen[pid, v_idx] = True
    return {
        "visible_in_each": seen.sum(axis=0).tolist(),
        "visible_in_any":  int((seen.any(axis=1)).sum()),
        "visible_in_all":  int((seen.all(axis=1)).sum()),
        "coverage_frac":   float(seen.any(axis=1).mean()) if n_linked > 0 else 0.0,
    }
