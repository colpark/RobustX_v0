"""Linked Primitives Video — spatiotemporal version of Linked Primitives.

The static "Linked Primitives" dataset (in ``linked_primitives.py``)
captures cross-view correspondences at one instant. This sibling dataset
adds **motion** — each primitive has a trajectory `p_i(τ)` over τ ∈ [0, 1]
and an appearance lifetime — so the same scene produces a *video pair*
(one short video per camera view) instead of a single image pair.

What this lets us do:

  * Generate paired videos: `(video_A, video_B)` where view A and view B
    are two cameras of the same 3D scene evolving over time.
  * Export GIFs for inspection.
  * Three kinds of correspondence ground truth (see § 1).

The implementation reuses the static-rendering primitives from
``linked_primitives.py``: same projection, same shape rasterization, same
seg-mask scheme. The only new thing is the per-primitive trajectory.

────────────────────────────────────────────────────────────────────────
§ 1. The three kinds of correspondence we get for free

    1. **cross-view at same time**     (pid, view=A, τ) ↔ (pid, view=B, τ)
       — same primitive in both cameras at one instant
    2. **cross-time within a view**    (pid, view=A, τ₁) ↔ (pid, view=A, τ₂)
       — same primitive at two times: tracking ground truth
    3. **cross-view AND cross-time**   (pid, view=A, τ₁) ↔ (pid, view=B, τ₂)
       — the union of the other two: the most flexible ground truth

Each primitive has a stable integer pid that appears wherever it's
visible, in any view and any frame. So all three correspondence types
collapse to "same pid".

────────────────────────────────────────────────────────────────────────
§ 2. Output of `render_video_pair`

    {
        "view_A": {
            "rgb":  (T, H, W, 3) uint8   video frames
            "seg":  (T, H, W)    int32   per-pixel pid; -1 background
            "kpts": (T, N, 2)    float32 2D projected centers at each frame
            "vis":  (T, N)       bool    visibility flag per (frame, primitive)
            "ids":  (N,)         int32   pids of all primitives in this view
        },
        "view_B": {... same structure ...},
        "times": (T,) float32  — the τ values sampled
    }

The two views share a primitive ID space: pid 0..n_linked-1 appear in
both, pid ≥ n_linked are view-only distractors.
"""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Optional, Union, Callable
import math, io

import numpy as np
from PIL import Image, ImageDraw

# Reuse static-rendering bits from the sibling file
from linked_primitives import (
    Primitive, Scene as StaticScene,
    SHAPES, DEFAULT_PALETTE,
    _project, _rotation_x, _rotation_y,
    FOCAL_DEFAULT,
)


# ===========================================================================
# Difficulty operating points (parallel structure to linked_primitives)
# ===========================================================================

OPERATING_POINTS = {
    # ── Base difficulty ladder ──────────────────────────────────────────
    # frequency_range is in CYCLES PER VIDEO (sinusoidal / circular only):
    #   0.25–1.0   = slow      (less than a cycle to one cycle over T)
    #   1.0–3.0    = medium
    #   3.0–8.0    = fast       (multiple cycles)
    "easy": dict(
        n_linked=4, n_shapes=2, n_colors=3,
        view_disparity_deg=30.0,
        n_distractors_A=0, n_distractors_B=0,
        scale_range=(0.10, 0.15),
        style="rgb", noise_sigma=0.0,
        adversarial_confusables=False,
        motion_type="static_or_slow_linear",
        motion_amplitude=0.0,
        frequency_range=(0.25, 0.75),
        n_frames=8, fps=8,
        lifetime_jitter=0.0,
        cross_modal_time_offset=0.0,
    ),
    "basic": dict(
        n_linked=16, n_shapes=4, n_colors=5,
        view_disparity_deg=60.0,
        n_distractors_A=2, n_distractors_B=2,
        scale_range=(0.06, 0.16),
        style="rgb", noise_sigma=0.02,
        adversarial_confusables=False,
        motion_type="linear_or_slow_sin",
        motion_amplitude=0.20,
        frequency_range=(0.5, 1.5),
        n_frames=12, fps=8,
        lifetime_jitter=0.0,
        cross_modal_time_offset=0.0,
    ),
    "hard": dict(
        n_linked=64, n_shapes=8, n_colors=8,
        view_disparity_deg=120.0,
        n_distractors_A=16, n_distractors_B=16,
        scale_range=(0.03, 0.20),
        style="grayscale_B", noise_sigma=0.05,
        adversarial_confusables=False,
        motion_type="mixed",
        motion_amplitude=0.35,
        frequency_range=(0.5, 3.0),
        n_frames=16, fps=8,
        lifetime_jitter=0.15,
        cross_modal_time_offset=0.01,
    ),
    "extreme": dict(
        n_linked=128, n_shapes=10, n_colors=10,
        view_disparity_deg=170.0,
        n_distractors_A=48, n_distractors_B=48,
        scale_range=(0.02, 0.25),
        style="edges_B", noise_sigma=0.08,
        adversarial_confusables=False,
        motion_type="mixed",
        motion_amplitude=0.50,
        frequency_range=(0.5, 5.0),
        n_frames=24, fps=12,
        lifetime_jitter=0.30,
        cross_modal_time_offset=0.02,
    ),
    "adversarial": dict(
        n_linked=128, n_shapes=10, n_colors=10,
        view_disparity_deg=170.0,
        n_distractors_A=64, n_distractors_B=64,
        scale_range=(0.02, 0.25),
        style="edges_B", noise_sigma=0.10,
        adversarial_confusables=True,
        motion_type="mixed",
        motion_amplitude=0.60,
        frequency_range=(0.5, 5.0),
        n_frames=24, fps=12,
        lifetime_jitter=0.30,
        cross_modal_time_offset=0.03,
    ),

    # ── Hz-focused operating points (test slow vs fast vs simultaneous) ──
    "slow_only": dict(
        n_linked=16, n_shapes=4, n_colors=5,
        view_disparity_deg=60.0,
        n_distractors_A=2, n_distractors_B=2,
        scale_range=(0.06, 0.16),
        style="rgb", noise_sigma=0.02,
        adversarial_confusables=False,
        motion_type="sinusoidal",
        motion_amplitude=0.30,
        frequency_range=(0.25, 1.0),   # ALL primitives slow
        n_frames=16, fps=8,
        lifetime_jitter=0.0,
        cross_modal_time_offset=0.0,
    ),
    "fast_only": dict(
        n_linked=16, n_shapes=4, n_colors=5,
        view_disparity_deg=60.0,
        n_distractors_A=2, n_distractors_B=2,
        scale_range=(0.06, 0.16),
        style="rgb", noise_sigma=0.02,
        adversarial_confusables=False,
        motion_type="sinusoidal",
        motion_amplitude=0.30,
        frequency_range=(3.0, 8.0),    # ALL primitives fast
        n_frames=24, fps=12,
        lifetime_jitter=0.0,
        cross_modal_time_offset=0.0,
    ),
    "mixed_hz": dict(
        n_linked=24, n_shapes=4, n_colors=5,
        view_disparity_deg=60.0,
        n_distractors_A=2, n_distractors_B=2,
        scale_range=(0.06, 0.16),
        style="rgb", noise_sigma=0.02,
        adversarial_confusables=False,
        motion_type="sinusoidal",
        motion_amplitude=0.30,
        frequency_range=(0.25, 6.0),   # WIDE: each primitive samples
                                        # independently → scene has both
                                        # slow AND fast simultaneously
        n_frames=24, fps=12,
        lifetime_jitter=0.0,
        cross_modal_time_offset=0.0,
    ),
    "multiscale_hz": dict(
        n_linked=64, n_shapes=8, n_colors=8,
        view_disparity_deg=120.0,
        n_distractors_A=16, n_distractors_B=16,
        scale_range=(0.03, 0.20),
        style="grayscale_B", noise_sigma=0.05,
        adversarial_confusables=False,
        motion_type="mixed",
        motion_amplitude=0.40,
        frequency_range=(0.25, 8.0),   # very wide range, harder backbone
        n_frames=24, fps=12,
        lifetime_jitter=0.15,
        cross_modal_time_offset=0.01,
    ),
}


# ===========================================================================
# Trajectory representation
# ===========================================================================

@dataclass
class Trajectory:
    """Time-parameterized 3D motion.

    `pos_at(τ)` returns the 3D position at normalized time τ ∈ [0, 1].
    All trajectory kinds are evaluated cheaply (closed-form), no integration.

    Kinds and their params:
      - "static"       : params = {} ; pos(τ) = pos_0
      - "linear"       : params = {"velocity": (3,)} ; pos(τ) = pos_0 + v · τ
      - "sinusoidal"   : params = {"amp": (3,), "freq": scalar, "phase": (3,)}
                         pos(τ) = pos_0 + amp · sin(2π · freq · τ + phase)
      - "circular"     : params = {"radius": scalar, "omega": scalar, "axis": (3,)}
                         pos(τ) traces a circle in the plane ⟂ axis, center = pos_0
    """
    kind: str
    pos_0: np.ndarray
    params: dict

    def pos_at(self, tau: float) -> np.ndarray:
        if self.kind == "static":
            return self.pos_0.copy()
        if self.kind == "linear":
            return (self.pos_0 + self.params["velocity"] * tau).astype(np.float32)
        if self.kind == "sinusoidal":
            amp = self.params["amp"]; freq = self.params["freq"]; phase = self.params["phase"]
            return (self.pos_0 + amp * np.sin(2 * math.pi * freq * tau + phase)).astype(np.float32)
        if self.kind == "circular":
            r = self.params["radius"]; omega = self.params["omega"]
            ang = omega * tau
            # circle in xy-plane around pos_0 by default
            return (self.pos_0 + np.array(
                [r * math.cos(ang), r * math.sin(ang), 0.0],
                dtype=np.float32,
            )).astype(np.float32)
        raise ValueError(self.kind)


# ===========================================================================
# Spatiotemporal primitive + scene
# ===========================================================================

@dataclass
class STPrimitive:
    pid: int
    shape_id: int
    color_idx: int
    color: np.ndarray
    size: float
    trajectory: Trajectory
    # Lifetime in normalized time. Primitive is visible iff lifetime[0] ≤ τ ≤ lifetime[1].
    lifetime: tuple = (0.0, 1.0)


@dataclass
class STScene:
    linked: list[STPrimitive]
    distractors_A: list[STPrimitive]
    distractors_B: list[STPrimitive]
    view_A: np.ndarray
    view_B: np.ndarray
    style_A: str = "rgb"
    style_B: str = "rgb"
    noise_sigma: float = 0.0
    cross_modal_time_offset: float = 0.0          # τ_B = τ_A + this
    n_frames: int = 12
    fps: int = 8
    knobs: dict = field(default_factory=dict)
    seed: int = 0


# ===========================================================================
# Sampling trajectories
# ===========================================================================

def _sample_trajectory(rng: np.random.RandomState,
                       motion_type: str,
                       amplitude: float,
                       freq_range: tuple,
                       pos_0: np.ndarray) -> Trajectory:
    """Pick a trajectory kind + parameters per the operating point's
    `motion_type` policy.

    `freq_range = (f_min, f_max)` is in **cycles per video** (sinusoidal
    and circular only). Each primitive samples its frequency
    independently from this range, so a wide range produces a scene
    containing both slow and fast primitives simultaneously.
    """
    if motion_type == "static_or_slow_linear":
        kind = rng.choice(["static", "linear"], p=[0.7, 0.3])
    elif motion_type == "linear_or_slow_sin":
        kind = rng.choice(["linear", "sinusoidal"], p=[0.6, 0.4])
    elif motion_type == "mixed":
        kind = rng.choice(["static", "linear", "sinusoidal", "circular"],
                           p=[0.15, 0.40, 0.30, 0.15])
    elif motion_type in ("static", "linear", "sinusoidal", "circular"):
        kind = motion_type
    else:
        raise ValueError(motion_type)

    A = float(amplitude)
    f_lo, f_hi = float(freq_range[0]), float(freq_range[1])

    if kind == "static":
        return Trajectory("static", pos_0, {"freq": 0.0})
    if kind == "linear":
        v = rng.uniform(-1, 1, 3).astype(np.float32) * A
        return Trajectory("linear", pos_0, {"velocity": v, "freq": 0.0})
    if kind == "sinusoidal":
        amp = rng.uniform(-1, 1, 3).astype(np.float32) * A
        # log-uniform sampling across the freq range gives a sensible
        # multi-scale spread (slow and fast equally likely on log axis)
        freq = float(math.exp(rng.uniform(math.log(f_lo), math.log(f_hi))))
        phase = rng.uniform(0, 2 * math.pi, 3).astype(np.float32)
        return Trajectory("sinusoidal", pos_0,
                          {"amp": amp, "freq": freq, "phase": phase})
    if kind == "circular":
        radius = float(rng.uniform(0.3, 1.0)) * A
        freq = float(math.exp(rng.uniform(math.log(f_lo), math.log(f_hi))))
        sign = float(rng.choice([-1.0, 1.0]))
        omega = 2.0 * math.pi * freq * sign
        return Trajectory("circular", pos_0,
                          {"radius": radius, "omega": omega, "freq": freq})
    raise ValueError(kind)


# ===========================================================================
# Generator
# ===========================================================================

class LinkedPrimitivesVideoGenerator:
    """Sample spatiotemporal scenes and render them as video pairs."""

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

    def sample_scene(self, seed: Optional[int] = None) -> STScene:
        rng = np.random.RandomState(seed if seed is not None else self.base_seed)
        k = self.knobs

        shape_ids = list(range(min(k["n_shapes"], len(SHAPES))))
        color_ids = list(range(min(k["n_colors"], len(self.palette))))

        def random_st_primitive(pid: int) -> STPrimitive:
            shape_id = int(rng.choice(shape_ids))
            color_idx = int(rng.choice(color_ids))
            color = self.palette[color_idx]
            # Wider initial position so primitives reach near the image edges
            # even after some motion. Motion can still push them off-screen
            # at the highest amplitudes — that's fine, it's a difficulty knob.
            pos_0 = rng.uniform(-1.0, 1.0, size=3).astype(np.float32)
            s_lo, s_hi = k["scale_range"]
            size = float(math.exp(rng.uniform(math.log(s_lo), math.log(s_hi))))
            freq_range = k.get("frequency_range", (0.5, 1.5))
            traj = _sample_trajectory(rng, k["motion_type"],
                                       k["motion_amplitude"], freq_range, pos_0)
            # Lifetime model: `lifetime_jitter` is the PROBABILITY of being
            # transient. Persistent primitives live the full [0, 1] and are
            # available for tracking across the whole video; transient ones
            # get a random sub-window (still containing >50% of the timeline
            # in expectation).
            jit = float(k.get("lifetime_jitter", 0.0))
            if jit > 0 and rng.uniform() < jit:
                t_lo = float(rng.uniform(0.0, 0.3))
                t_hi = float(rng.uniform(0.7, 1.0))
            else:
                t_lo, t_hi = 0.0, 1.0
            return STPrimitive(pid=pid, shape_id=shape_id, color_idx=color_idx,
                                color=color, size=size, trajectory=traj,
                                lifetime=(t_lo, t_hi))

        linked: list[STPrimitive] = []
        for i in range(k["n_linked"]):
            linked.append(random_st_primitive(i))

        # Adversarial confusables: cluster near-duplicates near random anchors
        if k.get("adversarial_confusables", False) and len(linked) > 0:
            n_extra = min(len(linked) // 2, 16)
            n_linked_before = len(linked)
            for j in range(n_extra):
                src = linked[rng.randint(n_linked_before)]
                pos0 = (src.trajectory.pos_0 + rng.uniform(-0.15, 0.15, 3)).astype(np.float32)
                traj = _sample_trajectory(
                    rng, k["motion_type"], k["motion_amplitude"],
                    k.get("frequency_range", (0.5, 1.5)), pos0,
                )
                linked.append(STPrimitive(
                    pid=n_linked_before + j,
                    shape_id=src.shape_id, color_idx=src.color_idx, color=src.color.copy(),
                    size=src.size * float(rng.uniform(0.7, 1.3)),
                    trajectory=traj, lifetime=src.lifetime,
                ))

        n_link = len(linked)
        distractors_A = []
        for j in range(k["n_distractors_A"]):
            p = random_st_primitive(n_link + j)
            distractors_A.append(p)
        distractors_B = []
        for j in range(k["n_distractors_B"]):
            p = random_st_primitive(n_link + len(distractors_A) + j)
            distractors_B.append(p)

        # Two viewpoints
        delta = math.radians(k["view_disparity_deg"]) * 0.5
        tilt = math.radians(float(rng.uniform(-15, 15)))
        view_A = _rotation_x(tilt) @ _rotation_y(-delta)
        view_B = _rotation_x(tilt) @ _rotation_y(+delta)

        style_B = k.get("style", "rgb")
        if style_B == "grayscale_B": style_B = "grayscale"
        elif style_B == "edges_B":  style_B = "edges"

        return STScene(
            linked=linked,
            distractors_A=distractors_A,
            distractors_B=distractors_B,
            view_A=view_A, view_B=view_B,
            style_A="rgb", style_B=style_B,
            noise_sigma=k.get("noise_sigma", 0.0),
            cross_modal_time_offset=k.get("cross_modal_time_offset", 0.0),
            n_frames=int(k.get("n_frames", 12)),
            fps=int(k.get("fps", 8)),
            knobs=dict(self.knobs),
            seed=int(seed if seed is not None else self.base_seed),
        )

    # ---- frame rendering ----

    def render_frame(self, scene: STScene, tau: float, view: str = "A") -> dict:
        """Render a single frame at normalized time τ ∈ [0, 1]."""
        H = W = self.image_size
        rgb_img = Image.new("RGB", (W, H), self.background)
        seg_arr_init = np.full((H, W), -1, dtype=np.int32)
        seg_img = Image.fromarray(seg_arr_init, mode="I")
        draw_rgb = ImageDraw.Draw(rgb_img)
        draw_seg = ImageDraw.Draw(seg_img)

        if view == "A":
            view_mat = scene.view_A
            primitives = scene.linked + scene.distractors_A
            tau_eff = tau
            style = scene.style_A
        elif view == "B":
            view_mat = scene.view_B
            primitives = scene.linked + scene.distractors_B
            tau_eff = tau + scene.cross_modal_time_offset
            style = scene.style_B
        else:
            raise ValueError(view)

        # Project all currently-alive primitives
        projected = []
        for p in primitives:
            if not (p.lifetime[0] <= tau <= p.lifetime[1]):
                continue
            pos_3d = p.trajectory.pos_at(tau_eff)
            screen, z = _project(pos_3d, view_mat)
            if screen is None:
                continue
            projected.append((z, p, screen))
        projected.sort(key=lambda t: -t[0])   # back to front

        for z, p, (sx, sy) in projected:
            r_px = max(2.0, p.size * 1.6 / z * (W / 2))
            cx = W * 0.5 + sx * (W / 2)
            cy = H * 0.5 - sy * (H / 2)
            color = tuple(int(c * 255) for c in p.color)
            self._draw_shape(draw_rgb, draw_seg, p.shape_id, cx, cy, r_px, color, p.pid)

        seg_arr = np.array(seg_img, dtype=np.int32)
        rgb_arr = np.array(rgb_img, dtype=np.uint8)

        # Style transform on view B
        if style == "grayscale":
            gray = (0.299 * rgb_arr[..., 0] + 0.587 * rgb_arr[..., 1] + 0.114 * rgb_arr[..., 2]).astype(np.uint8)
            rgb_arr = np.stack([gray, gray, gray], axis=-1)
        elif style == "edges":
            gray = (0.299 * rgb_arr[..., 0] + 0.587 * rgb_arr[..., 1] + 0.114 * rgb_arr[..., 2]).astype(np.float32)
            gx = np.zeros_like(gray); gy = np.zeros_like(gray)
            gx[:, 1:-1] = gray[:, 2:] - gray[:, :-2]
            gy[1:-1, :] = gray[2:, :] - gray[:-2, :]
            mag = np.sqrt(gx ** 2 + gy ** 2)
            mag = (mag / max(mag.max(), 1e-6) * 255).clip(0, 255).astype(np.uint8)
            inv = 255 - mag
            rgb_arr = np.stack([inv, inv, inv], axis=-1)

        # Noise (deterministic per (seed, view, frame index))
        if scene.noise_sigma > 0:
            tag = int(round(tau * 1000)) + (1000 if view == "B" else 0)
            noise = np.random.RandomState(scene.seed + tag).normal(
                0, scene.noise_sigma * 255, rgb_arr.shape
            )
            rgb_arr = (rgb_arr.astype(np.float32) + noise).clip(0, 255).astype(np.uint8)

        # Per-primitive keypoints + visibility
        ids = np.array([p.pid for p in primitives], dtype=np.int32)
        kpts = np.full((len(primitives), 2), np.nan, dtype=np.float32)
        vis = np.zeros(len(primitives), dtype=bool)
        for i, p in enumerate(primitives):
            if not (p.lifetime[0] <= tau <= p.lifetime[1]):
                continue
            pos_3d = p.trajectory.pos_at(tau_eff)
            screen, z = _project(pos_3d, view_mat)
            if screen is None: continue
            sx, sy = screen
            cx = W * 0.5 + sx * (W / 2); cy = H * 0.5 - sy * (H / 2)
            if 0 <= cx < W and 0 <= cy < H:
                kpts[i] = (cx, cy)
                vis[i] = bool((seg_arr == p.pid).any())
        return dict(rgb=rgb_arr, seg=seg_arr, kpts=kpts, vis=vis, ids=ids)

    def render_video_pair(self, scene: STScene) -> dict:
        """Render both views at scene.n_frames evenly spaced times."""
        times = np.linspace(0.0, 1.0, scene.n_frames, dtype=np.float32)
        out = {"times": times}
        for view in ("A", "B"):
            T_frames = []
            for tau in times:
                T_frames.append(self.render_frame(scene, float(tau), view=view))
            rgb_stack = np.stack([f["rgb"] for f in T_frames], axis=0)   # (T, H, W, 3)
            seg_stack = np.stack([f["seg"] for f in T_frames], axis=0)   # (T, H, W)
            kpts_stack = np.stack([f["kpts"] for f in T_frames], axis=0) # (T, N, 2)
            vis_stack = np.stack([f["vis"] for f in T_frames], axis=0)   # (T, N)
            ids = T_frames[0]["ids"]
            out[f"view_{view}"] = dict(
                rgb=rgb_stack, seg=seg_stack, kpts=kpts_stack,
                vis=vis_stack, ids=ids,
            )
        return out

    # ---- GIF export ----

    def save_gif(self, frames: np.ndarray, path: str, fps: int = 8,
                 loop: int = 0):
        """Save (T, H, W, 3) uint8 frames as an animated GIF.

        fps: frames per second
        loop: 0 = loop forever, otherwise number of times
        """
        imgs = [Image.fromarray(f) for f in frames]
        duration_ms = int(1000.0 / max(fps, 1))
        imgs[0].save(
            path, save_all=True, append_images=imgs[1:],
            duration=duration_ms, loop=loop, optimize=False,
        )

    # ---- shape rasterization (duplicate of static version's helper) ----

    def _draw_shape(self, draw_rgb, draw_seg, shape_id, cx, cy, r, color, pid):
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

    def compute_label(self, scene: STScene, kind: str = "count_modulo_K",
                       K: int = 4) -> int:
        linked = scene.linked
        if kind == "count_modulo_K":
            return len(linked) % K
        if kind == "has_pair":
            for i in range(len(linked)):
                for j in range(i + 1, len(linked)):
                    a, b = linked[i], linked[j]
                    if a.shape_id == b.shape_id and a.color_idx != b.color_idx:
                        return 1
            return 0
        if kind == "n_distinct_pairs":
            seen = {(p.shape_id, p.color_idx) for p in linked}
            return min(len(seen), K - 1)
        if kind == "has_motion_pattern":
            # 1 iff at least one primitive follows a circular or sinusoidal trajectory
            for p in linked:
                if p.trajectory.kind in ("circular", "sinusoidal"):
                    return 1
            return 0
        if kind == "n_distinct_motion_kinds":
            kinds = {p.trajectory.kind for p in linked}
            return min(len(kinds), K - 1)
        if kind == "has_fast_motion":
            # 1 iff any primitive has frequency > 2.0 cycles/video
            for p in linked:
                if abs(p.trajectory.params.get("freq", 0.0)) > 2.0:
                    return 1
            return 0
        if kind == "freq_band_count":
            # Number of distinct frequency bands present in the scene.
            # Useful for testing whether SSL detects multi-Hz simultaneously.
            # Bands: {0}=static/linear, (0, 1]=slow, (1, 3]=medium, >3=fast.
            bands = set()
            for p in linked:
                f = abs(p.trajectory.params.get("freq", 0.0))
                if f == 0.0:    bands.add(0)
                elif f <= 1.0:  bands.add(1)
                elif f <= 3.0:  bands.add(2)
                else:           bands.add(3)
            return min(len(bands), K - 1)
        raise ValueError(kind)


# ===========================================================================
# Correspondence helpers — three kinds
# ===========================================================================

def cross_view_pairs_at_time(video: dict, t_idx: int) -> np.ndarray:
    """Cross-view correspondences at a single frame index t_idx.

    Returns (M, 2, 2): for each primitive visible in both views at frame t,
    its (x, y) in view A and (x, y) in view B.
    """
    A = video["view_A"]; B = video["view_B"]
    ids_A = set(int(i) for i, v in zip(A["ids"], A["vis"][t_idx]) if v)
    ids_B = set(int(i) for i, v in zip(B["ids"], B["vis"][t_idx]) if v)
    common = sorted(ids_A & ids_B)
    if not common: return np.zeros((0, 2, 2), dtype=np.float32)
    out = []
    for pid in common:
        i_A = int(np.where(A["ids"] == pid)[0][0])
        i_B = int(np.where(B["ids"] == pid)[0][0])
        out.append([A["kpts"][t_idx, i_A], B["kpts"][t_idx, i_B]])
    return np.array(out, dtype=np.float32)


def cross_time_pairs_within_view(video: dict, view: str,
                                  t_idx_1: int, t_idx_2: int) -> np.ndarray:
    """Same primitive at two times within one view.

    Returns (M, 2, 2): for each primitive visible at both times,
    its (x, y) at frame t1 and (x, y) at frame t2.
    """
    v = video[f"view_{view}"]
    ids_1 = set(int(i) for i, b in zip(v["ids"], v["vis"][t_idx_1]) if b)
    ids_2 = set(int(i) for i, b in zip(v["ids"], v["vis"][t_idx_2]) if b)
    common = sorted(ids_1 & ids_2)
    if not common: return np.zeros((0, 2, 2), dtype=np.float32)
    out = []
    for pid in common:
        i = int(np.where(v["ids"] == pid)[0][0])
        out.append([v["kpts"][t_idx_1, i], v["kpts"][t_idx_2, i]])
    return np.array(out, dtype=np.float32)


def trajectories_for_view(video: dict, view: str) -> dict:
    """For each pid, return its full (T, 2) keypoint trajectory in this view
    (NaN where not visible). Useful for plotting motion paths or evaluating
    tracking."""
    v = video[f"view_{view}"]
    out = {}
    for i, pid in enumerate(v["ids"]):
        out[int(pid)] = v["kpts"][:, i].copy()
    return out
