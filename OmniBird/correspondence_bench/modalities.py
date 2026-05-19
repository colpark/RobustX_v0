"""Modalities — sensor-style visual representations of a rendered scene.

DESIGN
======

Different sensors observe the same 3D world in very different ways.
This module provides three callable "modalities" that take a rendered
scene (RGB + depth + seg) and return a sensor-style observation:

  * `DepthCameraModality`   — pseudo-depth-camera output. Each pixel is
                              colormapped depth (viridis); background
                              dark. Mimics a stereo / structured-light
                              depth sensor like Kinect / RealSense.
  * `LiDARModality`         — pseudo-LiDAR output. Sparse sampling of
                              depth (~10–30% of foreground pixels become
                              "returns"), colored by depth (plasma).
                              The rest of the image is near-black.
  * `InfraredModality`      — pseudo-thermal output. Each primitive has
                              a deterministic pseudo-"temperature" based
                              on its (shape_id, color_idx); the image
                              is the temperature heatmap (inferno) with
                              a small Gaussian blur to simulate IR
                              optics resolution.

The three modalities are designed to be **visually distinct** so
side-by-side plots make the sensor-fusion problem obvious. They're
also **semantically distinct**: depth is dense-but-no-identity, LiDAR
is sparse-with-depth, IR encodes a per-primitive scalar that can match
some primitives across modalities and confuse others.

INTERFACE
=========

Each modality is a callable:

    modality(render_out, rng=None) -> render_out_modified

The input dict must include a `depth` field (per-pixel z-distance from
the camera), in addition to the standard `rgb / seg / kpts / vis / ids`.
The multiview generator's `_render_one` automatically adds `depth`.

Each modality returns a NEW dict with:

  * `rgb`    modified to reflect the sensor's visual output
  * `seg`    optionally modified — LiDAR drops seg outside the sparse
             return set (because the sensor has no observation there)
  * other fields unchanged

NOTES
=====

* Modalities are NOT augmenters in the sense of `augmenters.py`. They
  are part of how the camera SEES, not what's applied on top. The two
  axes compose naturally: render → modality → augmenter (noise on top).
* For deterministic LiDAR sub-sampling per (camera, scene), the
  multiview generator passes a per-view rng seed.
"""
from __future__ import annotations
from typing import Optional, Union
import numpy as np


# ===========================================================================
# Helpers — built-in colormaps so we don't depend on matplotlib at import
# ===========================================================================

def _viridis_lookup(t: np.ndarray) -> np.ndarray:
    """Viridis colormap as (N, 3) uint8.

    Input `t` is in [0, 1]; output shape matches `t.shape + (3,)`.
    Polynomial approximation (no matplotlib dependency).
    """
    t = np.clip(t, 0.0, 1.0)
    r = 0.267 - 0.105*t + 4.07*t**2 - 7.06*t**3 + 3.69*t**4
    g = 0.005 + 1.404*t - 0.396*t**2 - 0.085*t**3 - 0.139*t**4
    b = 0.329 + 1.382*t - 4.69*t**2 + 5.04*t**3 - 1.97*t**4
    rgb = np.stack([r, g, b], axis=-1)
    rgb = np.clip(rgb, 0.0, 1.0)
    return (rgb * 255).astype(np.uint8)


def _plasma_lookup(t: np.ndarray) -> np.ndarray:
    """Plasma-like colormap as (..., 3) uint8."""
    t = np.clip(t, 0.0, 1.0)
    r = 0.05 + 2.6*t - 4.0*t**2 + 2.7*t**3
    g = 0.03 + 0.4*t + 1.7*t**2 - 1.1*t**3
    b = 0.53 + 0.9*t - 4.7*t**2 + 3.3*t**3
    rgb = np.stack([r, g, b], axis=-1)
    rgb = np.clip(rgb, 0.0, 1.0)
    return (rgb * 255).astype(np.uint8)


def _inferno_lookup(t: np.ndarray) -> np.ndarray:
    """Inferno-like colormap as (..., 3) uint8 (for IR)."""
    t = np.clip(t, 0.0, 1.0)
    r = 0.001 + 2.6*t - 3.1*t**2 + 1.5*t**3
    g = 0.001 + 0.18*t + 1.65*t**2 - 0.85*t**3
    b = 0.05 + 1.65*t - 5.0*t**2 + 4.3*t**3 - 1.0*t**4
    rgb = np.stack([r, g, b], axis=-1)
    rgb = np.clip(rgb, 0.0, 1.0)
    return (rgb * 255).astype(np.uint8)


def _box_blur_2d(arr: np.ndarray, kernel: int = 3) -> np.ndarray:
    """Simple separable box blur of a 2D (H, W) array. No scipy dep."""
    if kernel <= 1:
        return arr
    pad = kernel // 2
    padded = np.pad(arr, pad, mode="edge")
    out = np.zeros_like(arr, dtype=np.float32)
    # Horizontal pass
    for dx in range(kernel):
        out += padded[pad:pad + arr.shape[0], dx:dx + arr.shape[1]]
    out /= kernel
    # Vertical pass
    padded = np.pad(out, ((pad, pad), (0, 0)), mode="edge")
    tmp = np.zeros_like(arr, dtype=np.float32)
    for dy in range(kernel):
        tmp += padded[dy:dy + arr.shape[0], :]
    return tmp / kernel


# ===========================================================================
# Base
# ===========================================================================

class Modality:
    """Base class for sensor-style modality transforms.

    Subclasses implement `__call__(self, render_out, rng=None) -> dict`.
    """
    name = "base"

    def __call__(self, render_out: dict, rng=None) -> dict:
        raise NotImplementedError

    @staticmethod
    def _get_rng(rng):
        if rng is None: return np.random.RandomState()
        if isinstance(rng, (int, np.integer)): return np.random.RandomState(int(rng))
        return rng

    @staticmethod
    def _normalize_depth(depth: np.ndarray):
        """Return (normalized_depth in [0, 1], foreground_mask).

        Background pixels (where depth is non-finite) get d_norm = 0 and
        mask = False.
        """
        mask = np.isfinite(depth)
        d = depth.copy().astype(np.float32)
        if mask.any():
            dmin = float(d[mask].min())
            dmax = float(d[mask].max())
            denom = max(dmax - dmin, 1e-6)
            d_norm = (d - dmin) / denom
            d_norm[~mask] = 0.0
        else:
            d_norm = np.zeros_like(d, dtype=np.float32)
        # Invert: typically near = bright, far = dark. Without inversion,
        # near (small z) maps to dark. Flip so near = bright (colored).
        d_norm = 1.0 - d_norm
        d_norm[~mask] = 0.0
        return d_norm, mask


# ===========================================================================
# Depth camera
# ===========================================================================

class DepthCameraModality(Modality):
    """Pseudo depth-camera modality (e.g. Kinect, RealSense).

    Each pixel is colormapped by its depth (z-distance from the camera).
    Background is dark navy. The output is dense — every foreground
    pixel has a depth value, with smooth colormap gradient inside each
    primitive.

    Parameters
    ----------
    background_rgb : (R, G, B) uint8
        Color of the empty background. Default dark navy.
    invert : bool
        If True (default), near = bright / yellow, far = darker.
        Realistic depth-camera viz: brighter for closer objects.
    """
    name = "depth"

    def __init__(self,
                 background_rgb=(15, 15, 30),
                 invert: bool = True):
        self.background_rgb = tuple(int(v) for v in background_rgb)
        self.invert = bool(invert)

    def __call__(self, render_out, rng=None):
        out = {**render_out}
        depth = render_out["depth"]
        d_norm, mask = self._normalize_depth(depth)
        if not self.invert:
            d_norm = 1.0 - d_norm
            d_norm[~mask] = 0.0
        rgb = _viridis_lookup(d_norm)
        rgb[~mask] = np.array(self.background_rgb, dtype=np.uint8)
        out["rgb"] = rgb
        return out


# ===========================================================================
# LiDAR
# ===========================================================================

class LiDARModality(Modality):
    """Pseudo-LiDAR modality.

    Sparse returns: a random `keep_fraction` of foreground pixels become
    "LiDAR points", colored by depth on a plasma colormap. The rest of
    the image is near-black (sensor saw nothing).

    Real LiDAR scans at a regular angular grid; we approximate this with
    uniform random pixel sub-sampling, which gives a similar visual
    impression and the same downstream consequence: most of the FOV has
    no measurement at any given timestep.

    The `seg` is updated to -1 outside the sparse return set, because
    the model doesn't observe segmentation where the sensor returned no
    point.

    Parameters
    ----------
    keep_fraction : float in (0, 1]
        Fraction of foreground pixels that become returns.
        Typical real LiDAR is much sparser, but ~0.15 looks recognizable.
    background_rgb : (R, G, B) uint8
        Color of pixels with no return (dark grey/black).
    """
    name = "lidar"

    def __init__(self,
                 keep_fraction: float = 0.15,
                 background_rgb=(8, 10, 14)):
        if not 0 < keep_fraction <= 1:
            raise ValueError("keep_fraction must be in (0, 1]")
        self.keep_fraction = float(keep_fraction)
        self.background_rgb = tuple(int(v) for v in background_rgb)

    def __call__(self, render_out, rng=None):
        out = {**render_out}
        rng = self._get_rng(rng)
        depth = render_out["depth"]
        d_norm, fg_mask = self._normalize_depth(depth)
        # Sample returns inside foreground mask
        u = rng.uniform(size=depth.shape)
        keep = (u < self.keep_fraction) & fg_mask
        # Colored returns
        rgb_full = _plasma_lookup(d_norm)
        rgb = np.zeros_like(rgb_full, dtype=np.uint8)
        rgb[..., 0] = self.background_rgb[0]
        rgb[..., 1] = self.background_rgb[1]
        rgb[..., 2] = self.background_rgb[2]
        rgb[keep] = rgb_full[keep]
        out["rgb"] = rgb
        # Sensor observation: seg only where the sensor saw a return
        if "seg" in out:
            new_seg = np.full_like(out["seg"], -1)
            new_seg[keep] = out["seg"][keep]
            out["seg"] = new_seg
        return out


# ===========================================================================
# Infrared
# ===========================================================================

class InfraredModality(Modality):
    """Pseudo-infrared / thermal-camera modality.

    Each primitive carries a pseudo-"temperature" derived deterministically
    from its `(shape_id, color_idx)` — different shapes/materials emit
    different IR signatures. The background is ambient (low temperature).
    A small Gaussian blur is applied to the temperature field to simulate
    the modest spatial resolution of typical IR optics. The result is
    colormapped using an inferno-like palette.

    Importantly: different primitives can end up at the **same**
    temperature (the lookup is many-to-one), so an IR sensor cannot
    always distinguish them — exactly the kind of cross-modal ambiguity
    that makes multi-sensor fusion useful.

    Parameters
    ----------
    blur_kernel : int (odd)
        Box-blur kernel size for the temperature field. 1 disables blur.
        Default 3 → modest blur.
    background_temp : float in [0, 1]
        Ambient temperature for empty pixels. Default 0.25 → cool.
    """
    name = "infrared"

    # Pre-built temperature lookup. Maps (shape_id, color_idx) → temp.
    # Designed so:
    #   - Different shape_ids tend to differ in temperature
    #   - Different color_idxes within a shape add modest variation
    #   - Some collisions are guaranteed (different primitives, same temp)
    _BASE_TEMP_PER_SHAPE = np.array([
        0.45, 0.60, 0.75, 0.55, 0.80, 0.50, 0.65, 0.40, 0.70, 0.85
    ], dtype=np.float32)

    def __init__(self,
                 blur_kernel: int = 3,
                 background_temp: float = 0.25):
        if blur_kernel < 1 or blur_kernel % 2 == 0:
            raise ValueError("blur_kernel must be an odd positive integer")
        self.blur_kernel = int(blur_kernel)
        self.background_temp = float(background_temp)

    @classmethod
    def temperature_for(cls, shape_id: int, color_idx: int) -> float:
        """Public-facing helper: what temperature does this primitive emit?"""
        base = cls._BASE_TEMP_PER_SHAPE[shape_id % len(cls._BASE_TEMP_PER_SHAPE)]
        # color_idx mod 3 nudges +/- 0.05; not enough to fully resolve identity
        delta = ((color_idx % 3) - 1) * 0.05
        return float(np.clip(base + delta, 0.0, 1.0))

    def __call__(self, render_out, rng=None):
        out = {**render_out}
        seg = render_out["seg"]
        primitives_meta = render_out.get("primitives_meta")
        # primitives_meta is a dict pid -> (shape_id, color_idx) emitted by
        # the multiview generator. If not present, fall back to a hash of pid.
        H, W = seg.shape
        temp = np.full((H, W), self.background_temp, dtype=np.float32)
        if primitives_meta:
            for pid, (shape_id, color_idx) in primitives_meta.items():
                t = self.temperature_for(int(shape_id), int(color_idx))
                temp[seg == pid] = t
        else:
            for pid in np.unique(seg):
                if int(pid) < 0: continue
                t = ((int(pid) * 37) % 13) / 13.0 * 0.6 + 0.3
                temp[seg == pid] = t
        # Blur — simulates IR optics
        if self.blur_kernel > 1:
            temp = _box_blur_2d(temp, kernel=self.blur_kernel)
        rgb = _inferno_lookup(temp)
        out["rgb"] = rgb
        return out


# ===========================================================================
# Lookup by name
# ===========================================================================

MODALITY_REGISTRY = {
    "depth":    DepthCameraModality,
    "lidar":    LiDARModality,
    "infrared": InfraredModality,
}


def make_modality(name: str, **kwargs) -> Modality:
    """Construct a modality by name. `name` ∈ {'depth','lidar','infrared'}."""
    if name not in MODALITY_REGISTRY:
        raise ValueError(f"unknown modality {name!r}; expected one of {list(MODALITY_REGISTRY)}")
    return MODALITY_REGISTRY[name](**kwargs)
