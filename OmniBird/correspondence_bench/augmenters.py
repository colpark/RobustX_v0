"""Augmenters — post-rendering modifications to scene observations.

DESIGN CONTRACT
===============

Every augmenter is a callable that takes a rendered output dict and
returns a modified copy. The renderer produces **clean, deterministic**
outputs; all observation-noise, sparsity, occlusion, and view-limiting
transformations live in augmenters that operate on top.

This separation is deliberate:

  * **Difficulty** (the operating-point knobs in linked_primitives*.py)
    controls the *latent scene*: how many primitives, how irregular,
    how multi-scale, how fast they move, how different the cameras are.
  * **Noise / sparsity / occlusion** (augmenters in this file) control
    the *observation channel*: how the rendered images are corrupted
    before reaching the model.

These two axes are independent. You can run any operating point with
or without any augmenter, and you can mix-and-match augmenters via
`AugmenterPipeline`. This lets you ask precise scientific questions
like "does difficulty X interact with noise level Y?" without coupling
the knobs.

API
---

An augmenter is a callable:

    augmenter(out_dict, rng=None) -> out_dict_modified

where `out_dict` is the render output (`render()`, `render_frame()` or
one frame of `render_video_pair()`):

    {
        "rgb":  (H, W, 3) uint8 image
        "seg":  (H, W) int32 per-pixel primitive ID (-1 background)
        "kpts": (N, 2) float32 projected centers
        "vis":  (N,) bool visibility flags
        "ids":  (N,) int32 stable primitive IDs
    }

Augmenters should return a NEW dict (not mutate the input). Required
behaviour:

  * Modify `rgb` (and possibly `seg`, `vis`) consistently with the
    augmentation semantics — e.g. RandomSubsample drops pixels in both
    rgb and seg; LimitedFOV masks pixels outside the FOV in both.
  * Leave `kpts` and `ids` unchanged. These describe the *latent*
    scene, which the augmenter does not modify.
  * Be deterministic given the same `rng` (or seed integer).

Available augmenters
--------------------

  * IdentityAugmenter          — pass-through baseline
  * GaussianNoiseAugmenter     — IID Gaussian on RGB
  * SaltPepperNoiseAugmenter   — random salt-and-pepper corruption
  * RandomSubsampleAugmenter   — drop a fraction of pixels uniformly
  * CenterOcclusionAugmenter   — mask out a centered square region
  * LimitedFOVAugmenter        — keep only a sub-bbox of the image
  * AugmenterPipeline          — chain multiple augmenters in order

All augmenters work on both static (Dataset A) and per-frame video
outputs (Dataset B). For Dataset B you typically apply the augmenter
once per frame, varying `rng` across frames (so noise is temporally
independent) or fixing it (so corruption is temporally consistent —
e.g. the same occlusion at every frame).
"""
from __future__ import annotations
from typing import Optional, Union, Sequence

import numpy as np


# ===========================================================================
# Base class
# ===========================================================================

class Augmenter:
    """Abstract base class for post-render augmenters.

    Subclasses must implement `__call__(self, out, rng=None) -> dict`.

    The `_get_rng` helper resolves the `rng` argument into a usable
    `np.random.RandomState`:

      * None                   → fresh RandomState (non-deterministic)
      * int                    → RandomState seeded with that int
      * np.random.RandomState  → used directly
    """
    def __call__(self, out: dict, rng=None) -> dict:
        raise NotImplementedError

    @staticmethod
    def _get_rng(rng):
        if rng is None:
            return np.random.RandomState()
        if isinstance(rng, (int, np.integer)):
            return np.random.RandomState(int(rng))
        return rng


# ===========================================================================
# Identity
# ===========================================================================

class IdentityAugmenter(Augmenter):
    """Pass-through. Returns a shallow copy of the input dict.

    Useful as the "noise = 0" baseline in benchmark sweeps. Also useful
    when assembling pipelines that conditionally include augmenters.
    """
    def __call__(self, out, rng=None):
        return {**out}


# ===========================================================================
# Noise
# ===========================================================================

class GaussianNoiseAugmenter(Augmenter):
    """Add IID Gaussian noise to RGB pixels.

    Adds `N(0, sigma * 255)` to each (pixel, channel) and clips to [0, 255].
    A typical noise scan in benchmark sweeps:

        sigma = 0          (no noise)
        sigma = 0.02       (mild — barely visible)
        sigma = 0.05       (noticeable)
        sigma = 0.10       (heavy)
        sigma = 0.20       (severe — visible distortion)

    Parameters
    ----------
    sigma : float
        Std of the Gaussian noise expressed as a fraction of 255.
        At sigma=0.05, the noise has std ≈ 12.75 in grey levels.

    Modifies: rgb. Leaves seg / kpts / vis / ids unchanged.
    """
    def __init__(self, sigma: float):
        if sigma < 0:
            raise ValueError(f"sigma must be ≥ 0, got {sigma}")
        self.sigma = float(sigma)

    def __call__(self, out, rng=None):
        if self.sigma == 0:
            return {**out}
        rng = self._get_rng(rng)
        rgb = out["rgb"].astype(np.float32)
        noise = rng.normal(0.0, self.sigma * 255.0, rgb.shape)
        rgb = np.clip(rgb + noise, 0, 255).astype(np.uint8)
        return {**out, "rgb": rgb}


class SaltPepperNoiseAugmenter(Augmenter):
    """Set each pixel to all-black or all-white with probability p.

    Models impulsive noise from sensor dropouts / hot pixels. The
    fraction `p` is split 50/50 between salt (255) and pepper (0).

    Parameters
    ----------
    p : float in [0, 1]
        Probability of corruption per pixel.

    Modifies: rgb. Leaves seg / kpts / vis / ids unchanged.
    """
    def __init__(self, p: float):
        if not 0 <= p <= 1:
            raise ValueError(f"p must be in [0, 1], got {p}")
        self.p = float(p)

    def __call__(self, out, rng=None):
        if self.p == 0:
            return {**out}
        rng = self._get_rng(rng)
        rgb = out["rgb"].copy()
        H, W = rgb.shape[:2]
        u = rng.uniform(size=(H, W))
        salt = u < self.p / 2
        pepper = (u >= self.p / 2) & (u < self.p)
        rgb[salt] = 255
        rgb[pepper] = 0
        return {**out, "rgb": rgb}


# ===========================================================================
# Sparsity — drop pixels
# ===========================================================================

class RandomSubsampleAugmenter(Augmenter):
    """Drop pixels uniformly at random; set them to a background color.

    Simulates **sparse-sampling modalities** where only a subset of
    spatial locations have observations (point clouds, sparse pixel
    pools, event cameras with low activity).

    The set of kept pixels is determined by a uniform-random mask of
    shape `(H, W)`. The same mask is applied to all RGB channels and
    optionally to `seg` (so dropped pixels report background ID `-1`
    in segmentation ground truth too).

    Parameters
    ----------
    keep_fraction : float in (0, 1]
        Fraction of pixels to KEEP. 0.4 → drop 60% of pixels.
    background : (R, G, B) tuple of uint8
        Color used for dropped pixels. Default light grey matches the
        renderer's default background.
    drop_seg : bool
        If True (default), set seg to -1 for dropped pixels. If False,
        keep the original seg labels (the model still "sees" the
        ground-truth labels even where rgb was dropped — useful only
        for sanity tests).

    Modifies: rgb, seg (optionally). Leaves kpts / vis / ids unchanged.
    """
    def __init__(self, keep_fraction: float,
                 background=(245, 245, 245),
                 drop_seg: bool = True):
        if not 0 < keep_fraction <= 1:
            raise ValueError(f"keep_fraction must be in (0, 1], got {keep_fraction}")
        self.keep_fraction = float(keep_fraction)
        self.background = tuple(int(b) for b in background)
        self.drop_seg = bool(drop_seg)

    def __call__(self, out, rng=None):
        if self.keep_fraction >= 1.0:
            return {**out}
        rng = self._get_rng(rng)
        rgb = out["rgb"].copy()
        H, W = rgb.shape[:2]
        keep = rng.uniform(size=(H, W)) < self.keep_fraction
        drop = ~keep
        for c in range(3):
            rgb[..., c][drop] = self.background[c]
        new_out = {**out, "rgb": rgb}
        if self.drop_seg and "seg" in out:
            seg = out["seg"].copy()
            seg[drop] = -1
            new_out["seg"] = seg
        return new_out


# ===========================================================================
# Sparsity — occlude
# ===========================================================================

class CenterOcclusionAugmenter(Augmenter):
    """Mask out a centered rectangular region (occlusion).

    Models the case where a fixed region of the image is hidden — e.g.
    a foreground object blocking a portion of the camera view, or
    intentional masking for inpainting-style SSL tasks.

    Parameters
    ----------
    occlusion_fraction : float in [0, 1)
        Side length of the occlusion square as a fraction of
        min(H, W). 0.4 → a 40% × 40% square at the center is masked.
        At 0.0 nothing is occluded (no-op).
    background : (R, G, B) tuple of uint8
        Color used inside the occlusion.

    Modifies: rgb, seg (within the occlusion box). Leaves kpts / vis /
    ids unchanged (they describe the latent scene, not the observation).
    """
    def __init__(self, occlusion_fraction: float,
                 background=(245, 245, 245)):
        if not 0 <= occlusion_fraction < 1:
            raise ValueError(f"occlusion_fraction must be in [0, 1), got {occlusion_fraction}")
        self.occlusion_fraction = float(occlusion_fraction)
        self.background = tuple(int(b) for b in background)

    def __call__(self, out, rng=None):
        if self.occlusion_fraction == 0:
            return {**out}
        rgb = out["rgb"].copy()
        H, W = rgb.shape[:2]
        side = int(self.occlusion_fraction * min(H, W))
        y0 = (H - side) // 2
        x0 = (W - side) // 2
        rgb[y0:y0 + side, x0:x0 + side, :] = self.background
        new_out = {**out, "rgb": rgb}
        if "seg" in out:
            seg = out["seg"].copy()
            seg[y0:y0 + side, x0:x0 + side] = -1
            new_out["seg"] = seg
        return new_out


# ===========================================================================
# Sparsity — limited FOV (image-space rectangular crop)
# ===========================================================================

class LimitedFOVAugmenter(Augmenter):
    """Restrict the observable region to a rectangular sub-bbox.

    Pixels outside the bbox are masked to `background`. This simulates
    a camera with a limited image-space field of view (e.g. one tile of
    a multi-camera setup, or a sensor that only operates in one
    quadrant).

    Note: this operates in IMAGE space, not 3D camera-FOV space. For
    true 3D narrow-FOV cameras with different orientations, see
    `multiview_primitives.py`.

    Parameters
    ----------
    fov_bbox : (x0, y0, x1, y1) in [0, 1]^4
        Normalized bbox: x0=0.0, y0=0.0 = top-left; x1=1.0, y1=1.0 =
        bottom-right. Default (0, 0, 1, 1) is no restriction.
    background : (R, G, B) tuple of uint8

    Modifies: rgb, seg. Leaves kpts / vis / ids unchanged.
    """
    def __init__(self, fov_bbox=(0.0, 0.0, 1.0, 1.0),
                 background=(245, 245, 245)):
        x0, y0, x1, y1 = fov_bbox
        if not (0 <= x0 < x1 <= 1 and 0 <= y0 < y1 <= 1):
            raise ValueError(f"invalid fov_bbox {fov_bbox}; need 0 ≤ x0<x1 ≤ 1 and similarly y")
        self.fov_bbox = (float(x0), float(y0), float(x1), float(y1))
        self.background = tuple(int(b) for b in background)

    def __call__(self, out, rng=None):
        H, W = out["rgb"].shape[:2]
        x0, y0, x1, y1 = self.fov_bbox
        rx0, ry0 = int(round(x0 * W)), int(round(y0 * H))
        rx1, ry1 = int(round(x1 * W)), int(round(y1 * H))
        rgb = np.zeros_like(out["rgb"])
        rgb[..., 0] = self.background[0]
        rgb[..., 1] = self.background[1]
        rgb[..., 2] = self.background[2]
        rgb[ry0:ry1, rx0:rx1, :] = out["rgb"][ry0:ry1, rx0:rx1, :]
        new_out = {**out, "rgb": rgb}
        if "seg" in out:
            seg = np.full_like(out["seg"], -1)
            seg[ry0:ry1, rx0:rx1] = out["seg"][ry0:ry1, rx0:rx1]
            new_out["seg"] = seg
        return new_out


# ===========================================================================
# Pipeline
# ===========================================================================

class AugmenterPipeline(Augmenter):
    """Chain multiple augmenters in order.

    The first augmenter sees the original render, each subsequent one
    sees the output of the previous. This composes naturally:

        pipeline = AugmenterPipeline([
            RandomSubsampleAugmenter(0.4),    # drop 60% of pixels
            GaussianNoiseAugmenter(0.05),     # add noise to what's left
            CenterOcclusionAugmenter(0.2),    # then occlude the center
        ])
        observed = pipeline(rendered, rng=42)

    Parameters
    ----------
    augmenters : sequence of Augmenter
    """
    def __init__(self, augmenters: Sequence[Augmenter]):
        self.augmenters = list(augmenters)

    def __call__(self, out, rng=None):
        rng = self._get_rng(rng)
        # Give each sub-augmenter its own stream by stepping the rng
        for i, aug in enumerate(self.augmenters):
            sub_rng = np.random.RandomState(rng.randint(0, 2**31 - 1))
            out = aug(out, rng=sub_rng)
        return out

    def __len__(self):
        return len(self.augmenters)
