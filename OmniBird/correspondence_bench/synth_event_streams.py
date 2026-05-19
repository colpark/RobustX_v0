"""Synthetic Event Streams — sparse multimodal correspondence dataset.

GENERATIVE MODEL
----------------
A scene contains N "linked events". Each linked event has a latent
identity i ∈ {0, .., N-1}, a base location p_i ∈ [-1, 1]^2, a time
τ_i ∈ [0, 1], and a feature vector f_i.

The event manifests in TWO sparse modalities, with controllable mismatch:

  Modality A: event with position p_i^A = p_i + δ_i^A and attribute g_A(f_i)
  Modality B: event with position p_i^B = T_B(p_i) + δ_i^B and attribute g_B(f_i)

T_B is a known per-scene cross-modal transform (linear or weakly nonlinear).
δ_i^A, δ_i^B are small per-event noise offsets.
g_A, g_B are modality-specific attribute encoders (different channel
counts and possibly different statistics).

Around the linked events, M distractor events are sampled in each
modality. The model has to (a) discover the cross-modal point
correspondences and (b) ignore the distractors.

LABEL
-----
The scene label φ(z) depends on the SET of linked-event latents. Built-in
options:
    "count_modulo_K"        — N linked events mod K
    "majority_feature"      — most common feature class among linked
    "n_distinct_features"   — count of distinct linked-feature classes
    "transform_class"       — coarse class of the cross-modal transform
                              (e.g. rotation bucket, identity vs rotated)

OUTPUT
------
For each scene we return:
    A: dict with
       'pos':    (K_A, 2)  spatial positions
       'time':   (K_A,)    timestamps
       'attrs':  (K_A, D_A) attribute features
       'src':    (K_A,)    source ID  (i ∈ {0..N-1} for linked,
                                       -1 for distractor)
    B: same shape with possibly different D_B
    transform: dict describing T_B (so we can probe whether SSL learns it)
    label: int
    knobs: dict
"""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Optional, Union
import math

import numpy as np


# ===========================================================================
# Difficulty operating points (parallel structure to linked_primitives)
# ===========================================================================

OPERATING_POINTS = {
    "easy": dict(
        n_linked=8, n_features=2,
        n_distractors_A=0, n_distractors_B=0,
        transform="identity",
        pos_jitter=0.005,
        time_jitter=0.005,
        attr_dim_A=4, attr_dim_B=4,
        attr_corr=1.0,           # how aligned the two modalities' attributes are
        time_total=1.0,
    ),
    "basic": dict(
        n_linked=32, n_features=4,
        n_distractors_A=4, n_distractors_B=4,
        transform="rotation",
        pos_jitter=0.01,
        time_jitter=0.01,
        attr_dim_A=8, attr_dim_B=8,
        attr_corr=0.9,
        time_total=1.0,
    ),
    "hard": dict(
        n_linked=128, n_features=8,
        n_distractors_A=32, n_distractors_B=32,
        transform="affine",
        pos_jitter=0.02,
        time_jitter=0.02,
        attr_dim_A=8, attr_dim_B=12,    # different per modality
        attr_corr=0.7,
        time_total=1.0,
    ),
    "extreme": dict(
        n_linked=256, n_features=12,
        n_distractors_A=128, n_distractors_B=128,
        transform="nonlinear",
        pos_jitter=0.03,
        time_jitter=0.04,
        attr_dim_A=12, attr_dim_B=16,
        attr_corr=0.5,
        time_total=1.0,
    ),
    "adversarial": dict(
        n_linked=256, n_features=12,
        n_distractors_A=192, n_distractors_B=192,
        transform="nonlinear",
        pos_jitter=0.03,
        time_jitter=0.05,
        attr_dim_A=12, attr_dim_B=16,
        attr_corr=0.3,
        time_total=1.0,
    ),
}


# ===========================================================================
# Scene representation
# ===========================================================================

@dataclass
class EventStreamScene:
    A_pos: np.ndarray      # (K_A, 2)
    A_time: np.ndarray     # (K_A,)
    A_attrs: np.ndarray    # (K_A, D_A)
    A_src: np.ndarray      # (K_A,)
    B_pos: np.ndarray
    B_time: np.ndarray
    B_attrs: np.ndarray
    B_src: np.ndarray
    transform: dict
    label: int = 0
    knobs: dict = field(default_factory=dict)
    seed: int = 0


# ===========================================================================
# Cross-modal transforms T_B
# ===========================================================================

def _build_transform(kind: str, rng: np.random.RandomState) -> dict:
    if kind == "identity":
        M = np.eye(2, dtype=np.float32)
        return dict(kind="identity", matrix=M, bias=np.zeros(2, dtype=np.float32))
    if kind == "rotation":
        theta = float(rng.uniform(-math.pi, math.pi))
        c, s = math.cos(theta), math.sin(theta)
        M = np.array([[c, -s], [s, c]], dtype=np.float32)
        return dict(kind="rotation", theta=theta, matrix=M,
                    bias=np.zeros(2, dtype=np.float32))
    if kind == "affine":
        theta = float(rng.uniform(-math.pi, math.pi))
        c, s = math.cos(theta), math.sin(theta)
        scale = np.diag(rng.uniform(0.7, 1.3, size=2)).astype(np.float32)
        R = np.array([[c, -s], [s, c]], dtype=np.float32)
        M = R @ scale
        bias = rng.uniform(-0.2, 0.2, size=2).astype(np.float32)
        return dict(kind="affine", matrix=M, bias=bias)
    if kind == "nonlinear":
        # Identity-perturbed-by-radial-warp: T(p) = p + ε · p · ‖p‖²
        eps = float(rng.uniform(0.15, 0.45)) * float(rng.choice([-1, 1]))
        theta = float(rng.uniform(-math.pi, math.pi))
        c, s = math.cos(theta), math.sin(theta)
        R = np.array([[c, -s], [s, c]], dtype=np.float32)
        return dict(kind="nonlinear", eps=eps, R=R,
                    matrix=R, bias=np.zeros(2, dtype=np.float32))
    raise ValueError(kind)


def _apply_transform(pos: np.ndarray, T: dict) -> np.ndarray:
    if T["kind"] in ("identity", "rotation", "affine"):
        return pos @ T["matrix"].T + T["bias"]
    if T["kind"] == "nonlinear":
        # nonlinear: p' = R · (p + ε · p · ‖p‖²)
        r2 = (pos * pos).sum(-1, keepdims=True)
        warped = pos + T["eps"] * pos * r2
        return warped @ T["R"].T
    raise ValueError(T["kind"])


# ===========================================================================
# Generator
# ===========================================================================

class SyntheticEventStreamsGenerator:
    def __init__(self,
                 operating_point: Union[str, dict] = "basic",
                 base_seed: int = 0):
        if isinstance(operating_point, str):
            self.knobs = dict(OPERATING_POINTS[operating_point])
            self.knobs["_name"] = operating_point
        else:
            self.knobs = dict(operating_point)
            self.knobs.setdefault("_name", "custom")
        self.base_seed = base_seed

    def _modality_attr(self, f: np.ndarray, D_target: int,
                        corr: float, side: str, rng: np.random.RandomState) -> np.ndarray:
        """Produce a modality-specific attribute vector for latent feature f.
        `corr` controls how aligned A and B attributes are: corr=1 means
        same encoder, corr=0 means random."""
        D_in = f.shape[-1]
        # Use a fixed (seed-dependent) projection per side
        seed_offset = 1001 if side == "A" else 2002
        proj_rng = np.random.RandomState(self.base_seed * 7 + seed_offset)
        # Stable linear encoder shared across scenes
        W = proj_rng.normal(size=(D_in, D_target)).astype(np.float32) / math.sqrt(D_in)
        base = f @ W
        # Add per-modality random noise of strength (1 - corr)
        noise = rng.normal(size=base.shape).astype(np.float32) * (1.0 - corr)
        out = corr * base + noise
        # Mild element-wise nonlinearity to make the two encoders not perfectly
        # linearly identifiable from each other
        if side == "B":
            out = np.tanh(out * 1.2)
        return out

    def sample_scene(self, seed: Optional[int] = None,
                     label_kind: str = "count_modulo_K", K: int = 4) -> EventStreamScene:
        rng = np.random.RandomState(seed if seed is not None else self.base_seed)
        k = self.knobs
        N = k["n_linked"]
        F = k["n_features"]

        # Linked-event latents
        feat_class = rng.randint(0, F, size=N)
        f_table = rng.normal(size=(F, 4)).astype(np.float32)        # (F, 4)
        f_per_event = f_table[feat_class] + rng.normal(scale=0.1, size=(N, 4)).astype(np.float32)

        p_latent = rng.uniform(-0.9, 0.9, size=(N, 2)).astype(np.float32)
        tau = rng.uniform(0.0, k["time_total"], size=N).astype(np.float32)

        T = _build_transform(k["transform"], rng)

        # Per-event jitter in each modality
        p_A = p_latent + rng.normal(scale=k["pos_jitter"], size=(N, 2)).astype(np.float32)
        p_B = _apply_transform(p_latent, T) + rng.normal(scale=k["pos_jitter"], size=(N, 2)).astype(np.float32)
        t_A = tau + rng.normal(scale=k["time_jitter"], size=N).astype(np.float32)
        t_B = tau + rng.normal(scale=k["time_jitter"], size=N).astype(np.float32)

        attrs_A = self._modality_attr(f_per_event, k["attr_dim_A"], k["attr_corr"], "A", rng)
        attrs_B = self._modality_attr(f_per_event, k["attr_dim_B"], k["attr_corr"], "B", rng)

        # Source IDs for linked events (i.e. their latent index)
        src_A = np.arange(N, dtype=np.int64)
        src_B = np.arange(N, dtype=np.int64)

        # Distractors per modality
        M_A = k["n_distractors_A"]
        M_B = k["n_distractors_B"]

        def random_attrs(M, D, rng_):
            return rng_.normal(size=(M, D)).astype(np.float32)

        dpos_A = rng.uniform(-1.0, 1.0, size=(M_A, 2)).astype(np.float32)
        dt_A   = rng.uniform(0.0, k["time_total"], size=M_A).astype(np.float32)
        dattr_A= random_attrs(M_A, k["attr_dim_A"], rng)
        dsrc_A = -np.ones(M_A, dtype=np.int64)

        dpos_B = rng.uniform(-1.0, 1.0, size=(M_B, 2)).astype(np.float32)
        dt_B   = rng.uniform(0.0, k["time_total"], size=M_B).astype(np.float32)
        dattr_B= random_attrs(M_B, k["attr_dim_B"], rng)
        dsrc_B = -np.ones(M_B, dtype=np.int64)

        # Concatenate linked + distractors, then random-shuffle per modality
        A_pos   = np.concatenate([p_A, dpos_A], axis=0)
        A_time  = np.concatenate([t_A, dt_A],   axis=0)
        A_attrs = np.concatenate([attrs_A, dattr_A], axis=0)
        A_src   = np.concatenate([src_A, dsrc_A], axis=0)

        B_pos   = np.concatenate([p_B, dpos_B], axis=0)
        B_time  = np.concatenate([t_B, dt_B],   axis=0)
        B_attrs = np.concatenate([attrs_B, dattr_B], axis=0)
        B_src   = np.concatenate([src_B, dsrc_B], axis=0)

        # Independent random permutations per modality (so order ≠ correspondence)
        perm_A = rng.permutation(len(A_pos))
        perm_B = rng.permutation(len(B_pos))
        A_pos, A_time, A_attrs, A_src = A_pos[perm_A], A_time[perm_A], A_attrs[perm_A], A_src[perm_A]
        B_pos, B_time, B_attrs, B_src = B_pos[perm_B], B_time[perm_B], B_attrs[perm_B], B_src[perm_B]

        label = self._compute_label(label_kind, feat_class, T, K)
        return EventStreamScene(
            A_pos=A_pos, A_time=A_time, A_attrs=A_attrs, A_src=A_src,
            B_pos=B_pos, B_time=B_time, B_attrs=B_attrs, B_src=B_src,
            transform=T,
            label=int(label),
            knobs=dict(self.knobs),
            seed=int(seed if seed is not None else self.base_seed),
        )

    def _compute_label(self, kind: str, feat_class: np.ndarray, T: dict,
                        K: int) -> int:
        if kind == "count_modulo_K":
            return int(len(feat_class) % K)
        if kind == "majority_feature":
            return int(np.bincount(feat_class).argmax())
        if kind == "n_distinct_features":
            return min(int(np.unique(feat_class).shape[0]), K - 1)
        if kind == "transform_class":
            mapping = {"identity": 0, "rotation": 1, "affine": 2, "nonlinear": 3}
            return mapping.get(T["kind"], 0)
        raise ValueError(kind)


# ===========================================================================
# Correspondence ground-truth helpers
# ===========================================================================

def correspondence_indices(scene: EventStreamScene) -> np.ndarray:
    """Return (M, 2) array of (idx_in_A, idx_in_B) pairs that share a source.

    Only returns linked (source >= 0) pairs.
    """
    out = []
    src_to_A = {int(s): i for i, s in enumerate(scene.A_src) if s >= 0}
    src_to_B = {int(s): i for i, s in enumerate(scene.B_src) if s >= 0}
    common = sorted(set(src_to_A) & set(src_to_B))
    for s in common:
        out.append([src_to_A[s], src_to_B[s]])
    if not out:
        return np.zeros((0, 2), dtype=np.int64)
    return np.array(out, dtype=np.int64)
