"""HRR Patch JEPA — Holographic Reduced Representations for sparse-input patches.

The mathematical content lives in `bind`, `bundle`, `unbind`, and Fractional
Power Encoding (FPE) for continuous positions. The training architecture is
mathematically identical to `rope_patch_core` (HRR with FPE positions, in the
frequency domain, IS RoPE-NUDFT); for that reason the encoder/predictor
classes are aliased from `rope_patch_core`.

What HRR exposes that RoPE/NUDFT does not:
    * `bind(c, p)`  — circular convolution; mixes content with position
    * `bundle(*items)` — summation; permutation-invariant aggregation
    * `unbind(S, p)` — approximate inverse of bind; recovers an event from
                      the patch summary given its position

The unbinding operation is the conceptual win: it shows that the patch summary
can be *queried* at arbitrary positions, recovering the content that was bound
there. This is the basis for unbinding-based prediction heads (future work).
"""
from __future__ import annotations
import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


# ===========================================================================
# Pure-NumPy HRR primitives (for visualization)
# ===========================================================================

def hrr_bind_np(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Circular convolution: (a ⊛ b)_n = Σ_k a_k · b_{(n-k) mod N}.

    Equivalent (and faster) via the FFT:
        a ⊛ b = IFFT(FFT(a) · FFT(b))
    """
    A = np.fft.fft(a, axis=-1)
    B = np.fft.fft(b, axis=-1)
    return np.real(np.fft.ifft(A * B, axis=-1))


def hrr_unbind_np(s: np.ndarray, p: np.ndarray) -> np.ndarray:
    """Approximate inverse of bind: convolve with p^{-1}.

    For a unitary p (|FFT(p)_l| = 1 for all l), the exact inverse equals
    the *involution* of p: p^{-1}_n = p_{-n mod N}. In the frequency domain
    this is just conjugation: FFT(p^{-1}) = conj(FFT(p)). So:
        unbind(s, p) = IFFT(FFT(s) · conj(FFT(p)))
    """
    S = np.fft.fft(s, axis=-1)
    P = np.fft.fft(p, axis=-1)
    return np.real(np.fft.ifft(S * np.conj(P), axis=-1))


def hrr_bundle_np(*items: np.ndarray) -> np.ndarray:
    """Sum (the "addition" operation in HRR algebra). Permutation-invariant."""
    return np.sum(np.stack(items, axis=0), axis=0)


def fpe_pos_vec_np(x: float | np.ndarray, n_modes: int,
                   phases: np.ndarray) -> np.ndarray:
    """Fractional Power Encoding — encode a continuous coord as a unitary vector.

    Position vector at coord x is `b^x` where `b` is a fixed base unitary
    vector with FFT phases `phases`. Equivalently in the FFT domain:
        FFT(p(x))_l = exp(j · phases_l · x)

    Composition law: `p(x) ⊛ p(y) = p(x + y)` — positions form an abelian
    group under binding.

    Args
    ----
    x       : scalar coord or shape (...,) array of coords
    n_modes : output vector length (in time domain)
    phases  : (n_modes // 2 + 1,) array of phases — one per rfft mode
              Must satisfy phases[0] = 0 (DC is real) and phases[-1] = 0
              if n_modes is even (Nyquist must be real).

    Returns
    -------
    p(x) : shape (..., n_modes) real unitary vector(s)
    """
    x = np.asarray(x)
    L = n_modes // 2 + 1
    assert phases.shape == (L,), f"phases must be shape ({L},), got {phases.shape}"
    # FFT-domain construction: |FFT(p)_l| = 1 for all l
    # Broadcasting: x can be (...,) and phases is (L,), so FFT is (..., L)
    fft_p = np.exp(1j * phases[None, ...] * x[..., None])
    # Force DC + Nyquist to be real (== 1) for Hermitian symmetry
    fft_p[..., 0] = 1.0
    if n_modes % 2 == 0:
        fft_p[..., -1] = 1.0
    return np.fft.irfft(fft_p, n=n_modes, axis=-1)


# ===========================================================================
# Torch HRR Patchifier — explicit time-domain bind/bundle implementation
# ===========================================================================

class HRRPatchifierTime(nn.Module):
    """Reference implementation that performs bind/bundle in the *time domain*
    using circular convolution explicitly (via FFT internally).

    Mathematically equivalent to RoPEPatchifier when phases are log-spaced,
    but the bind/bundle structure is exposed. Per-event flow:
        c_i = signal_proj(signal_i)         # real vector of size d_model
        p_i = FPE(rel_pos_i)                 # unitary real vector
        S   = Σ_i  c_i ⊛ p_i                  # bind + bundle (in FFT: pointwise multiply + sum)
        out = out_proj(S)
    """
    def __init__(self, signal_dim: int, coord_dim: int, d_model: int,
                 base: float = 100.0, agg: str = "mean"):
        super().__init__()
        assert d_model % (2 * coord_dim) == 0, (
            f"d_model={d_model} must be divisible by 2*coord_dim={2*coord_dim}"
        )
        self.coord_dim = coord_dim
        self.d_model = d_model
        # Per-axis vector length (HRR operates per axis)
        self.d_axis = d_model // coord_dim
        self.L = self.d_axis // 2 + 1   # number of rfft modes per axis
        self.agg = agg

        self.signal_proj = nn.Sequential(
            nn.Linear(signal_dim, d_model), nn.GELU(),
            nn.Linear(d_model, d_model),
        )
        self.out_proj = nn.Sequential(
            nn.Linear(d_model, d_model), nn.GELU(),
            nn.Linear(d_model, d_model),
        )

        # Phases for FPE: log-spaced, one per non-DC, non-Nyquist mode
        # The active modes are indices 1, ..., L-2 (or L-1 for odd d_axis)
        active_count = self.L - (2 if self.d_axis % 2 == 0 else 1)
        # log-spaced like RoPE
        phases_active = base ** (-torch.arange(active_count).float() * 2 / self.d_axis)
        # build full phase vector with DC=0 and (if even) Nyquist=0
        phases = torch.zeros(self.L)
        phases[1:1 + active_count] = phases_active
        self.register_buffer("phases", phases)   # (L,)

    def _fpe(self, rel_coord_axis: torch.Tensor) -> torch.Tensor:
        """Build unitary position vectors in *FFT space*.
        rel_coord_axis: (..., ) scalar per event.
        Returns: (..., L) complex tensor representing FFT(p(rel)).
        """
        # angle: (..., L)
        angles = rel_coord_axis.unsqueeze(-1) * self.phases
        return torch.complex(angles.cos(), angles.sin())

    def forward(self, patch_events: torch.Tensor,
                patch_centroids: torch.Tensor,
                event_kpm=None) -> torch.Tensor:
        """patch_events: (B, P, K, coord_dim + signal_dim).
        patch_centroids: (B, P, coord_dim).
        Returns: (B, P, d_model).
        """
        coord_dim = self.coord_dim
        coords = patch_events[..., :coord_dim]
        signal = patch_events[..., coord_dim:]
        rel = coords - patch_centroids.unsqueeze(2)             # (B, P, K, coord_dim)
        c = self.signal_proj(signal)                              # (B, P, K, d_model)
        B, P, K, _ = c.shape

        # Reshape into (axes, d_axis)
        c_axes = c.view(B, P, K, coord_dim, self.d_axis)          # (B, P, K, axes, d_axis)

        # rfft along the last dim → (B, P, K, axes, L) complex
        C = torch.fft.rfft(c_axes, dim=-1)

        bound_axes = []
        for ax in range(coord_dim):
            # FFT of position vector for this axis: (B, P, K, L) complex
            P_fft = self._fpe(rel[..., ax])
            # Bind = multiply in FFT domain: (B, P, K, L)
            bound = C[..., ax, :] * P_fft
            bound_axes.append(bound)
        bound = torch.stack(bound_axes, dim=-2)                   # (B, P, K, axes, L)

        # Bundle: sum across K events
        if event_kpm is not None:
            mask = (~event_kpm).float().unsqueeze(-1).unsqueeze(-1)  # (B, P, K, 1, 1)
            bound = bound * mask
            if self.agg == "mean":
                count = mask.sum(dim=2).clamp(min=1.0)
                S = bound.sum(dim=2) / count
            else:
                S = bound.sum(dim=2)
        else:
            if self.agg == "mean":
                S = bound.mean(dim=2)
            else:
                S = bound.sum(dim=2)
        # S: (B, P, axes, L) complex

        # IFFT back to time domain per axis: (B, P, axes, d_axis)
        S_time = torch.fft.irfft(S, n=self.d_axis, dim=-1)
        S_flat = S_time.reshape(B, P, self.d_model)               # (B, P, d_model)
        return self.out_proj(S_flat)


# ===========================================================================
# Re-export RoPE classes for training — HRR-with-FPE = RoPE-NUDFT in FFT
# ===========================================================================

from rope_patch_core import (
    RoPEPatchifier,
    CentroidRoPEMultiHeadAttention,
    RoPETransformerBlock,
    RoPEViTEncoder,
    RoPEViTPredictor,
)

# Aliases. The training pipeline uses these — same math, different framing.
HRRPatchifier   = RoPEPatchifier
HRRViTEncoder   = RoPEViTEncoder
HRRViTPredictor = RoPEViTPredictor
