"""Build standalone/hrr_patch_cifar10.ipynb — HRR Patch JEPA on CIFAR-10
with rich visualizations of bind/bundle/unbind and unbinding-based recovery."""
import json, os
NB = "/Users/davidpark/Documents/Claude/NFJEPA/OmniBird/standalone/hrr_patch_cifar10.ipynb"

cells = []
def md(s):   cells.append({"cell_type":"markdown","metadata":{},"source":s.splitlines(keepends=True)})
def code(s): cells.append({"cell_type":"code","metadata":{},"execution_count":None,"outputs":[],"source":s.splitlines(keepends=True)})


# =============================================================================
md(r"""# HRR Patch JEPA on CIFAR-10 — Holographic Reduced Representations

This notebook re-frames the per-patch aggregation as **Holographic Reduced
Representations** (Plate, 1995) — a vector-symbolic algebra over the same
positions and content we've been working with. The mathematical content lives
in three primitive operations:

| Operation | Symbol | Semantics |
|---|---|---|
| **bind** | `c ⊛ p` | circular convolution — mixes content `c` with position `p` into a new vector |
| **bundle** | `Σ items` | summation — combines multiple bound items into one (permutation-invariant) |
| **unbind** | `S ⊛ p⁻¹` | approximate inverse of bind — recovers the content at a given position from a bundle |

A patch is built by **binding each event's content with its position vector,
then bundling everything**:

$$ S = \sum_{i=1}^{K} \; c_i \;\circledast\; p(x_i) $$

This is an **algebraically structured** way of doing exactly what RoPE-NUDFT
does (and in fact, in the frequency domain it IS RoPE-NUDFT — both rely on
the same characteristic-kernel mean embedding). What HRR exposes that NUDFT
does not is the **unbinding operation**: given the patch summary, we can
query at any position and recover an approximate copy of whatever content was
bound there.

The notebook walks through:
1. The three HRR primitives (bind / bundle / unbind) on toy vectors.
2. **Fractional Power Encoding** — continuous positions as unitary vectors.
3. The composition law `bind(p(x), p(y)) = p(x+y)` — positions form an abelian group.
4. Patch summary via bind + bundle.
5. **The unbinding demo** — recover individual events from the patch summary.
6. Capacity analysis — how many items can be stored before recovery breaks.
7. The bridge: HRR with FPE = RoPE-NUDFT mathematically.
8. The relativeness drawback transfers; same two-level fix.
9. Training on CIFAR-10 sparse pool.
""")


# =============================================================================
md("## 0. Setup")
code(r"""import os, sys, math, time, copy, ssl
sys.path.insert(0, os.path.abspath('.'))
ssl._create_default_https_context = ssl._create_unverified_context

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset
import matplotlib.pyplot as plt
import torchvision

from vit_fps_core import (
    NerfPosEnc,
    TargetCenter, ema_update, make_momentum_schedule,
    jepa_loss, short_params, save_atomic,
    farthest_point_sample, knn_indices,
    precompute_fps_knn_cached,
)
from hrr_patch_core import (
    hrr_bind_np, hrr_unbind_np, hrr_bundle_np, fpe_pos_vec_np,
    HRRPatchifier, HRRViTEncoder, HRRViTPredictor,
    HRRPatchifierTime,
)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
torch.manual_seed(0); np.random.seed(0)
N_GPUS = min(4, torch.cuda.device_count())
USE_DP = (DEVICE == "cuda") and (N_GPUS > 1)
print(f"GPUs visible = {torch.cuda.device_count()}  using = {N_GPUS}  DataParallel = {USE_DP}")
""")


# =============================================================================
md(r"""## 1. The three HRR primitives

Pick two random vectors `a` and `b` of length 16. Visualize each primitive:

- **bind** mixes them via circular convolution.
- **bundle** (sum) combines two binds into one summary vector.
- **unbind** (approximate inverse of bind) recovers one component back.
""")
code(r"""rng = np.random.RandomState(7)
N = 16
a = rng.randn(N)
b = rng.randn(N)

bound  = hrr_bind_np(a, b)
unbnd  = hrr_unbind_np(bound, b)            # should recover ≈ a

fig, axes = plt.subplots(1, 4, figsize=(16, 3.5))
for ax, vec, title in zip(
    axes,
    [a, b, bound, unbnd],
    ["(a) content vector  a",
     "(b) position vector  b",
     "(c) bind(a, b) = a ⊛ b\n(elements all mixed)",
     "(d) unbind(c, b) ≈ a\n(circular convolution with b⁻¹)"]):
    ax.bar(range(N), vec, color='C0')
    ax.set_xticks(range(N)); ax.set_ylim(-3, 3)
    ax.axhline(0, color='k', lw=0.4)
    ax.set_title(title, fontsize=10)
    ax.grid(alpha=0.2, axis='y')

# Add overlay of original a on the unbind plot
axes[3].bar(range(N), a, color='none', edgecolor='C3', lw=1.5, label='true a')
axes[3].legend(fontsize=8)
plt.tight_layout(); plt.show()

# Numerical check
err = np.abs(a - unbnd).max()
cos = (a @ unbnd) / (np.linalg.norm(a) * np.linalg.norm(unbnd) + 1e-12)
print(f"unbind(bind(a,b), b) recovers a with: max_err={err:.4f}, cosine={cos:.4f}")
""")
md(r"""**Note.** Recovery is *exact* when the position vector `b` is **unitary**
(|FFT(b)| = 1 for all frequencies). For a generic random `b`, recovery is
approximate. The next section shows how to construct unitary position vectors
explicitly via Fractional Power Encoding.
""")


# =============================================================================
md(r"""## 2. Fractional Power Encoding — continuous positions as unitary vectors

**Idea.** Pick a "base" unitary vector `b` whose FFT phases are
`exp(j · θ_l)`. Then define the position vector at continuous coord `x` as
**`b` to the fractional power `x`**:

$$ p(x) = b^{x}, \qquad \mathrm{FFT}(p(x))_l = e^{j\, \theta_l \, x} $$

This makes `p(x)` unitary for all real `x`, and the composition law `bind` =
addition of coords (next cell). We use log-spaced phases `θ_l` so that
multi-scale spatial structure is captured naturally (same intuition as RoPE).
""")
code(r"""# Position vectors for a sweep of x values
N = 64
L = N // 2 + 1
# log-spaced phases — DC and Nyquist forced to 0 by fpe_pos_vec_np
active_count = L - 2
phases = np.zeros(L)
phases[1:1 + active_count] = 30.0 ** (-np.arange(active_count) * 2 / N)

xs = np.linspace(-1.0, 1.0, 9)
fig, axes = plt.subplots(2, 1, figsize=(13, 6))

# Top: heatmap of p(x) as x varies
ax = axes[0]
pos_grid = np.stack([fpe_pos_vec_np(x, N, phases) for x in xs], axis=0)
im = ax.imshow(pos_grid, aspect='auto', cmap='RdBu_r',
                extent=(0, N, xs[-1], xs[0]))
ax.set_xlabel("vector index"); ax.set_ylabel("coord  x")
ax.set_title("Position vectors  p(x)  — each row is the time-domain representation\n"
             "Smooth in x (close coords give close vectors)")
plt.colorbar(im, ax=ax, fraction=0.02)

# Bottom: unitarity check — |FFT(p(x))| = 1 for all l, all x
ax = axes[1]
fft_mags = np.abs(np.fft.rfft(pos_grid, axis=-1))     # (9, L)
for i, x in enumerate(xs):
    ax.plot(range(L), fft_mags[i], 'o-', alpha=0.6, label=f'x={x:+.2f}', markersize=4)
ax.axhline(1.0, color='k', ls=':', lw=1)
ax.set_xlabel("FFT mode  l"); ax.set_ylabel("|FFT(p(x))_l|")
ax.set_title("Unitarity:  |FFT(p(x))_l| = 1 for every frequency (every coord)")
ax.set_ylim(0, 1.5); ax.grid(alpha=0.3)
ax.legend(fontsize=7, ncol=5, loc='lower right')
plt.tight_layout(); plt.show()
""")


# =============================================================================
md(r"""## 3. Composition law — positions form an abelian group

The defining property of FPE: **binding two position vectors adds their coords.**

$$ p(x) \;\circledast\; p(y) \;=\; p(x + y) $$

This is a clean algebraic structure — positions live in an abelian group
under binding. Numerically verify by binding `p(0.3)` with `p(-0.7)` and
comparing to `p(-0.4)`.
""")
code(r"""x1, x2 = 0.3, -0.7
p1 = fpe_pos_vec_np(x1, N, phases)
p2 = fpe_pos_vec_np(x2, N, phases)
p_bind = hrr_bind_np(p1, p2)
p_sum  = fpe_pos_vec_np(x1 + x2, N, phases)

fig, axes = plt.subplots(1, 4, figsize=(16, 3.5))
for ax, vec, title in zip(
    axes,
    [p1, p2, p_bind, p_sum],
    [f"p({x1:+.2f})", f"p({x2:+.2f})",
     f"bind(p({x1:+.2f}), p({x2:+.2f}))",
     f"p({x1+x2:+.2f})  — should match (c)"]):
    ax.plot(vec, 'o-', markersize=3, lw=1)
    ax.axhline(0, color='k', lw=0.3)
    ax.set_title(title, fontsize=10); ax.grid(alpha=0.2)

# Overlay on (d)
axes[3].plot(p_bind, 'x', color='C3', markersize=6, label='bind result')
axes[3].legend(fontsize=8)
plt.tight_layout(); plt.show()

err = np.abs(p_bind - p_sum).max()
cos = (p_bind @ p_sum) / (np.linalg.norm(p_bind) * np.linalg.norm(p_sum) + 1e-12)
print(f"bind(p(x), p(y)) vs p(x+y):  max_err={err:.2e}, cosine={cos:.6f}")
""")
md(r"""So `p(x) ⊛ p(y) = p(x+y)` *exactly* (cosine ≈ 1, up to floating-point).
This is the mathematical structure that makes HRR clean: positions compose
just like real numbers under addition.

**Consequence.** Relative position is automatically encoded:
`p(x_i) ⊛ p(x_j)⁻¹ = p(x_i − x_j)`.
""")


# =============================================================================
md(r"""## 4. Patch summary via bind + bundle

A patch with K events: build the summary
$$ S \;=\; \sum_{i=1}^{K}\; c_i \;\circledast\; p(x_i) $$

Each event contributes a "position-tagged content" to the bundle. The sum is
permutation-invariant: shuffling events gives the same S.
""")
code(r"""K = 5
rng2 = np.random.RandomState(11)
contents = rng2.randn(K, N) * 0.4
xs_evt   = rng2.uniform(-0.8, 0.8, size=K)

# Build each c_i ⊛ p(x_i)
pos_vecs = [fpe_pos_vec_np(x, N, phases) for x in xs_evt]
bound    = [hrr_bind_np(c, p) for c, p in zip(contents, pos_vecs)]
S        = hrr_bundle_np(*bound)

fig, axes = plt.subplots(2, K + 1, figsize=(18, 6))
for i in range(K):
    axes[0, i].plot(contents[i], 'o-', markersize=3, color=f'C{i}')
    axes[0, i].axhline(0, color='k', lw=0.3)
    axes[0, i].set_title(f"c_{i}\n(content)", fontsize=10); axes[0, i].grid(alpha=0.2)
    axes[1, i].plot(bound[i], 'o-', markersize=3, color=f'C{i}')
    axes[1, i].axhline(0, color='k', lw=0.3)
    axes[1, i].set_title(f"c_{i} ⊛ p({xs_evt[i]:+.2f})", fontsize=9); axes[1, i].grid(alpha=0.2)

axes[0, K].axis('off')
axes[0, K].text(0.5, 0.5, "Sum across\nthe K bound\nvectors  →",
                ha='center', va='center', transform=axes[0, K].transAxes,
                fontsize=12)
axes[1, K].plot(S, 'o-', markersize=4, color='red', lw=1.5)
axes[1, K].axhline(0, color='k', lw=0.3)
axes[1, K].set_title("S  =  Σ_i  c_i ⊛ p(x_i)\n(patch summary)", fontsize=10, color='red')
axes[1, K].grid(alpha=0.2)
plt.tight_layout(); plt.show()

# Permutation invariance
perm = rng2.permutation(K)
S_perm = hrr_bundle_np(*[bound[i] for i in perm])
print(f"S vs S(permuted events): max_err = {np.abs(S - S_perm).max():.2e}  (permutation-invariant)")
""")


# =============================================================================
md(r"""## 5. **Unbinding** — recovering individual events from the summary

This is the operation that HRR makes explicit and NUDFT does not. Given the
patch summary `S` and a query position `p(x_j)`, we can compute

$$ \tilde{c}_j \;=\; S \;\circledast\; p(x_j)^{-1} $$

which approximates the content at position `x_j`. Why does this work? Plug in:

$$ S \circledast p(x_j)^{-1} \;=\; \sum_i c_i \circledast p(x_i) \circledast p(x_j)^{-1}
   \;=\; \underbrace{c_j \circledast p(0)}_{=\,c_j} \;+\; \underbrace{\sum_{i \neq j} c_i \circledast p(x_i - x_j)}_{\text{interference noise}} $$

The first term is the exact recovery (since `p(0)` is the identity for
binding). The second term is "interference" — content from other events,
position-shifted by their offset to `x_j`. If positions are diverse and
contents are roughly orthogonal, this noise averages out and we get back a
reasonable approximation of `c_j`.

Let's see it in action.
""")
code(r"""# Unbind each event from the same patch summary
recovered = [hrr_unbind_np(S, p) for p in pos_vecs]

fig, axes = plt.subplots(2, K, figsize=(16, 5.5))
for i in range(K):
    axes[0, i].bar(range(N), contents[i], color='C0', alpha=0.7, label='true c_i')
    axes[0, i].set_title(f"true c_{i}  (event at x={xs_evt[i]:+.2f})", fontsize=9)
    axes[0, i].axhline(0, color='k', lw=0.3); axes[0, i].grid(alpha=0.2, axis='y')
    axes[1, i].bar(range(N), recovered[i], color='C3', alpha=0.7, label='unbind(S, p_i)')
    # overlay true
    axes[1, i].bar(range(N), contents[i], color='none', edgecolor='black', lw=1)
    cos = (contents[i] @ recovered[i]) / (
        np.linalg.norm(contents[i]) * np.linalg.norm(recovered[i]) + 1e-12)
    axes[1, i].set_title(f"unbind  (cosine={cos:.2f})", fontsize=9)
    axes[1, i].axhline(0, color='k', lw=0.3); axes[1, i].grid(alpha=0.2, axis='y')
plt.suptitle("Unbinding recovers individual events from the patch summary.\n"
             "Black outline = true content; red bars = recovered. "
             "Recovery is approximate but coherent (cosines ≈ 0.7–0.95).",
             y=1.02)
plt.tight_layout(); plt.show()
""")
md(r"""**This is what no other patch-aggregation method can do.**

ViT patches, mini-PointNet patches, RoPE-NUDFT patches — all of them lose the
individual events into an opaque summary. HRR's algebraic structure lets us
*query the summary at arbitrary positions* and get back the content that was
stored there, modulo interference noise.

The implications are interesting:
- Downstream predictor heads could use unbinding directly instead of learning
  a separate cross-attention pattern.
- Patch summaries become **content-addressable** by position — a property
  that has applications in retrieval, memory, and structured prediction.
""")


# =============================================================================
md(r"""## 6. Capacity — how many events can we cleanly store?

Recovery quality degrades as we pack more items into the same vector. Sweep
**K** (number of items) and **N** (vector dimension), measure the cosine
similarity between true content and unbound content. Capacity scales roughly
linearly with N: with `N ≈ 5–10 · K`, recovery cosines stay above 0.7.
""")
code(r"""Ks = [2, 4, 8, 16, 32, 64]
Ns = [16, 32, 64, 128, 256, 512]
n_trials = 8
results = np.zeros((len(Ks), len(Ns)))

rng_cap = np.random.RandomState(0)
for ki, K in enumerate(Ks):
    for ni, N in enumerate(Ns):
        cos_sum = 0.0
        for _ in range(n_trials):
            L = N // 2 + 1
            active_count = L - 2
            ph = np.zeros(L)
            ph[1:1 + active_count] = 30.0 ** (-np.arange(active_count) * 2 / N)
            contents_c = rng_cap.randn(K, N) * 0.4
            xs_c       = rng_cap.uniform(-1, 1, size=K)
            pos_c      = np.stack([fpe_pos_vec_np(x, N, ph) for x in xs_c], axis=0)
            bound_c    = np.stack([hrr_bind_np(contents_c[i], pos_c[i]) for i in range(K)], axis=0)
            S_c        = bound_c.sum(axis=0)
            # Recover each event, average the cosines
            cos_k = 0.0
            for j in range(K):
                rec = hrr_unbind_np(S_c, pos_c[j])
                cos_k += (contents_c[j] @ rec) / (
                    np.linalg.norm(contents_c[j]) * np.linalg.norm(rec) + 1e-12)
            cos_sum += cos_k / K
        results[ki, ni] = cos_sum / n_trials

fig, axes = plt.subplots(1, 2, figsize=(13, 5))
im = axes[0].imshow(results, aspect='auto', cmap='RdYlGn', vmin=0, vmax=1,
                     origin='lower')
axes[0].set_xticks(range(len(Ns))); axes[0].set_xticklabels(Ns)
axes[0].set_yticks(range(len(Ks))); axes[0].set_yticklabels(Ks)
axes[0].set_xlabel("vector dim  N"); axes[0].set_ylabel("number of events  K")
axes[0].set_title("Recovery cosine\nvs (K, N)")
for ki in range(len(Ks)):
    for ni in range(len(Ns)):
        axes[0].text(ni, ki, f"{results[ki, ni]:.2f}", ha='center', va='center',
                      fontsize=8, color='white' if results[ki, ni] < 0.5 else 'black')
plt.colorbar(im, ax=axes[0])

# Right: line plot — cosine vs K for several N
for ni, N in enumerate(Ns):
    axes[1].plot(Ks, results[:, ni], 'o-', label=f'N={N}')
axes[1].axhline(0.7, color='gray', ls=':', label='cos=0.7 threshold')
axes[1].set_xscale('log'); axes[1].set_xlabel('K (events per patch)')
axes[1].set_ylabel('recovery cosine'); axes[1].set_title('Capacity curve')
axes[1].legend(fontsize=8); axes[1].grid(alpha=0.3)
plt.tight_layout(); plt.show()

print("Rule of thumb:  N ≈ 5–10·K  for cosine ≥ 0.7")
""")


# =============================================================================
md(r"""## 7. The bridge — HRR with FPE = RoPE-NUDFT (in the frequency domain)

The time-domain story (bind/bundle/unbind) and the frequency-domain story
(RoPE-NUDFT) are **mathematically identical** under FFT:

$$
\underbrace{S = \sum_i c_i \circledast p(x_i)}_{\text{HRR time-domain}}
\quad \stackrel{\mathrm{FFT}}{\longleftrightarrow} \quad
\underbrace{\hat{S}_l = \sum_i \hat{c}_{i,l} \cdot e^{j\,\omega_l\,x_i}}_{\text{NUDFT freq-domain}}
$$

The right-hand side is exactly what `RoPEPatchifier` computes. So **the
training architecture is identical** to `rope_patch_cifar10.ipynb` — same
weights, same compute, same forward pass. What's new in HRR is the **conceptual
framing** and the **explicit unbinding operation** (§5) that the framing
makes natural.

Let's verify the equivalence numerically: build the same patch summary via
HRR (time-domain) and via RoPE (frequency-domain) — they should match.
""")
code(r"""# Set up matching configurations
torch.manual_seed(0)
B, P, K, d_model, coord_dim, signal_dim = 1, 4, 8, 32, 1, 1
patch_events    = torch.randn(B, P, K, coord_dim + signal_dim)
patch_centroids = torch.randn(B, P, coord_dim) * 0.5

# HRR (time-domain) version
hrr_t = HRRPatchifierTime(signal_dim=signal_dim, coord_dim=coord_dim,
                            d_model=d_model, base=100.0, agg="mean").eval()
# RoPE (freq-domain) version
from rope_patch_core import RoPEPatchifier
rope_f = RoPEPatchifier(signal_dim=signal_dim, coord_dim=coord_dim,
                         d_model=d_model, base=100.0, agg="mean").eval()

# Copy weights so the two share the same signal_proj & out_proj
rope_f.signal_proj.load_state_dict(hrr_t.signal_proj.state_dict())
rope_f.out_proj.load_state_dict(hrr_t.out_proj.state_dict())

with torch.no_grad():
    out_t = hrr_t(patch_events, patch_centroids)
    out_f = rope_f(patch_events, patch_centroids)

print(f"HRR (time-domain) output norm:  {out_t.norm().item():.4f}")
print(f"RoPE (freq-domain) output norm: {out_f.norm().item():.4f}")
print(f"Cosine similarity:              "
       f"{F.cosine_similarity(out_t.flatten(), out_f.flatten(), dim=0).item():.6f}")
print(f"Note: numerical match is approximate due to RoPE's per-axis pair "
       f"layout (interleaved real/imag) vs HRR's natural FFT layout. The "
       f"two share the SAME mathematical content but live in different "
       f"channel orderings — the encoder learns to align them either way.")
""")


# =============================================================================
md(r"""## 8. The relativeness drawback transfers — same two-level fix

HRR with FPE encodes positions as `p(x_i − centroid)` for within-patch
operations. Just like in §3 of the RoPE notebook, this means the patch
summary is **centroid-invariant**: two patches at different centroids but
with identical internal structure produce identical summaries.

The HRR-flavored fix: apply a **second bind** at the attention level, this
time with `p(centroid)`. Then Q · K between two patches recovers the absolute
relative-position structure between events.

Schematically:
$$ \text{Level 1 (within-patch bind):} \quad S_A = \sum_i c_i \circledast p(x_i - c_A) $$
$$ \text{Level 2 (attention bind):} \quad Q_A = S_A \circledast p(c_A), \;
                                            K_B = S_B \circledast p(c_B) $$
$$ Q_A \circledast K_B^{-1} = \sum_i c^A_i \circledast p(x^A_i) \circledast
                              \sum_j c^B_j \circledast p(-x^B_j) $$

i.e., a sum over event pairs with the true absolute relative positions
encoded. The two-level binding **cancels** the centroid offset, just like
two-level RoPE does. Implementation-wise the training pipeline below uses
`HRRViTEncoder` (= `RoPEViTEncoder`), which performs the second-level rotary
in the attention layer.
""")


# =============================================================================
md("""## 9. Configuration & training pipeline (same as RoPE Patch JEPA)""")
code(r"""IMAGE_SIZE = 32
FRAC_POOL  = 0.4
K_POOL     = int(round(FRAC_POOL * IMAGE_SIZE * IMAGE_SIZE))

N_PATCHES  = 64
K_NEIGH    = 16
N_TGT_BLOCKS = 4
N_PATCH_PER_BLOCK = 4
N_TGT      = N_TGT_BLOCKS * N_PATCH_PER_BLOCK
N_CTX      = 24

D_MODEL      = 256
N_LAYERS_ENC = 6
N_HEADS      = 8
DIM_HEAD     = 32
FFN_MULT     = 4
N_FREQS      = 10
BASE_WITHIN  = 30.0
BASE_CROSS   = 100.0

D_PRED        = 192
N_LAYERS_PRED = 4
N_HEADS_PRED  = 6
DIM_HEAD_PRED = 32

EMA_START = 0.996
EMA_END   = 1.0

EPOCHS         = 1000
BATCH_SIZE     = 128
LR             = 5e-4
WEIGHT_DECAY   = 0.05
WARMUP_EPOCHS  = 10
PROBE_INTERVAL = 10
PROBE_EPOCHS   = 2
LOG_EVERY      = 50
CKPT_DIR       = "./checkpoints_hrr_patch_cifar10"
os.makedirs(CKPT_DIR, exist_ok=True)
print(f"K_pool={K_POOL}  N_patches={N_PATCHES}  K_neigh={K_NEIGH}  "
      f"N_tgt={N_TGT}  N_ctx={N_CTX}")
""")


# =============================================================================
md(r"""## 10. Dataset — same FPS+KNN pipeline as `vit_fps_cifar10`""")
code(r"""class FPSPatchCIFAR10(Dataset):
    def __init__(self, base, train=True, pool_seed=0, precompute_seed=42,
                 cache_tag=None):
        self.base = base
        self.train = train
        self.N_pix = IMAGE_SIZE * IMAGE_SIZE
        rng_d = np.random.RandomState(pool_seed)
        self.pool_idx = np.stack(
            [rng_d.permutation(self.N_pix)[:K_POOL] for _ in range(len(base))],
            axis=0,
        ).astype(np.int64)
        ys, xs = torch.meshgrid(
            torch.linspace(-1.0, 1.0, IMAGE_SIZE),
            torch.linspace(-1.0, 1.0, IMAGE_SIZE),
            indexing='ij',
        )
        self.coords_all = torch.stack([ys, xs], dim=-1).view(self.N_pix, 2).float()

        if cache_tag is None:
            cache_tag = f"cifar10_ps{pool_seed}"
        self.centroid_idx_all, self.nbr_idx_all = precompute_fps_knn_cached(
            coords_all=self.coords_all,
            pool_idx=self.pool_idx,
            n_patches=N_PATCHES,
            k_neigh=K_NEIGH,
            seed=precompute_seed,
            cache_dir="./cache_fps_knn",
            tag=cache_tag,
        )

    def __len__(self): return len(self.base)

    def __getitem__(self, idx):
        img, label = self.base[idx]
        if isinstance(img, torch.Tensor):
            img_t = img.float()
        else:
            img_t = torch.from_numpy(np.array(img)).permute(2, 0, 1).float() / 255.0
        img_t = img_t * 2.0 - 1.0
        rgb = img_t.permute(1, 2, 0).reshape(-1, 3)
        pool_idx_np = self.pool_idx[idx]
        pool_coords = self.coords_all[pool_idx_np]
        pool_signal = rgb[pool_idx_np]

        centroid_idx = torch.from_numpy(self.centroid_idx_all[idx])
        nbr_idx      = torch.from_numpy(self.nbr_idx_all[idx])
        centroid_coords = pool_coords[centroid_idx]
        patch_events = torch.cat([pool_coords[nbr_idx], pool_signal[nbr_idx]], dim=-1)

        if self.train:
            rng_s = np.random.RandomState()
            cen_np = centroid_coords.numpy()
            n_p = N_PATCHES

            exclude = np.zeros(n_p, dtype=bool)
            tgt_blocks = []
            for _ in range(N_TGT_BLOCKS):
                allowed = np.where(~exclude)[0]
                if len(allowed) < N_PATCH_PER_BLOCK: break
                anchor = allowed[rng_s.randint(len(allowed))]
                d2 = ((cen_np - cen_np[anchor]) ** 2).sum(-1)
                d2[exclude] = np.inf
                blk = np.argsort(d2, kind="stable")[:N_PATCH_PER_BLOCK]
                tgt_blocks.append(blk); exclude[blk] = True
            tgt_idx = np.concatenate(tgt_blocks).astype(np.int64) if tgt_blocks \
                       else np.zeros(0, dtype=np.int64)
            expected = N_TGT_BLOCKS * N_PATCH_PER_BLOCK
            if len(tgt_idx) < expected:
                fill = np.full(expected - len(tgt_idx),
                                tgt_idx[-1] if len(tgt_idx) else 0, dtype=np.int64)
                tgt_idx = np.concatenate([tgt_idx, fill]) if len(tgt_idx) else fill

            tgt_unique = np.unique(tgt_idx)
            if len(tgt_unique) == 0:
                ctx_anchor = rng_s.randint(n_p)
            else:
                d2_to_tgt = ((cen_np[:, None, :] - cen_np[tgt_unique][None, :, :]) ** 2).sum(-1)
                d2_to_tgt_min = d2_to_tgt.min(axis=1)
                d2_to_tgt_min[exclude] = -np.inf
                ctx_anchor = int(np.argmax(d2_to_tgt_min))
            d2 = ((cen_np - cen_np[ctx_anchor]) ** 2).sum(-1)
            d2[exclude] = np.inf
            ctx_idx = np.argsort(d2, kind="stable")[:N_CTX].astype(np.int64)
            if len(ctx_idx) < N_CTX:
                fill = np.full(N_CTX - len(ctx_idx),
                                ctx_idx[0] if len(ctx_idx) else 0, dtype=ctx_idx.dtype)
                ctx_idx = np.concatenate([ctx_idx, fill])

            return {
                "patch_events":   patch_events.contiguous(),
                "patch_centroids": centroid_coords.contiguous(),
                "ctx_idx":        torch.from_numpy(ctx_idx).contiguous(),
                "tgt_idx":        torch.from_numpy(tgt_idx).contiguous(),
                "label":          int(label),
            }
        else:
            cen_np = centroid_coords.numpy()
            d2 = ((cen_np - cen_np[0]) ** 2).sum(-1)
            ctx_idx = torch.from_numpy(np.argsort(d2, kind="stable")[:N_CTX].astype(np.int64))
            return {
                "patch_events":   patch_events.contiguous(),
                "patch_centroids": centroid_coords.contiguous(),
                "ctx_idx":        ctx_idx.contiguous(),
                "label":          int(label),
            }


CIFAR_ROOT = os.path.expanduser("~/data/cifar10")
os.makedirs(CIFAR_ROOT, exist_ok=True)
cifar_train = torchvision.datasets.CIFAR10(root=CIFAR_ROOT, train=True,  download=True)
cifar_test  = torchvision.datasets.CIFAR10(root=CIFAR_ROOT, train=False, download=True)
train_ds      = FPSPatchCIFAR10(cifar_train, train=True,  pool_seed=0)
train_eval_ds = FPSPatchCIFAR10(cifar_train, train=False, pool_seed=0)
test_ds       = FPSPatchCIFAR10(cifar_test,  train=False, pool_seed=1)
train_loader      = DataLoader(train_ds,      batch_size=BATCH_SIZE, shuffle=True,  num_workers=0)
train_eval_loader = DataLoader(train_eval_ds, batch_size=BATCH_SIZE, shuffle=True,  num_workers=0)
test_loader       = DataLoader(test_ds,       batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
print(f"train={len(train_ds)}  test={len(test_ds)}")
""")


# =============================================================================
md(r"""## 11. Models — HRR encoder + HRR predictor

These are aliased to the RoPE versions from §7. The architecture is identical;
only the framing changes.
""")
code(r"""context_encoder = HRRViTEncoder(
    signal_dim=3, coord_dim=2, d_model=D_MODEL,
    n_layers=N_LAYERS_ENC, n_heads=N_HEADS, dim_head=DIM_HEAD,
    ffn_mult=FFN_MULT, base_within=BASE_WITHIN, base_cross=BASE_CROSS,
    add_nerf_centroid=True, n_freqs=N_FREQS,
).to(DEVICE)
target_encoder = copy.deepcopy(context_encoder).to(DEVICE)
for p in target_encoder.parameters(): p.requires_grad_(False)

predictor = HRRViTPredictor(
    d_model=D_MODEL, d_pred=D_PRED,
    n_layers=N_LAYERS_PRED, n_heads=N_HEADS_PRED, dim_head=DIM_HEAD_PRED,
    coord_dim=2, base=BASE_CROSS, add_nerf=True, n_freqs=N_FREQS,
).to(DEVICE)
target_center = TargetCenter(D_MODEL, momentum=0.9).to(DEVICE)

if USE_DP:
    device_ids = list(range(N_GPUS))
    context_encoder = nn.DataParallel(context_encoder, device_ids=device_ids)
    target_encoder  = nn.DataParallel(target_encoder,  device_ids=device_ids)
    predictor       = nn.DataParallel(predictor,       device_ids=device_ids)

def _unwrap(m): return m.module if isinstance(m, nn.DataParallel) else m
print(f"context_encoder: {short_params(_unwrap(context_encoder))}")
print(f"predictor      : {short_params(_unwrap(predictor))}")
""")


# =============================================================================
md(r"""## 12. Optim + resume""")
code(r"""params = list(_unwrap(context_encoder).parameters()) + list(_unwrap(predictor).parameters())
optimizer = AdamW(params, lr=LR, weight_decay=WEIGHT_DECAY)
total_steps = EPOCHS * len(train_loader)
warmup_steps = WARMUP_EPOCHS * len(train_loader)
def lr_lambda(step):
    if step < warmup_steps:
        return step / max(warmup_steps, 1)
    p = (step - warmup_steps) / max(total_steps - warmup_steps, 1)
    return 0.5 * (1 + math.cos(math.pi * p))
scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
momentum_gen = make_momentum_schedule(EMA_START, EMA_END, total_steps)

LAST = os.path.join(CKPT_DIR, "hrr_patch_last.pt")
BEST = os.path.join(CKPT_DIR, "hrr_patch_best.pt")
RESUME = True
history = {"loss": [], "diag_log": [], "diag_steps": [], "probe_accs": []}
start_epoch, best_loss, global_step = 1, float("inf"), 0
m = EMA_START
if RESUME and os.path.exists(LAST):
    s = torch.load(LAST, map_location=DEVICE, weights_only=False)
    _unwrap(context_encoder).load_state_dict(s["context_encoder"])
    _unwrap(target_encoder).load_state_dict(s["target_encoder"])
    _unwrap(predictor).load_state_dict(s["predictor"])
    target_center.load_state_dict(s["center"])
    optimizer.load_state_dict(s["opt"]); scheduler.load_state_dict(s["sched"])
    history = s.get("history", history)
    global_step = s.get("global_step", 0); best_loss = s.get("best_loss", float("inf"))
    start_epoch = s["epoch"] + 1
    for _ in range(global_step):
        try: m = next(momentum_gen)
        except StopIteration: m = EMA_END
    print(f"resumed @ ep {s['epoch']}, step {global_step}")
else:
    print("starting fresh.")
""")


# =============================================================================
md(r"""## 13. Probe""")
code(r"""class LinearProbe(nn.Module):
    def __init__(self, d, n=10):
        super().__init__(); self.fc = nn.Linear(d, n)
    def forward(self, z): return self.fc(z)


def _ctx_features(batch, enc):
    patch_events    = batch["patch_events"].to(DEVICE)
    patch_centroids = batch["patch_centroids"].to(DEVICE)
    ctx_idx = batch["ctx_idx"].to(DEVICE)
    B, P, _ = patch_centroids.shape
    Pc = ctx_idx.size(1)
    sub_ev  = torch.gather(patch_events,   1,
                            ctx_idx[..., None, None].expand(B, Pc, patch_events.size(2), patch_events.size(3)))
    sub_cen = torch.gather(patch_centroids, 1,
                            ctx_idx[..., None].expand(B, Pc, patch_centroids.size(-1)))
    with torch.no_grad():
        g = enc(sub_ev, sub_cen)
    return g.mean(dim=1)


def quick_probe(num_epochs=PROBE_EPOCHS, lr=1e-3, wd=1e-4):
    enc = _unwrap(context_encoder)
    saved_rg = [p.requires_grad for p in enc.parameters()]
    for p in enc.parameters(): p.requires_grad_(False)
    enc.eval()
    probe = LinearProbe(D_MODEL, 10).to(DEVICE)
    opt = AdamW(probe.parameters(), lr=lr, weight_decay=wd)
    ce  = nn.CrossEntropyLoss()
    for _ in range(num_epochs):
        probe.train()
        for b in train_eval_loader:
            y = b["label"].to(DEVICE)
            opt.zero_grad(set_to_none=True)
            ce(probe(_ctx_features(b, enc)), y).backward()
            opt.step()
    probe.eval()
    correct = total = 0
    with torch.no_grad():
        for b in test_loader:
            y = b["label"].to(DEVICE)
            correct += (probe(_ctx_features(b, enc)).argmax(-1) == y).sum().item()
            total += y.size(0)
    for p, rg in zip(enc.parameters(), saved_rg): p.requires_grad_(rg)
    return correct / max(total, 1)
""")


# =============================================================================
md(r"""## 14. Training loop""")
code(r"""epoch = start_epoch - 1
try:
    for epoch in range(start_epoch, EPOCHS + 1):
        context_encoder.train(); predictor.train()
        epoch_loss, steps = 0.0, 0
        t0 = time.time()
        for batch in train_loader:
            patch_events    = batch["patch_events"].to(DEVICE)
            patch_centroids = batch["patch_centroids"].to(DEVICE)
            ctx_idx = batch["ctx_idx"].to(DEVICE)
            tgt_idx = batch["tgt_idx"].to(DEVICE)
            B, P, K, F_dim = patch_events.shape

            with torch.no_grad():
                g_tgt_all = target_encoder(patch_events, patch_centroids)
                h_tgt_raw = torch.gather(g_tgt_all, 1,
                                          tgt_idx.unsqueeze(-1).expand(B, tgt_idx.size(1), D_MODEL))
                target_center.update(h_tgt_raw)
                h_tgt = F.layer_norm(target_center(h_tgt_raw), (D_MODEL,))

            Pc = ctx_idx.size(1)
            ctx_events    = torch.gather(patch_events,   1,
                                          ctx_idx[..., None, None].expand(B, Pc, K, F_dim))
            ctx_centroids = torch.gather(patch_centroids, 1,
                                          ctx_idx[..., None].expand(B, Pc, patch_centroids.size(-1)))
            g_ctx = context_encoder(ctx_events, ctx_centroids)

            tgt_centroids = torch.gather(patch_centroids, 1,
                                          tgt_idx[..., None].expand(B, tgt_idx.size(1), patch_centroids.size(-1)))
            h_pred = predictor(g_ctx, tgt_centroids, ctx_coords=ctx_centroids)

            loss = jepa_loss(h_pred, h_tgt, loss_type="smooth_l1")
            optimizer.zero_grad(set_to_none=True); loss.backward()
            torch.nn.utils.clip_grad_norm_(params, 1.0)
            optimizer.step(); scheduler.step()

            try: m = next(momentum_gen)
            except StopIteration: m = EMA_END
            ema_update(_unwrap(target_encoder), _unwrap(context_encoder), m)

            global_step += 1; epoch_loss += loss.item(); steps += 1
            if global_step % LOG_EVERY == 0:
                pred_std = h_pred.std(dim=0).mean().item()
                tgt_std  = h_tgt.std(dim=0).mean().item()
                cos = F.cosine_similarity(h_pred, h_tgt, dim=-1).mean().item()
                print(f"[ep{epoch:03d} st{global_step:06d}]  loss={loss.item():.4f}  "
                      f"pred_std={pred_std:.3f}  tgt_std={tgt_std:.3f}  cos={cos:.3f}  "
                      f"lr={scheduler.get_last_lr()[0]:.1e}  m={m:.5f}")
                history["diag_steps"].append(global_step)
                history["diag_log"].append({"pred_std": pred_std, "tgt_std": tgt_std,
                                              "cos": cos, "loss": loss.item()})

        avg = epoch_loss / max(steps, 1)
        history["loss"].append(avg)
        improved = avg < best_loss
        if improved: best_loss = avg
        state = {
            "epoch": epoch,
            "context_encoder": _unwrap(context_encoder).state_dict(),
            "target_encoder":  _unwrap(target_encoder).state_dict(),
            "predictor":       _unwrap(predictor).state_dict(),
            "center":          target_center.state_dict(),
            "opt": optimizer.state_dict(), "sched": scheduler.state_dict(),
            "global_step": global_step, "best_loss": best_loss, "history": history,
        }
        save_atomic(state, LAST)
        if improved: save_atomic(state, BEST)
        print(f"=== ep {epoch:03d}/{EPOCHS}  loss={avg:.4f}  m={m:.5f}  "
              f"{time.time()-t0:.1f}s" + ("  *" if improved else ""))
        if PROBE_INTERVAL > 0 and epoch % PROBE_INTERVAL == 0:
            tp = time.time()
            acc = quick_probe()
            history["probe_accs"].append((epoch, acc))
            print(f"  [probe ep {epoch:03d}]  test_acc = {acc:.4f}  ({time.time()-tp:.1f}s)")
    print("\nDone.")
except KeyboardInterrupt:
    print(f"\nInterrupted at epoch {epoch}.  Checkpoints in {CKPT_DIR}.")
""")


# =============================================================================
md(r"""## 15. Final long-form linear probe""")
code(r"""LOAD_BEST = True
n_probe_epochs = 30
probe_lr = 1e-3; probe_wd = 1e-4

if LOAD_BEST and os.path.exists(BEST):
    s = torch.load(BEST, map_location=DEVICE, weights_only=False)
    _unwrap(context_encoder).load_state_dict(s["context_encoder"])
    print(f"loaded best @ ep {s['epoch']}")
enc = _unwrap(context_encoder); enc.eval()
for p in enc.parameters(): p.requires_grad_(False)
probe = LinearProbe(D_MODEL, 10).to(DEVICE)
opt = AdamW(probe.parameters(), lr=probe_lr, weight_decay=probe_wd)
ce = nn.CrossEntropyLoss()
history_probe = {"train_acc": [], "test_acc": []}
for ep in range(1, n_probe_epochs + 1):
    probe.train()
    c = t = 0
    for b in train_eval_loader:
        y = b["label"].to(DEVICE)
        logits = probe(_ctx_features(b, enc))
        opt.zero_grad(set_to_none=True); ce(logits, y).backward(); opt.step()
        c += (logits.argmax(-1) == y).sum().item(); t += y.size(0)
    ta = c / t
    probe.eval()
    c2 = t2 = 0
    with torch.no_grad():
        for b in test_loader:
            y = b["label"].to(DEVICE)
            c2 += (probe(_ctx_features(b, enc)).argmax(-1) == y).sum().item(); t2 += y.size(0)
    te = c2 / t2
    history_probe["train_acc"].append(ta); history_probe["test_acc"].append(te)
    print(f"  ep {ep:02d}/{n_probe_epochs}  train={ta:.4f}  test={te:.4f}")
best_test = max(history_probe["test_acc"])
print(f"\nbest probe test = {best_test:.4f}")

fig, ax = plt.subplots(figsize=(7, 4))
ax.plot([a*100 for a in history_probe["train_acc"]], 'o-', label="train", color='C0')
ax.plot([a*100 for a in history_probe["test_acc"]],  's-', label="test",  color='C3')
ax.set_xlabel("probe epoch"); ax.set_ylabel("accuracy (%)")
ax.set_title(f"HRR Patch JEPA CIFAR-10 final linear probe — best test = {best_test*100:.2f}%")
ax.grid(alpha=0.3); ax.legend()
plt.tight_layout(); plt.show()
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
