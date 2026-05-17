"""Build standalone/rope_patch_cifar10.ipynb — RoPE Patch JEPA on CIFAR-10
with rich visualizations motivating the two-level rotary design."""
import json, os
NB = "/Users/davidpark/Documents/Claude/NFJEPA/OmniBird/standalone/rope_patch_cifar10.ipynb"

cells = []
def md(s):   cells.append({"cell_type":"markdown","metadata":{},"source":s.splitlines(keepends=True)})
def code(s): cells.append({"cell_type":"code","metadata":{},"execution_count":None,"outputs":[],"source":s.splitlines(keepends=True)})


# =============================================================================
md(r"""# RoPE Patch JEPA on CIFAR-10 — Rotary Patch Aggregation

This notebook proposes a new patch-aggregation algorithm for sparse inputs and
motivates it visually.

**Problem.** For each patch we have K events, each carrying:
- a content vector `c_i ∈ ℝᶜ`
- a continuous position `p_i ∈ ℝ³`

We want **one** vector per patch (like ViT) that:
- summarizes the joint (content, position) structure of all K events,
- is permutation-invariant in K,
- preserves multi-scale spatial information about *where* content lives.

The current `Patchifier` uses a mini-PointNet (MLP per event + max-pool). Max-pool
is severely lossy — most per-event information is discarded.

**Idea.** Treat the patch as a sparsely sampled complex-valued field and compute
its truncated **Non-Uniform DFT** (NUDFT) as the summary:

$$ S = \sum_i \text{proj}(c_i) \cdot e^{j \, \omega \cdot p_i} $$

Each channel pair `(c_{i,2l}, c_{i,2l+1})` is rotated by `ω_l · p_i` — exactly the
**RoPE** mechanism, but applied at the per-event level inside a patch and then
summed. This is what makes it parameter-free, multi-scale, and permutation-invariant.

**The catch — and the fix.** Aggregating with *relative* positions (`p_i − centroid`)
introduces a parasitic centroid-offset phase in cross-patch attention. This is
fixed by a **two-level RoPE**: within-patch RoPE on relative positions, plus
attention-level RoPE on centroids. The two phases cancel exactly.

The visualizations below build this picture from scratch.
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
from matplotlib.patches import FancyArrowPatch, Circle
import torchvision

from vit_fps_core import (
    NerfPosEnc,
    TargetCenter, ema_update, make_momentum_schedule,
    jepa_loss, short_params, save_atomic,
    farthest_point_sample, knn_indices,
)
from rope_patch_core import (
    RoPEPatchifier, RoPEViTEncoder, RoPEViTPredictor,
    rope_aggregate_complex,
)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
torch.manual_seed(0); np.random.seed(0)
N_GPUS = min(4, torch.cuda.device_count())
USE_DP = (DEVICE == "cuda") and (N_GPUS > 1)
print(f"GPUs visible = {torch.cuda.device_count()}  using = {N_GPUS}  DataParallel = {USE_DP}")
""")


# =============================================================================
md(r"""## 1. Visualization — what does "rotate content by position" actually mean?

Pick **one** channel pair `(c_even, c_odd)` and treat it as a complex number
`z = c_even + j·c_odd`. RoPE rotates this complex number by an angle proportional
to a position:

$$ z' = z \cdot e^{j \omega p} $$

Visualized below: a single event's content vector (red arrow) rotating around the
origin as its position sweeps from `p = −1` to `p = +1`, at different frequencies `ω`.

- Low ω: small rotation over the position range — barely moves.
- Medium ω: a partial sweep — useful for capturing structure.
- High ω: wraps around many times — captures fine-grained position differences,
  but aliases easily.
""")
code(r"""# A single complex content vector
z = np.array([1.0, 0.3])
positions = np.linspace(-1.0, 1.0, 60)

fig, axes = plt.subplots(1, 4, figsize=(16, 4))
for ax, omega in zip(axes, [0.5, math.pi, 2*math.pi, 6*math.pi]):
    # Trace the rotated point
    angles = positions * omega
    z_rot = np.stack([
        z[0]*np.cos(angles) - z[1]*np.sin(angles),
        z[0]*np.sin(angles) + z[1]*np.cos(angles),
    ], axis=1)
    ax.plot(z_rot[:, 0], z_rot[:, 1], '-', color='gray', alpha=0.5)
    # Draw a few colored vectors along the trace
    n_show = 6
    idxs = np.linspace(0, len(positions)-1, n_show).astype(int)
    cmap = plt.cm.viridis(np.linspace(0, 1, n_show))
    for c, i in zip(cmap, idxs):
        ax.annotate('', xy=z_rot[i], xytext=(0, 0),
                    arrowprops=dict(arrowstyle='->', color=c, lw=2))
        ax.text(z_rot[i, 0]*1.15, z_rot[i, 1]*1.15, f'p={positions[i]:.1f}',
                fontsize=7, ha='center', color=c)
    ax.add_patch(Circle((0, 0), 1.2, fill=False, ls=':', color='gray', lw=0.5))
    ax.set_aspect('equal')
    ax.set_xlim(-1.8, 1.8); ax.set_ylim(-1.8, 1.8)
    ax.axhline(0, color='k', lw=0.3); ax.axvline(0, color='k', lw=0.3)
    ax.set_title(f"ω = {omega:.2f}  →  swept angle = {(positions[-1]-positions[0])*omega:.1f} rad")
    ax.set_xlabel("Re (c_even)"); ax.set_ylabel("Im (c_odd)" if ax is axes[0] else "")
plt.suptitle("One channel pair rotates as the event's position changes.\n"
             "Different ω → different 'spatial frequencies' encoded in different channel pairs.")
plt.tight_layout(); plt.show()
""")


# =============================================================================
md(r"""## 2. Aggregation — summing K rotated content vectors

A patch has **K events**. Each event's content vector gets rotated by its own
position; we then **sum** the rotated vectors. The sum is permutation-invariant
(reordering events gives the same result).

In complex notation:
$$ S = \sum_{i=1}^{K} z_i \cdot e^{j \omega p_i} $$

This is the Non-Uniform DFT of the content sequence `{z_i}` sampled at positions
`{p_i}`, evaluated at frequency ω. The summary vector encodes "the spatial spectrum
of content inside this patch".
""")
code(r"""rng = np.random.RandomState(7)
K = 8
positions = rng.uniform(-1.0, 1.0, size=K)
contents  = rng.randn(K, 2) * 0.7
omega = 2 * math.pi

# Rotate each event's content
angles = positions * omega
rotated = np.stack([
    contents[:, 0]*np.cos(angles) - contents[:, 1]*np.sin(angles),
    contents[:, 0]*np.sin(angles) + contents[:, 1]*np.cos(angles),
], axis=1)
S = rotated.sum(axis=0)

fig, axes = plt.subplots(1, 3, figsize=(15, 5))

# (a) raw events (content vectors)
ax = axes[0]
cmap = plt.cm.tab10(np.arange(K) % 10)
for i in range(K):
    ax.annotate('', xy=contents[i], xytext=(0, 0),
                arrowprops=dict(arrowstyle='->', color=cmap[i], lw=1.5))
    ax.text(contents[i, 0]*1.1, contents[i, 1]*1.1, f'z_{i}\n(p={positions[i]:+.2f})',
            fontsize=7, color=cmap[i], ha='center')
ax.add_patch(Circle((0, 0), 1.5, fill=False, ls=':', color='gray', lw=0.5))
ax.set_aspect('equal'); ax.set_xlim(-2.0, 2.0); ax.set_ylim(-2.0, 2.0)
ax.axhline(0, color='k', lw=0.3); ax.axvline(0, color='k', lw=0.3)
ax.set_title(f"(a) K={K} raw content vectors z_i, each tagged with position p_i")

# (b) after rotation by p_i · ω
ax = axes[1]
for i in range(K):
    ax.annotate('', xy=rotated[i], xytext=(0, 0),
                arrowprops=dict(arrowstyle='->', color=cmap[i], lw=1.5))
    ax.text(rotated[i, 0]*1.1, rotated[i, 1]*1.1, f"z'_{i}",
            fontsize=7, color=cmap[i], ha='center')
ax.add_patch(Circle((0, 0), 1.5, fill=False, ls=':', color='gray', lw=0.5))
ax.set_aspect('equal'); ax.set_xlim(-2.0, 2.0); ax.set_ylim(-2.0, 2.0)
ax.axhline(0, color='k', lw=0.3); ax.axvline(0, color='k', lw=0.3)
ax.set_title(f"(b) After rotation z'_i = z_i · exp(jω·p_i) — ω={omega:.2f}")

# (c) the sum S
ax = axes[2]
# draw rotated arrows in light gray for context, then sum
for i in range(K):
    ax.annotate('', xy=rotated[i], xytext=(0, 0),
                arrowprops=dict(arrowstyle='->', color='lightgray', lw=1))
ax.annotate('', xy=S, xytext=(0, 0),
            arrowprops=dict(arrowstyle='->', color='red', lw=3))
ax.text(S[0]*1.1, S[1]*1.1, f"S = Σ z'_i\n  = ({S[0]:+.2f}, {S[1]:+.2f})",
        fontsize=10, color='red', ha='center', weight='bold')
ax.add_patch(Circle((0, 0), 1.5, fill=False, ls=':', color='gray', lw=0.5))
ax.set_aspect('equal'); ax.set_xlim(-3.5, 3.5); ax.set_ylim(-3.5, 3.5)
ax.axhline(0, color='k', lw=0.3); ax.axvline(0, color='k', lw=0.3)
ax.set_title(f"(c) Patch summary S = NUDFT of content at ω={omega:.2f}")

plt.tight_layout(); plt.show()
""")


# =============================================================================
md(r"""### Sanity check — permutation invariance

Shuffle the K events and recompute S. The summary must be identical (up to
floating-point noise), since addition is commutative.
""")
code(r"""perm_S = []
for _ in range(5):
    p = rng.permutation(K)
    angles_p = positions[p] * omega
    rotated_p = np.stack([
        contents[p, 0]*np.cos(angles_p) - contents[p, 1]*np.sin(angles_p),
        contents[p, 0]*np.sin(angles_p) + contents[p, 1]*np.cos(angles_p),
    ], axis=1)
    perm_S.append(rotated_p.sum(axis=0))
perm_S = np.stack(perm_S)
print("S from 5 different shufflings:")
for i, s in enumerate(perm_S):
    print(f"  shuffle {i}: S = ({s[0]:+.6f}, {s[1]:+.6f})")
print(f"\nMax difference from S = ({S[0]:+.6f}, {S[1]:+.6f}): "
      f"{np.abs(perm_S - S).max():.2e}")
""")


# =============================================================================
md(r"""## 3. The drawback — relative positions kill all spatial structure in attention

In the JEPA pipeline we have many patches across the input. Each patch's events
are positioned RELATIVE to **its own centroid**: `rel_pos_i = p_i − centroid_A`.

This is natural — it gives a local frame of reference per patch. But it has a
*severe* side effect on downstream attention.

### The math

Two patches A and B with summaries:
$$ S_A = \sum_i z^A_i \cdot e^{j\omega (p^A_i - c_A)} \quad,\quad
    S_B = \sum_j z^B_j \cdot e^{j\omega (p^B_j - c_B)} $$

Notice that **`S_A` depends only on `rel_pos_i = p^A_i − c_A`**, not on `c_A`
itself. Two patches with identical internal structure but at different absolute
centroids produce **literally identical** summaries — the patch token doesn't
know where in the image it sits.

So when attention computes `S_A · S_B`, the result is a function of internal
content only. **There is zero spatial structure for attention to scaffold on.**

By contrast, the "ideal" summary using absolute positions:
$$ S^{abs}_A = \sum_i z^A_i \cdot e^{j\omega p^A_i} = e^{j\omega c_A} \cdot S_A $$

has the property that `<S^{abs}_A, S^{abs}_B>` carries the centroid relative
phase `e^{jω(c_A − c_B)}` — exactly the spatial structure attention needs.

Let's visualize the gap.
""")
code(r"""# Set up: two patches A and B with identical INTERNAL structure
# (same rel_pos and content), but different absolute centroids. Vary c_B.
rng2 = np.random.RandomState(11)
K = 6
omega_list = [0.5*math.pi, math.pi, 3*math.pi, 8*math.pi]

# Internal structure (shared by A and B)
rel_pos = rng2.uniform(-0.15, 0.15, size=K)        # small intra-patch range
content = rng2.randn(K, 2) * 0.5

c_A = 0.0
c_B_range = np.linspace(-1.0, 1.0, 200)

def aggregate(rel_pos, content, omega):
    ang = rel_pos * omega
    re = content[:, 0]*np.cos(ang) - content[:, 1]*np.sin(ang)
    im = content[:, 0]*np.sin(ang) + content[:, 1]*np.cos(ang)
    return np.array([re.sum(), im.sum()])

def aggregate_abs(rel_pos, content, c, omega):
    ang = (rel_pos + c) * omega
    re = content[:, 0]*np.cos(ang) - content[:, 1]*np.sin(ang)
    im = content[:, 0]*np.sin(ang) + content[:, 1]*np.cos(ang)
    return np.array([re.sum(), im.sum()])

fig, axes = plt.subplots(2, 2, figsize=(14, 9))
for ax, omega in zip(axes.flat, omega_list):
    # NAIVE rel-aggregation: S_A and S_B are identical regardless of c_B
    sa = aggregate(rel_pos, content, omega)
    ip_rel = np.full_like(c_B_range, sa @ sa)

    # IDEAL abs-aggregation: <S_A_abs(c_A=0), S_B_abs(c_B)>
    sa_abs = aggregate_abs(rel_pos, content, c_A, omega)
    ip_abs = []
    for c_B in c_B_range:
        sb_abs = aggregate_abs(rel_pos, content, c_B, omega)
        ip_abs.append(sa_abs @ sb_abs)
    ip_abs = np.array(ip_abs)

    ax.plot(c_B_range, ip_abs, '-', color='C2', lw=2.5,
            label='IDEAL: abs-aggregation  ⇒  spatial structure preserved')
    ax.plot(c_B_range, ip_rel, '-', color='C3', lw=2.5,
            label='NAIVE: rel-aggregation  ⇒  flat — no spatial info at all')
    ax.axhline(0, color='k', lw=0.3); ax.axvline(0, color='k', lw=0.3)
    ax.set_xlabel("c_B (absolute centroid of patch B)")
    ax.set_ylabel("inner product <S_A, S_B>")
    ax.set_title(f"ω = {omega:.2f}    "
                 f"({(omega/(2*math.pi)):.2f} cycles per unit of centroid)")
    ax.legend(fontsize=8, loc='upper right')
    ax.grid(alpha=0.3)

plt.suptitle("Naive relative-aggregation: the inner product between two patch tokens\n"
             "is COMPLETELY FLAT in the centroid offset — attention sees no spatial\n"
             "structure. The patch token literally doesn't know where it is.",
             y=1.02)
plt.tight_layout(); plt.show()
""")


# =============================================================================
md(r"""**Reading the plot.** Two curves per frequency:

- **Green (ideal):** what you'd get if you aggregated with absolute positions —
  the inner product oscillates with `c_B − c_A` at frequency ω, encoding the
  centroid relative position. This is the spatial signal attention needs.
- **Red (naive rel-agg):** completely flat. Two patches with the same internal
  structure produce identical summaries regardless of their absolute positions.
  Attention has no way to tell them apart spatially. The patch token has lost
  all absolute-position information.

You might think "well, just add a NeRF γ(centroid) to the patch token" — and
indeed that recovers *some* position awareness, but only as a separate additive
embedding, not as a multiplicative *modulation* of the content. The content
itself remains centroid-blind. The two-level RoPE approach in §4 fixes this at
the source.
""")


# =============================================================================
md(r"""## 4. The fix — two-level RoPE

Apply RoPE **again** at the attention level, using each patch's centroid:

$$ Q_A = S_A \cdot e^{j\omega c_A} \quad,\quad K_B = S_B \cdot e^{j\omega c_B} $$

Then `Q_A · conj(K_B) = S_A · conj(S_B) · e^{jω(c_A − c_B)}`. The centroid
relative phase that the within-patch RoPE had **discarded** is now **restored**
multiplicatively. The total effect is identical to absolute-position aggregation:

$$ Q_A \cdot \overline{K_B} \;=\; \sum_{i,j} z^A_i \, \overline{z^B_j} \cdot e^{j\omega(p^A_i - p^B_j)} \;=\; S^{abs}_A \cdot \overline{S^{abs}_B} $$

**Two levels:**
- **Level 1 (within-patch):** rotate event content by relative position, sum.
  Permutation-invariant, captures intra-patch structure.
- **Level 2 (cross-patch attention):** rotate Q, K by centroids. Restores
  inter-patch spatial structure.

Below: same setup as §3 but with centroid-RoPE applied at the attention level.
The flat red curve becomes the green ideal.
""")
code(r"""fig, axes = plt.subplots(2, 2, figsize=(14, 9))
for ax, omega in zip(axes.flat, omega_list):
    sa = aggregate(rel_pos, content, omega)         # S_A from rel-aggregation
    # S_B in rel-aggregation == S_A (same internal structure)

    def centroid_rope(s, c, om):
        ang = om * c
        return np.array([s[0]*np.cos(ang) - s[1]*np.sin(ang),
                         s[0]*np.sin(ang) + s[1]*np.cos(ang)])

    ip_fixed = []
    for c_B in c_B_range:
        Q_A = centroid_rope(sa, c_A, omega)
        K_B = centroid_rope(sa, c_B, omega)
        ip_fixed.append(Q_A @ K_B)
    ip_fixed = np.array(ip_fixed)

    sa_abs = aggregate_abs(rel_pos, content, c_A, omega)
    ip_abs = []
    for c_B in c_B_range:
        sb_abs = aggregate_abs(rel_pos, content, c_B, omega)
        ip_abs.append(sa_abs @ sb_abs)
    ip_abs = np.array(ip_abs)

    ax.plot(c_B_range, ip_abs, '-', color='C2', lw=4,
            label='IDEAL (abs-aggregation)')
    ax.plot(c_B_range, ip_fixed, '--', color='C0', lw=2,
            label='Two-level RoPE  (rel-agg + centroid-RoPE attention)')
    ax.axhline(0, color='k', lw=0.3); ax.axvline(0, color='k', lw=0.3)
    ax.set_xlabel("c_B (absolute centroid of patch B)")
    ax.set_ylabel("inner product Q_A · K_B")
    ax.set_title(f"ω = {omega:.2f}    →  curves overlap exactly")
    ax.legend(fontsize=8, loc='upper right')
    ax.grid(alpha=0.3)

plt.suptitle("Two-level RoPE: the centroid-RoPE applied in attention reintroduces\n"
             "the spatial phase that within-patch relative aggregation had discarded.\n"
             "Inner product MATCHES the absolute-aggregation ideal exactly.",
             y=1.02)
plt.tight_layout(); plt.show()
""")


# =============================================================================
md(r"""## 5. Scale mismatch — within-patch vs cross-patch frequencies

The two-level cancellation is exact in the math, but the optimal **range of phase
angles** is very different for the two RoPEs:

- **Within-patch** positions are small (`rel_pos ≈ [−0.1, 0.1]` for typical FPS+KNN
  patches). To sweep a meaningful phase range, ω must be **large**.
- **Cross-patch** centroids span the full input (`c ≈ [−1, 1]`). At the same ω,
  centroid offsets sweep many full rotations — aliasing.

A single frequency schedule can't be optimal for both. Below: at each frequency,
plot the "phase angle range" actually swept by within-patch motion and by
cross-patch motion. The useful regime is roughly `phase ∈ [0.1, 2π]` — below 0.1
the rotation is too small to be informative, above 2π we alias.
""")
code(r"""# Sweep frequency
freqs = np.logspace(-1, 2, 80)
within_patch_range = 0.1     # ~half-radius of a typical FPS+KNN patch
cross_patch_range  = 2.0      # full coord span [-1, 1]

phase_within = freqs * within_patch_range
phase_cross  = freqs * cross_patch_range

fig, ax = plt.subplots(figsize=(9, 5.5))
ax.loglog(freqs, phase_within, '-', color='C0', lw=2.5, label='within-patch phase range')
ax.loglog(freqs, phase_cross,  '-', color='C3', lw=2.5, label='cross-patch phase range')
ax.axhspan(0.1, 2*math.pi, color='C2', alpha=0.1, label='useful range  [0.1, 2π]')
ax.axhline(0.1, color='C2', ls=':', lw=1)
ax.axhline(2*math.pi, color='C2', ls=':', lw=1)
ax.text(freqs[0]*1.1, 0.105, 'too small (no rotation)', fontsize=8, color='C2')
ax.text(freqs[0]*1.1, 6.5, 'too large (aliasing)', fontsize=8, color='C2')

# Mark sweet spots
sweet_within = freqs[(phase_within > 0.1) & (phase_within < 2*math.pi)]
sweet_cross  = freqs[(phase_cross  > 0.1) & (phase_cross  < 2*math.pi)]
ax.axvspan(sweet_within.min(), sweet_within.max(), color='C0', alpha=0.08)
ax.axvspan(sweet_cross.min(),  sweet_cross.max(),  color='C3', alpha=0.08)

ax.set_xlabel("frequency ω")
ax.set_ylabel("phase angle swept (radians)")
ax.set_title("Within-patch and cross-patch need different ω ranges.\n"
             "Sweet-spot frequencies (shaded) barely overlap.")
ax.legend(loc='lower right')
ax.grid(alpha=0.3, which='both')
plt.tight_layout(); plt.show()

print(f"Within-patch sweet-spot ω ≈ [{sweet_within.min():.2f},  {sweet_within.max():.2f}]")
print(f"Cross-patch  sweet-spot ω ≈ [{sweet_cross.min():.2f},  {sweet_cross.max():.2f}]")
print(f"Overlap: ω ≈ [{max(sweet_within.min(), sweet_cross.min()):.2f}, "
      f"{min(sweet_within.max(), sweet_cross.max()):.2f}]")
""")
md(r"""**Implication.** With a single RoPE base for both within-patch and cross-patch,
the two regimes can't both be in their sweet spots. Standard RoPE uses **log-spaced
frequencies** across channel pairs — a multi-scale schedule that naturally spans
both regimes, at the cost of having "useless" channels at each end (low channels
encode only cross-patch position; high channels encode only within-patch position).

This is the "scale-mismatch" drawback. In practice it manifests as some channels
being effectively dead and others being aliased. The fix is:

1. **Use different bases** for within-patch and cross-patch RoPE (we expose
   `base_within` and `base_cross` separately in `RoPEViTEncoder`). This breaks
   exact mathematical cancellation but each regime gets a tuned schedule.
2. **Or accept the loss** and use a single base = some compromise value
   (e.g. `base = 100` gives the most useful overlap on `[−1, 1]` coords).

We use option 1 by default — different bases — and rely on the encoder to learn
which channels carry which kind of information.
""")


# =============================================================================
md(r"""## 6. Implementation summary

`RoPEPatchifier` (axial RoPE inside each patch):
- For each event, project signal → d_model real-valued content.
- Split d_model into 2 axis groups, each of `d_model // 2` channels = pairs of
  (real, imag).
- For each pair `l`, rotate by angle `inv_freq_l · rel_pos`.
- Sum / mean across the K events → one d_model vector per patch.

`CentroidRoPEMultiHeadAttention` (level-2 RoPE):
- Standard MHA, except Q and K are rotated by centroid coords using the same
  axial-RoPE pattern at the per-head level.

`RoPEViTEncoder`:
- `RoPEPatchifier` → optional NeRF γ(centroid) added → stack of
  `RoPETransformerBlock` (each applies centroid-RoPE in attention).

Below: instantiate, forward a dummy batch, and inspect output shapes & magnitudes.
""")
code(r"""# Dummy batch
B, P, K = 2, 64, 16
coord_dim = 2
signal_dim = 3
patch_events_dummy    = torch.randn(B, P, K, coord_dim + signal_dim)
patch_centroids_dummy = torch.randn(B, P, coord_dim) * 0.8

patchifier = RoPEPatchifier(signal_dim=3, coord_dim=2, d_model=256, base=100.0)
print(f"RoPEPatchifier params: {sum(p.numel() for p in patchifier.parameters())/1e3:.1f} K")
print(f"  channels_per_axis = {patchifier.channels_per_axis}  pairs_per_axis = {patchifier.pairs_per_axis}")
print(f"  inv_freq = [{patchifier.inv_freq[0]:.4f},  ...,  {patchifier.inv_freq[-1]:.4f}]")
out = patchifier(patch_events_dummy, patch_centroids_dummy)
print(f"\nout shape = {tuple(out.shape)}")
print(f"out stats: mean={out.mean().item():+.3f}  std={out.std().item():.3f}  "
      f"abs_max={out.abs().max().item():.3f}")

# And the full encoder
enc = RoPEViTEncoder(
    signal_dim=3, coord_dim=2, d_model=256,
    n_layers=2, n_heads=8, dim_head=32,
    base_within=100.0, base_cross=100.0,
    add_nerf_centroid=True, n_freqs=10,
)
print(f"\nRoPEViTEncoder (2 layers) params: {sum(p.numel() for p in enc.parameters())/1e6:.2f} M")
g = enc(patch_events_dummy, patch_centroids_dummy)
print(f"encoder out shape = {tuple(g.shape)}")
""")


# =============================================================================
md(r"""## 7. Sanity check — verify two-level RoPE cancellation on real tensors

Build two identical-structure patches at different centroids using the real
`RoPEPatchifier` (not the toy 1-pair version from §3). Pass them through the
encoder with and without centroid-RoPE attention; compare the patch-token cosine
similarities across centroid offsets.
""")
code(r"""torch.manual_seed(0)
K = 16
coord_dim = 2
signal_dim = 3
d_model = 256

# Build a "template" patch: K random events with content. Place this same
# template at a sweep of centroid offsets. The internal structure (rel_pos +
# content) is identical for all of them; only the absolute centroid differs.
template_rel = torch.randn(K, coord_dim) * 0.1     # tight intra-patch
template_sig = torch.randn(K, signal_dim) * 0.5

n_cen = 41
centroid_xs = torch.linspace(-1.0, 1.0, n_cen)
centroids = torch.stack([centroid_xs, torch.zeros(n_cen)], dim=-1)        # (n_cen, 2)
patch_centroids = centroids.unsqueeze(0)                                    # (1, n_cen, 2)
patch_events = torch.zeros(1, n_cen, K, coord_dim + signal_dim)
for p in range(n_cen):
    patch_events[0, p, :, :coord_dim] = template_rel + centroids[p]
    patch_events[0, p, :, coord_dim:] = template_sig

# (a) Patchifier output is INVARIANT to centroid (uses rel_pos only)
patchifier = RoPEPatchifier(signal_dim=3, coord_dim=2, d_model=256, base=100.0).eval()
with torch.no_grad():
    g0 = patchifier(patch_events, patch_centroids)   # (1, n_cen, 256)
g0n = F.normalize(g0[0], dim=-1)
cos_patchifier = (g0n @ g0n[n_cen//2].unsqueeze(-1)).squeeze(-1).cpu().numpy()

# (b) After ENCODER with centroid-RoPE attention: patch tokens differ
enc_rope = RoPEViTEncoder(
    signal_dim=3, coord_dim=2, d_model=256,
    n_layers=2, n_heads=8, dim_head=32,
    base_within=100.0, base_cross=100.0,
    add_nerf_centroid=False,        # isolate the RoPE attention contribution
    n_freqs=10,
).eval()
with torch.no_grad():
    g1 = enc_rope(patch_events, patch_centroids)
g1n = F.normalize(g1[0], dim=-1)
cos_enc_rope = (g1n @ g1n[n_cen//2].unsqueeze(-1)).squeeze(-1).cpu().numpy()

fig, ax = plt.subplots(figsize=(10, 5))
ax.plot(centroid_xs.numpy(), cos_patchifier, 'o-', color='gray', lw=2,
        label='Just RoPEPatchifier (centroid-blind, cos = 1 everywhere)')
ax.plot(centroid_xs.numpy(), cos_enc_rope, 's-', color='C0', lw=2,
        label='Encoder with centroid-RoPE attention (patches differentiated)')
ax.axvline(0.0, color='k', lw=0.4, ls=':')
ax.axhline(1.0, color='k', lw=0.4, ls=':')
ax.set_xlabel("centroid offset from reference patch")
ax.set_ylabel("cosine similarity to reference patch token")
ax.set_title("RoPEPatchifier alone is centroid-invariant.\n"
             "Centroid-RoPE attention restores absolute-position structure.")
ax.legend(loc='lower right'); ax.grid(alpha=0.3)
plt.tight_layout(); plt.show()
""")


# =============================================================================
md(r"""## 8. Configuration & build the actual training pipeline""")
code(r"""IMAGE_SIZE = 32
FRAC_POOL  = 0.4
K_POOL     = int(round(FRAC_POOL * IMAGE_SIZE * IMAGE_SIZE))   # 410

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
BASE_WITHIN  = 30.0     # within-patch: tighter range, higher freq
BASE_CROSS   = 100.0    # cross-patch:  wider range, lower freq

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
CKPT_DIR       = "./checkpoints_rope_patch_cifar10"
os.makedirs(CKPT_DIR, exist_ok=True)
print(f"K_pool={K_POOL}  N_patches={N_PATCHES}  K_neigh={K_NEIGH}  N_tgt={N_TGT}  N_ctx={N_CTX}")
print(f"base_within={BASE_WITHIN}  base_cross={BASE_CROSS}")
""")


# =============================================================================
md(r"""## 9. Dataset — same FPS+KNN pipeline as `vit_fps_cifar10`""")
code(r"""class FPSPatchCIFAR10(Dataset):
    def __init__(self, base, train=True, pool_seed=0, precompute_seed=42):
        self.base = base
        self.train = train
        self.N_pix = IMAGE_SIZE * IMAGE_SIZE
        rng = np.random.RandomState(pool_seed)
        self.pool_idx = np.stack(
            [rng.permutation(self.N_pix)[:K_POOL] for _ in range(len(base))],
            axis=0,
        ).astype(np.int64)
        ys, xs = torch.meshgrid(
            torch.linspace(-1.0, 1.0, IMAGE_SIZE),
            torch.linspace(-1.0, 1.0, IMAGE_SIZE),
            indexing='ij',
        )
        self.coords_all = torch.stack([ys, xs], dim=-1).view(self.N_pix, 2).float()

        print(f"  precomputing FPS+KNN for {len(base)} samples...")
        t0 = time.time()
        torch.manual_seed(precompute_seed)
        centroid_idx_all = np.zeros((len(base), N_PATCHES), dtype=np.int64)
        nbr_idx_all      = np.zeros((len(base), N_PATCHES, K_NEIGH), dtype=np.int64)
        for i in range(len(base)):
            pc = self.coords_all[self.pool_idx[i]].unsqueeze(0)
            cen_idx = farthest_point_sample(pc, N_PATCHES).squeeze(0)
            cen_coords = pc[0, cen_idx]
            nbrs = knn_indices(cen_coords.unsqueeze(0), pc, K_NEIGH).squeeze(0)
            centroid_idx_all[i] = cen_idx.numpy()
            nbr_idx_all[i]      = nbrs.numpy()
        self.centroid_idx_all = centroid_idx_all
        self.nbr_idx_all      = nbr_idx_all
        print(f"  done in {time.time()-t0:.1f}s")

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
            rng = np.random.RandomState()
            cen_np = centroid_coords.numpy()
            n_p = N_PATCHES

            exclude = np.zeros(n_p, dtype=bool)
            tgt_blocks = []
            for _ in range(N_TGT_BLOCKS):
                allowed = np.where(~exclude)[0]
                if len(allowed) < N_PATCH_PER_BLOCK: break
                anchor = allowed[rng.randint(len(allowed))]
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
                ctx_anchor = rng.randint(n_p)
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
md("## 10. Visualize FPS patches + masking on one sample")
code(r"""classes = ['plane','car','bird','cat','deer','dog','frog','horse','ship','truck']
sample = train_ds[0]
print(f"label = {classes[sample['label']]}")

pe  = sample["patch_events"].numpy()
pc  = sample["patch_centroids"].numpy()
ctx = sample["ctx_idx"].numpy()
tgt = sample["tgt_idx"].numpy()

fig, axes = plt.subplots(1, 3, figsize=(15, 5))
pool_events = pe.reshape(-1, 5)
ax = axes[0]
ax.scatter(pool_events[:, 1], -pool_events[:, 0], s=8, c='lightgray', alpha=0.4)
ax.scatter(pc[:, 1], -pc[:, 0], s=30, c='red', marker='x', label='FPS centroids')
ax.set_aspect('equal'); ax.set_title(f"(a) FPS gives {N_PATCHES} centroids"); ax.legend(fontsize=8)
ax.set_xlim(-1.05, 1.05); ax.set_ylim(-1.05, 1.05)

ax = axes[1]
rng_cmap = np.random.RandomState(0)
patch_colors = plt.cm.tab20(rng_cmap.permutation(N_PATCHES) % 20)
for p in range(N_PATCHES):
    nbrs = pe[p, :, :2]
    ax.scatter(nbrs[:, 1], -nbrs[:, 0], s=10, c=[patch_colors[p]] * K_NEIGH)
ax.scatter(pc[:, 1], -pc[:, 0], s=15, c='black', marker='+')
ax.set_aspect('equal'); ax.set_title(f"(b) K-NN={K_NEIGH} around each centroid")
ax.set_xlim(-1.05, 1.05); ax.set_ylim(-1.05, 1.05)

ax = axes[2]
ax.scatter(pool_events[:, 1], -pool_events[:, 0], s=4, c='lightgray', alpha=0.3)
ax.scatter(pc[ctx, 1], -pc[ctx, 0], s=40, c='#ef4444', label='context')
TGT_COLORS = ['#fbbf24', '#34d399', '#60a5fa', '#f472b6']
for k in range(N_TGT_BLOCKS):
    blk = tgt[k*N_PATCH_PER_BLOCK:(k+1)*N_PATCH_PER_BLOCK]
    ax.scatter(pc[blk, 1], -pc[blk, 0], s=80, c=TGT_COLORS[k], marker='s',
                label=f'tgt block {k+1}')
ax.set_aspect('equal'); ax.set_title("(c) ctx (red) + 4 target blocks (colors)")
ax.legend(fontsize=8); ax.set_xlim(-1.05, 1.05); ax.set_ylim(-1.05, 1.05)
plt.tight_layout(); plt.show()
""")


# =============================================================================
md(r"""## 11. Build models — two-level RoPE encoder + RoPE predictor""")
code(r"""context_encoder = RoPEViTEncoder(
    signal_dim=3, coord_dim=2, d_model=D_MODEL,
    n_layers=N_LAYERS_ENC, n_heads=N_HEADS, dim_head=DIM_HEAD,
    ffn_mult=FFN_MULT, base_within=BASE_WITHIN, base_cross=BASE_CROSS,
    add_nerf_centroid=True, n_freqs=N_FREQS,
).to(DEVICE)
target_encoder = copy.deepcopy(context_encoder).to(DEVICE)
for p in target_encoder.parameters(): p.requires_grad_(False)

predictor = RoPEViTPredictor(
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

LAST = os.path.join(CKPT_DIR, "rope_patch_last.pt")
BEST = os.path.join(CKPT_DIR, "rope_patch_best.pt")
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
md(r"""## 13. Probe (mean-pool over context patch features)""")
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
ax.set_title(f"RoPE Patch JEPA CIFAR-10 final linear probe — best test = {best_test*100:.2f}%")
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
