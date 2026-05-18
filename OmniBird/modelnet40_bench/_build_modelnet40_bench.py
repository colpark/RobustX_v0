"""Build modelnet40_bench/modelnet40_bench.ipynb — head-to-head comparison
of PointNet vs RoPE/HRR patch aggregator on ModelNet40 classification."""
import json, os
NB = "/Users/davidpark/Documents/Claude/NFJEPA/OmniBird/modelnet40_bench/modelnet40_bench.ipynb"

cells = []
def md(s):   cells.append({"cell_type":"markdown","metadata":{},"source":s.splitlines(keepends=True)})
def code(s): cells.append({"cell_type":"code","metadata":{},"execution_count":None,"outputs":[],"source":s.splitlines(keepends=True)})


# =============================================================================
md(r"""# ModelNet40 — PointNet vs RoPE/HRR aggregator (head-to-head)

A minimal benchmark testing **whether the patch aggregator alone matters** on
ModelNet40 classification, holding everything else fixed.

**Setup.** ModelNet40 (40-class 3D point-cloud classification, ~10K samples).
Each cloud is sampled to 1024 points; FPS picks 64 centroids; each patch is
the K-NN=32 around its centroid. The patch is summarized by a single d_model
vector, and the encoder + classifier head process the resulting 64-token
sequence in the standard way.

**The only thing that varies between the two methods is the aggregator:**

| | PointNet | RoPE/HRR |
|---|---|---|
| Per-point op | `MLP(concat(rel_coord, signal))` | `MLP(signal)` then rotate by rel_coord |
| Cross-point op | **max-pool** over K | **sum** over K |
| Multi-scale position | learned by MLP | built-in via log-spaced frequencies |
| Position-content interaction | concatenation through MLP | multiplicative modulation (NUDFT) |

**Identical** for both:
- FPS+KNN patch construction (cached to disk)
- NeRF γ(centroid) added to patch tokens (absolute position info)
- 6-layer ViT encoder with vanilla MHA (no RoPE in attention — the
  comparison isolates aggregator effects only)
- Mean-pool → MLP head for classification
- AdamW + cosine LR, 100 epochs, batch=32
- Same augmentations: rotation around z, scale, jitter

**Per-point "content".** Vanilla ModelNet40 points are just (x, y, z) with
no per-point features. We pass a constant `signal = 1` per point so both
aggregators have the same input shape. With constant content, the RoPE
aggregator reduces to `Σ_i signal_proj(1) · exp(jω · rel_pos_i)` — literally
a **truncated Fourier transform of the patch's point density**. This is the
regime where RoPE's spatial-density encoding should be its strongest.
""")


# =============================================================================
md("## 0. Setup")
code(r"""import os, sys, math, time, copy
sys.path.insert(0, os.path.abspath('.'))                                       # bench_core
sys.path.insert(0, os.path.abspath(os.path.join('..', 'standalone')))          # vit_fps_core, rope_patch_core

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset
import matplotlib.pyplot as plt

from bench_core import (
    FlexibleViTEncoder, ModelNet40Classifier,
    PointNetPatchifier, RoPEPatchifier,
    download_modelnet40, load_modelnet40,
    precompute_fps_knn_modelnet,
    augment_pointcloud, normalize_to_unit_sphere,
    short_params, save_atomic,
)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
torch.manual_seed(0); np.random.seed(0)
N_GPUS = min(4, torch.cuda.device_count())
USE_DP = (DEVICE == "cuda") and (N_GPUS > 1)
print(f"GPUs visible = {torch.cuda.device_count()}  using = {N_GPUS}  DataParallel = {USE_DP}")
""")


# =============================================================================
md("## 1. Config")
code(r"""# ── Data ───────────────────────────────────────────────────────────────────
N_INPUT     = 1024            # points per cloud after subsampling from 2048
N_PATCHES   = 64              # FPS centroids per cloud
K_NEIGH     = 32              # K-NN per patch
COORD_DIM   = 3               # (x, y, z)
SIGNAL_DIM  = 1               # constant content per point

# ── Model (IDENTICAL backbone for both aggregators) ────────────────────────
D_MODEL      = 192            # divisible by 2*COORD_DIM=6, so RoPE pairs cleanly
N_LAYERS_ENC = 6
N_HEADS      = 6
DIM_HEAD     = 32             # n_heads × dim_head = 192 = d_model
FFN_MULT     = 4
N_FREQS      = 8
BASE_WITHIN  = 30.0           # RoPE base frequency for within-patch positions

# ── Training ───────────────────────────────────────────────────────────────
EPOCHS         = 100
BATCH_SIZE     = 32
LR             = 1e-3
WEIGHT_DECAY   = 1e-4
WARMUP_EPOCHS  = 5
LOG_EVERY      = 50
LABEL_SMOOTHING = 0.1

CKPT_ROOT = "./checkpoints_mn40_bench"
os.makedirs(CKPT_ROOT, exist_ok=True)
print(f"N_INPUT={N_INPUT}  N_PATCHES={N_PATCHES}  K_NEIGH={K_NEIGH}")
print(f"D_MODEL={D_MODEL}  layers={N_LAYERS_ENC}  heads={N_HEADS}  dim_head={DIM_HEAD}")
""")


# =============================================================================
md("## 2. Download & load ModelNet40 (~400MB)")
code(r"""train_pts_raw, train_lbl = load_modelnet40(train=True)
test_pts_raw,  test_lbl  = load_modelnet40(train=False)
print(f"raw train: {train_pts_raw.shape}  test: {test_pts_raw.shape}")

# Subsample N_INPUT points from each cloud (deterministic per cloud)
def subsample_clouds(pts, n_target, seed=0):
    rng = np.random.RandomState(seed)
    N, K, D = pts.shape
    out = np.empty((N, n_target, D), dtype=pts.dtype)
    for i in range(N):
        idx = rng.permutation(K)[:n_target]
        out[i] = pts[i, idx]
    return out

train_pts = subsample_clouds(train_pts_raw, N_INPUT, seed=0)
test_pts  = subsample_clouds(test_pts_raw,  N_INPUT, seed=1)
# Normalize each cloud to unit sphere (canonical form — FPS will operate on this)
train_pts = normalize_to_unit_sphere(train_pts)
test_pts  = normalize_to_unit_sphere(test_pts)
print(f"after subsample+normalize:  train {train_pts.shape}  test {test_pts.shape}")

class_names_path = os.path.expanduser("~/data/modelnet40_ply_hdf5_2048/shape_names.txt")
with open(class_names_path) as f:
    CLASS_NAMES = [l.strip() for l in f if l.strip()]
assert len(CLASS_NAMES) == 40
print(f"classes ({len(CLASS_NAMES)}):  {CLASS_NAMES[:8]} ...")
""")


# =============================================================================
md("## 3. Precompute FPS + K-NN (cached)")
code(r"""train_cen, train_nbr = precompute_fps_knn_modelnet(
    train_pts, N_PATCHES, K_NEIGH, seed=42, tag="train",
)
test_cen, test_nbr = precompute_fps_knn_modelnet(
    test_pts,  N_PATCHES, K_NEIGH, seed=42, tag="test",
)
print(f"train_cen {train_cen.shape}  train_nbr {train_nbr.shape}")
print(f"test_cen  {test_cen.shape}   test_nbr  {test_nbr.shape}")
""")


# =============================================================================
md("## 4. Visualize a few clouds + their FPS patches")
code(r"""fig = plt.figure(figsize=(15, 5))
for i, idx in enumerate([0, 100, 500]):
    ax = fig.add_subplot(1, 3, i+1, projection='3d')
    pts = train_pts[idx]
    ax.scatter(pts[:, 0], pts[:, 1], pts[:, 2], s=2, c='lightgray', alpha=0.5)
    cen = pts[train_cen[idx]]
    ax.scatter(cen[:, 0], cen[:, 1], cen[:, 2], s=40, c='red', marker='x')
    ax.set_title(f"{CLASS_NAMES[train_lbl[idx]]}  ({N_PATCHES} FPS centroids in red)")
    ax.set_box_aspect((1, 1, 1))
plt.tight_layout(); plt.show()
""")


# =============================================================================
md("## 5. Dataset class")
code(r"""class ModelNet40PatchDataset(Dataset):
    def __init__(self, points, labels, cen_idx, nbr_idx, train=False):
        self.points = points
        self.labels = labels
        self.cen_idx = cen_idx
        self.nbr_idx = nbr_idx
        self.train = train

    def __len__(self): return len(self.points)

    def __getitem__(self, idx):
        pts = self.points[idx]                                # (N_INPUT, 3) canonical
        if self.train:
            pts = augment_pointcloud(pts)
        cen = self.cen_idx[idx]                                # (N_PATCHES,)
        nbr = self.nbr_idx[idx]                                # (N_PATCHES, K_NEIGH)
        centroids = pts[cen]                                   # (N_PATCHES, 3)
        nbrs = pts[nbr]                                        # (N_PATCHES, K_NEIGH, 3)
        signal = np.ones((N_PATCHES, K_NEIGH, SIGNAL_DIM), dtype=np.float32)
        patch_events = np.concatenate([nbrs, signal], axis=-1)  # (N_PATCHES, K_NEIGH, 3+1)
        return {
            "patch_events":   torch.from_numpy(patch_events.astype(np.float32)).contiguous(),
            "patch_centroids": torch.from_numpy(centroids.astype(np.float32)).contiguous(),
            "label":          int(self.labels[idx]),
        }


train_ds = ModelNet40PatchDataset(train_pts, train_lbl, train_cen, train_nbr, train=True)
test_ds  = ModelNet40PatchDataset(test_pts,  test_lbl,  test_cen,  test_nbr,  train=False)
train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,  num_workers=0)
test_loader  = DataLoader(test_ds,  batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
print(f"train batches={len(train_loader)}  test batches={len(test_loader)}")
""")


# =============================================================================
md(r"""## 6. Build the two models — identical encoder backbone, different aggregators
""")
code(r"""def build_model(aggregator_name: str):
    if aggregator_name == "pointnet":
        patchifier = PointNetPatchifier(
            signal_dim=SIGNAL_DIM, coord_dim=COORD_DIM, d_model=D_MODEL,
        )
    elif aggregator_name == "rope":
        patchifier = RoPEPatchifier(
            signal_dim=SIGNAL_DIM, coord_dim=COORD_DIM, d_model=D_MODEL,
            base=BASE_WITHIN, agg="mean",
        )
    else:
        raise ValueError(aggregator_name)
    encoder = FlexibleViTEncoder(
        patchifier=patchifier, coord_dim=COORD_DIM, d_model=D_MODEL,
        n_layers=N_LAYERS_ENC, n_heads=N_HEADS, dim_head=DIM_HEAD,
        ffn_mult=FFN_MULT, n_freqs=N_FREQS,
    )
    classifier = ModelNet40Classifier(encoder, d_model=D_MODEL, n_classes=40)
    return classifier


# Quick sanity: both models have the same param count except for the patchifier
m_pn   = build_model("pointnet")
m_rope = build_model("rope")
print(f"PointNet aggregator: total {short_params(m_pn)}")
print(f"  patchifier alone:  {short_params(m_pn.encoder.patchifier)}")
print(f"RoPE aggregator:     total {short_params(m_rope)}")
print(f"  patchifier alone:  {short_params(m_rope.encoder.patchifier)}")
print(f"  (encoder + classifier are byte-identical between the two)")
""")


# =============================================================================
md(r"""## 7. Training routine (run once per aggregator)""")
code(r"""def train_one_model(aggregator_name: str, epochs: int = EPOCHS):
    print(f"\n{'='*72}\n  Training {aggregator_name.upper()} aggregator\n{'='*72}")
    torch.manual_seed(0); np.random.seed(0)
    model = build_model(aggregator_name).to(DEVICE)
    if USE_DP:
        model = nn.DataParallel(model, device_ids=list(range(N_GPUS)))
    def _unwrap(m): return m.module if isinstance(m, nn.DataParallel) else m

    opt = AdamW(_unwrap(model).parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    total_steps  = epochs * len(train_loader)
    warmup_steps = WARMUP_EPOCHS * len(train_loader)
    def lr_lambda(step):
        if step < warmup_steps: return step / max(warmup_steps, 1)
        p = (step - warmup_steps) / max(total_steps - warmup_steps, 1)
        return 0.5 * (1 + math.cos(math.pi * p))
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda)
    ce = nn.CrossEntropyLoss(label_smoothing=LABEL_SMOOTHING)

    ckpt_dir = os.path.join(CKPT_ROOT, aggregator_name)
    os.makedirs(ckpt_dir, exist_ok=True)
    history = {"train_loss": [], "train_acc": [], "test_acc": [], "best_test": 0.0}
    global_step = 0
    for ep in range(1, epochs + 1):
        model.train()
        t0 = time.time()
        ep_loss, ep_correct, ep_total = 0.0, 0, 0
        for batch in train_loader:
            pe  = batch["patch_events"].to(DEVICE)
            pc  = batch["patch_centroids"].to(DEVICE)
            y   = batch["label"].to(DEVICE)
            logits = model(pe, pc)
            loss = ce(logits, y)
            opt.zero_grad(set_to_none=True); loss.backward()
            torch.nn.utils.clip_grad_norm_(_unwrap(model).parameters(), 1.0)
            opt.step(); sched.step()
            ep_loss += loss.item() * y.size(0)
            ep_correct += (logits.argmax(-1) == y).sum().item()
            ep_total   += y.size(0)
            global_step += 1
        train_loss = ep_loss / ep_total
        train_acc  = ep_correct / ep_total

        # eval
        model.eval()
        te_correct = te_total = 0
        with torch.no_grad():
            for batch in test_loader:
                pe = batch["patch_events"].to(DEVICE)
                pc = batch["patch_centroids"].to(DEVICE)
                y  = batch["label"].to(DEVICE)
                logits = model(pe, pc)
                te_correct += (logits.argmax(-1) == y).sum().item()
                te_total   += y.size(0)
        test_acc = te_correct / te_total
        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["test_acc"].append(test_acc)
        improved = test_acc > history["best_test"]
        if improved:
            history["best_test"] = test_acc
            save_atomic({"model": _unwrap(model).state_dict(),
                         "epoch": ep, "test_acc": test_acc,
                         "history": history},
                         os.path.join(ckpt_dir, "best.pt"))
        print(f"  ep {ep:03d}/{epochs}  train_loss={train_loss:.4f}  "
              f"train_acc={train_acc:.4f}  test_acc={test_acc:.4f}  "
              f"lr={sched.get_last_lr()[0]:.1e}  {time.time()-t0:.1f}s"
              + ("  *" if improved else ""))
    print(f"\n  Final {aggregator_name}: best test acc = {history['best_test']:.4f}")
    return history
""")


# =============================================================================
md(r"""## 8. Train PointNet baseline""")
code(r"""hist_pointnet = train_one_model("pointnet", epochs=EPOCHS)
""")


# =============================================================================
md(r"""## 9. Train RoPE/HRR aggregator""")
code(r"""hist_rope = train_one_model("rope", epochs=EPOCHS)
""")


# =============================================================================
md(r"""## 10. Head-to-head comparison""")
code(r"""fig, axes = plt.subplots(1, 2, figsize=(14, 5))

ax = axes[0]
ax.plot(hist_pointnet["test_acc"], 'o-', color='C0', lw=2, label=f"PointNet (best={hist_pointnet['best_test']*100:.2f}%)")
ax.plot(hist_rope["test_acc"],     's-', color='C3', lw=2, label=f"RoPE/HRR (best={hist_rope['best_test']*100:.2f}%)")
ax.set_xlabel("epoch"); ax.set_ylabel("test accuracy")
ax.set_title("ModelNet40 test accuracy")
ax.legend(); ax.grid(alpha=0.3)

ax = axes[1]
ax.plot(hist_pointnet["train_loss"], 'o-', color='C0', lw=2, label="PointNet")
ax.plot(hist_rope["train_loss"],     's-', color='C3', lw=2, label="RoPE/HRR")
ax.set_xlabel("epoch"); ax.set_ylabel("train loss")
ax.set_title("Training loss")
ax.legend(); ax.grid(alpha=0.3)
plt.tight_layout(); plt.show()

# Summary table
print(f"\n{'='*48}")
print(f"  Final best test accuracy")
print(f"{'='*48}")
print(f"  PointNet aggregator:  {hist_pointnet['best_test']*100:.2f}%")
print(f"  RoPE/HRR aggregator:  {hist_rope['best_test']*100:.2f}%")
delta = (hist_rope['best_test'] - hist_pointnet['best_test']) * 100
print(f"  Δ (RoPE − PointNet):  {delta:+.2f} pts")
print(f"{'='*48}")
print()
print("Interpretation guide:")
print("  Δ > +1.5  →  RoPE clearly wins; the aggregator matters here.")
print("  -1.0 ≤ Δ ≤ +1.5 → effectively a tie within run-to-run noise.")
print("  Δ < -1.0  →  PointNet wins; max-pool's feature-selection has the right prior")
print("              for this regime, or RoPE's spectral prior is hurting somehow.")
""")


# =============================================================================
md(r"""## 11. Per-class accuracy (where does each method win?)""")
code(r"""def per_class_accuracy(aggregator_name):
    ckpt = torch.load(os.path.join(CKPT_ROOT, aggregator_name, "best.pt"),
                      map_location=DEVICE, weights_only=False)
    model = build_model(aggregator_name).to(DEVICE)
    model.load_state_dict(ckpt["model"]); model.eval()
    correct = np.zeros(40); total = np.zeros(40)
    with torch.no_grad():
        for batch in test_loader:
            pe = batch["patch_events"].to(DEVICE)
            pc = batch["patch_centroids"].to(DEVICE)
            y  = batch["label"].numpy()
            preds = model(pe, pc).argmax(-1).cpu().numpy()
            for yy, pp in zip(y, preds):
                total[yy] += 1
                if yy == pp: correct[yy] += 1
    return correct / np.maximum(total, 1)


acc_pn   = per_class_accuracy("pointnet")
acc_rope = per_class_accuracy("rope")
diff = acc_rope - acc_pn
order = np.argsort(diff)

fig, ax = plt.subplots(figsize=(14, 8))
xs = np.arange(40)
bar_w = 0.4
ax.barh(xs - bar_w/2, acc_pn[order]   * 100, bar_w, color='C0', label='PointNet')
ax.barh(xs + bar_w/2, acc_rope[order] * 100, bar_w, color='C3', label='RoPE/HRR')
ax.set_yticks(xs); ax.set_yticklabels([CLASS_NAMES[i] for i in order], fontsize=7)
ax.set_xlabel("per-class test accuracy (%)")
ax.set_title("Per-class accuracy — classes sorted by RoPE−PointNet difference\n"
             "(top = PointNet wins; bottom = RoPE wins)")
ax.legend(loc='lower right'); ax.grid(alpha=0.3, axis='x')
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
