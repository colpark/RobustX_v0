"""Build standalone/vit_fps_cifar10.ipynb — ViT-FPS JEPA on CIFAR-10 sparse pool."""
import json, os
NB = "/Users/davidpark/Documents/Claude/NFJEPA/OmniBird/standalone/vit_fps_cifar10.ipynb"

cells = []
def md(s):  cells.append({"cell_type":"markdown","metadata":{},"source":s.splitlines(keepends=True)})
def code(s): cells.append({"cell_type":"code","metadata":{},"execution_count":None,"outputs":[],"source":s.splitlines(keepends=True)})

md(r"""# ViT-FPS JEPA on CIFAR-10 (40% sparse pool)

Sparse-input ViT-style JEPA. The closest analog of canonical ViT-JEPA for inputs without a regular grid:

| Component | This recipe | ViT-JEPA on dense images |
|---|---|---|
| **Patch formation** | FPS on the sparse pool → K-NN groups around each centroid | Fixed 16×16 grid divisions |
| **Patch token** | mini-PointNet over (rel_coord, signal) of the K group members | Linear projection of 16×16×3 pixels |
| **Position embedding** | NeRF γ(centroid) → Linear → d_model | Learned per-patch-index lookup |
| **Encoder** | Dense ViT (standard self-attention over patches) | Dense ViT |
| **Masking** | Multi-block on the FPS centroid set; **context is a single CONTIGUOUS K-NN block placed as FAR AS POSSIBLE from all target blocks**, so the predictor has to *extrapolate* across the image rather than interpolate between surrounding context tokens. Same FPS pool for both encoders. | Multi-block on the patch grid; context block excludes target patches |
| **JEPA loss** | Smooth-L1 with DINO-style target centering + per-token LN | Smooth-L1 with per-token LN |

**The central design point:** FPS is run **once per sample** on the whole pool. Both context and target encoders work with the same fixed patch definitions; context and target are different *subsets* of the same FPS centroid set. This eliminates the "context and target see different patches" pitfall and is the cleanest analog of ViT-JEPA's "context/target are different subsets of the same patch grid".

**Patch content vs position is the load-bearing fix:**
- xattn's cross-attention readout uses centroid queries that have **only position**. The pool's content is retrieved by Q-K attention — extra learning subtask, prone to underutilization.
- This recipe's patch tokens have **both content and position**, exactly like ViT-JEPA patch tokens. mini-PointNet aggregates content; NeRF γ encodes position. The encoder operates on content-rich tokens throughout. Direct gather at target patch indices replaces the cross-attention readout.

**Caveat on positional encoding:** ViT uses a learnable per-index lookup, which we can't because patch indices don't correspond to fixed grid cells across samples (FPS output is data-conditional). We use NeRF γ + a trainable Linear instead — a continuous function over coordinates, which is a weaker prior than ViT's lookup table but the only option for sparse inputs.
""")

md("## 1. Setup")
code(r"""import os, sys, math, time, copy, ssl
sys.path.insert(0, os.path.abspath('.'))   # so vit_fps_core.py is importable
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
    ViTPatchEncoder, ViTFPSPredictor,
    NerfPosEnc,
    TargetCenter, ema_update, make_momentum_schedule,
    jepa_loss, short_params, save_atomic,
    farthest_point_sample, knn_indices,
    precompute_fps_knn_cached,
)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
torch.manual_seed(0); np.random.seed(0)
N_GPUS = min(4, torch.cuda.device_count())
USE_DP = (DEVICE == "cuda") and (N_GPUS > 1)
print(f"GPUs visible = {torch.cuda.device_count()}  using = {N_GPUS}  DataParallel = {USE_DP}")

# ── Config ────────────────────────────────────────────────────────────────
IMAGE_SIZE = 32
FRAC_POOL  = 0.4
K_POOL     = int(round(FRAC_POOL * IMAGE_SIZE * IMAGE_SIZE))   # 410

N_PATCHES  = 64                # FPS gives 64 centroids per sample
K_NEIGH    = 16                # K-NN per patch
N_TGT_BLOCKS = 4               # i-JEPA style multi-block
N_PATCH_PER_BLOCK = 4          # patches per target block
N_TGT      = N_TGT_BLOCKS * N_PATCH_PER_BLOCK                   # 16 target patches
# Context is a SINGLE CONTIGUOUS K-NN block, placed far from all target
# blocks (its anchor is the centroid maximizing distance to the nearest
# target patch). This forces the predictor to *extrapolate* across the
# image rather than interpolate between surrounding context tokens.
N_CTX      = 24                # contiguous context block

D_MODEL      = 256
N_LAYERS_ENC = 6
N_HEADS      = 8
DIM_HEAD     = 32
FFN_MULT     = 4
N_FREQS      = 10

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
CKPT_DIR       = "./checkpoints_vit_fps_cifar10"
os.makedirs(CKPT_DIR, exist_ok=True)
print(f"K_pool={K_POOL}  N_patches={N_PATCHES}  K_neigh={K_NEIGH}  N_tgt={N_TGT}  N_ctx={N_CTX}")
""")

md("""## 2. Dataset — FPS once per sample, then context/target are subsets

For each image:
1. Sample 40% of pixels (per-image fixed permutation).
2. **Run FPS over the pool** → 64 patch centroids.
3. **K-NN around each centroid** → each patch holds 16 pool pixels (overlap allowed).
4. **Multi-block target sampling on the FPS centroid set itself**:
   - Pick 4 anchor centroids randomly. Each anchor's `N_PATCH_PER_BLOCK` nearest centroids (in the centroid coord-space) form one target block. Successive blocks exclude previously-claimed centroids.
   - Remaining centroids = context.
5. Both context and target encoders consume the **same patch definitions**; they just see different *subsets* of the same fixed 64 patches.
""")
code(r"""class FPSPatchCIFAR10(Dataset):
    def __init__(self, base, train=True, pool_seed=0,
                 precompute_seed=42, cache_tag=None):
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

        # FPS + K-NN precompute is cached to disk. Key includes pool_seed (which
        # determines pool_idx), dataset size, patch params, and precompute_seed.
        # FPS+KNN result depends only on pool_idx (≡ pool_seed) and patch
        # params; train vs. eval differ only in __getitem__ masking, so they
        # share the cache. Disambiguation between CIFAR train/test splits is
        # via pool_seed (different per split in the call site).
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
        pool_coords = self.coords_all[pool_idx_np]     # (K_pool, 2)
        pool_signal = rgb[pool_idx_np]                  # (K_pool, 3)

        # CACHED FPS + KNN (precomputed at init)
        centroid_idx = torch.from_numpy(self.centroid_idx_all[idx])
        nbr_idx      = torch.from_numpy(self.nbr_idx_all[idx])
        centroid_coords = pool_coords[centroid_idx]
        patch_events = torch.cat([
            pool_coords[nbr_idx],
            pool_signal[nbr_idx],
        ], dim=-1)

        if self.train:
            rng = np.random.RandomState()
            cen_np = centroid_coords.numpy()
            n_p = N_PATCHES

            # ── Step 1: sample 4 target blocks (KNN around random anchors,
            #    disjoint between blocks). ─────────────────────────────────
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

            # ── Step 2: build a CONTIGUOUS context block FAR from targets.
            #   Context anchor = centroid maximizing min-distance to any
            #   target patch. Context block = K-NN around the anchor,
            #   excluding target patches. ─────────────────────────────────
            tgt_unique = np.unique(tgt_idx)
            if len(tgt_unique) == 0:
                ctx_anchor = rng.randint(n_p)
            else:
                # min distance from each centroid to nearest target
                d2_to_tgt = ((cen_np[:, None, :] - cen_np[tgt_unique][None, :, :]) ** 2).sum(-1)
                d2_to_tgt_min = d2_to_tgt.min(axis=1)
                d2_to_tgt_min[exclude] = -np.inf       # don't pick a target as ctx anchor
                ctx_anchor = int(np.argmax(d2_to_tgt_min))
            # Grow context: K nearest to ctx_anchor, excluding target patches
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
            # Test path: deterministic contiguous K-NN block around the
            # first FPS centroid. Same shape as the training context.
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

md("## 3. Visualize FPS patches + masking on one sample")
code(r"""classes = ['plane','car','bird','cat','deer','dog','frog','horse','ship','truck']
sample = train_ds[0]
print(f"label = {classes[sample['label']]}")

pe  = sample["patch_events"].numpy()       # (N_patches, K_neigh, 5)
pc  = sample["patch_centroids"].numpy()    # (N_patches, 2)
ctx = sample["ctx_idx"].numpy()
tgt = sample["tgt_idx"].numpy()

fig, axes = plt.subplots(1, 3, figsize=(15, 5))

# (a) FPS centroids on the pool
ax = axes[0]
pool_events = pe.reshape(-1, 5)
ax.scatter(pool_events[:, 1], -pool_events[:, 0], s=8, c='lightgray', alpha=0.4)
ax.scatter(pc[:, 1], -pc[:, 0], s=30, c='red', marker='x', label='FPS centroids')
ax.set_aspect('equal'); ax.set_title(f"(a) FPS gives {N_PATCHES} centroids"); ax.legend(fontsize=8)
ax.set_xlim(-1.05, 1.05); ax.set_ylim(-1.05, 1.05)

# (b) Each patch's K-NN coloured by patch id
ax = axes[1]
rng_cmap = np.random.RandomState(0)
patch_colors = plt.cm.tab20(rng_cmap.permutation(N_PATCHES) % 20)
for p in range(N_PATCHES):
    nbrs = pe[p, :, :2]
    ax.scatter(nbrs[:, 1], -nbrs[:, 0], s=10, c=[patch_colors[p]] * K_NEIGH)
ax.scatter(pc[:, 1], -pc[:, 0], s=15, c='black', marker='+')
ax.set_aspect('equal'); ax.set_title(f"(b) K-NN={K_NEIGH} around each centroid")
ax.set_xlim(-1.05, 1.05); ax.set_ylim(-1.05, 1.05)

# (c) Context (red) and target (4 colors) on the SAME centroid set
ax = axes[2]
ax.scatter(pool_events[:, 1], -pool_events[:, 0], s=4, c='lightgray', alpha=0.3)
ax.scatter(pc[ctx, 1], -pc[ctx, 0], s=40, c='#ef4444', label='context')
TGT_COLORS = ['#fbbf24', '#34d399', '#60a5fa', '#f472b6']
for k in range(N_TGT_BLOCKS):
    blk = tgt[k*N_PATCH_PER_BLOCK:(k+1)*N_PATCH_PER_BLOCK]
    ax.scatter(pc[blk, 1], -pc[blk, 0], s=80, c=TGT_COLORS[k], marker='s',
                label=f'tgt block {k+1}')
ax.set_aspect('equal'); ax.set_title("(c) ctx = red squares, 4 target blocks (colors)")
ax.legend(fontsize=8); ax.set_xlim(-1.05, 1.05); ax.set_ylim(-1.05, 1.05)
plt.tight_layout(); plt.show()
""")

md("## 4. Models — ViT encoder on patch tokens + dense predictor")
code(r"""context_encoder = ViTPatchEncoder(
    signal_dim=3, coord_dim=2, d_model=D_MODEL,
    n_layers=N_LAYERS_ENC, n_heads=N_HEADS, dim_head=DIM_HEAD,
    ffn_mult=FFN_MULT, n_freqs=N_FREQS,
).to(DEVICE)
target_encoder = copy.deepcopy(context_encoder).to(DEVICE)
for p in target_encoder.parameters(): p.requires_grad_(False)

predictor = ViTFPSPredictor(
    d_model=D_MODEL, d_pred=D_PRED,
    n_layers=N_LAYERS_PRED, n_heads=N_HEADS_PRED, dim_head=DIM_HEAD_PRED,
    coord_dim=2, n_freqs=N_FREQS, ffn_mult=FFN_MULT, pos_symmetric=True,
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

md("## 5. Optim + resume")
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

LAST = os.path.join(CKPT_DIR, "vit_fps_last.pt")
BEST = os.path.join(CKPT_DIR, "vit_fps_best.pt")
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

md("## 6. Probe — mean-pool over CONTEXT patch features (past-only at test)")
code(r"""class LinearProbe(nn.Module):
    def __init__(self, d, n=10):
        super().__init__()
        self.fc = nn.Linear(d, n)
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
        g = enc(sub_ev, sub_cen)                  # (B, Pc, D)
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

md("## 7. Training loop")
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

            # ── Target: encode ALL patches, gather at tgt indices ──
            with torch.no_grad():
                g_tgt_all = target_encoder(patch_events, patch_centroids)
                h_tgt_raw = torch.gather(g_tgt_all, 1,
                                          tgt_idx.unsqueeze(-1).expand(B, tgt_idx.size(1), D_MODEL))
                target_center.update(h_tgt_raw)
                h_tgt = F.layer_norm(target_center(h_tgt_raw), (D_MODEL,))

            # ── Context: encode ONLY context subset of patches ──
            Pc = ctx_idx.size(1)
            ctx_events    = torch.gather(patch_events,   1,
                                          ctx_idx[..., None, None].expand(B, Pc, K, F_dim))
            ctx_centroids = torch.gather(patch_centroids, 1,
                                          ctx_idx[..., None].expand(B, Pc, patch_centroids.size(-1)))
            g_ctx = context_encoder(ctx_events, ctx_centroids)

            # ── Predictor: ctx tokens + mask tokens at target centroids ──
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

md("## 8. Final long-form linear probe (30 epochs from best checkpoint)")
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
ax.set_title(f"ViT-FPS CIFAR-10 final linear probe — best test = {best_test*100:.2f}%")
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
