# CIFAR-10 JEPA baselines — two self-contained implementations

Two JEPA recipes on the **40% sparse pixel pool** of CIFAR-10, packaged
for sharing. The two baselines differ in **how the encoder reads from the
sparse input** — direct per-pixel tokens vs. cross-attention pool from
centroid queries — and represent the two ends of the design space we
explored before settling on the recipe used in the main repo.

## The shared setup

- CIFAR-10, 32×32 RGB. Each sample is reduced to **40% of its pixels**
  (410 of 1024) via a per-image fixed permutation. Sparsity is artificial
  but the recipe is intended for genuinely sparse inputs (event cameras,
  point clouds); CIFAR-10 is the easiest sanity test.
- Per-pixel content is 24 bits of RGB (high per-event entropy — the
  regime where JEPA's stop-grad + EMA + target-centering recipe is
  stable).
- Self-supervised pretraining via JEPA: predict EMA-target features at
  target positions from features at context positions.
- Evaluation: linear probe on the frozen context-encoder features.

## The two methods

### `pbb_v2/` — per-pixel BigBird (the strong baseline)

The PointBigBird recipe. **Per-pixel tokens** through a 6-layer BigBird
sparse-attention encoder, with target features gathered directly at the
target-pixel positions (no aggregation step). DINO-style target
centering before per-token LayerNorm is the load-bearing anti-collapse
fix.

- 410 per-pixel tokens per sample
- 100 context tokens, 4×50 = 200 target tokens
- Smooth-L1 loss in feature space
- See `pbb_v2/README.md` for the full architecture.

### `xattn/` — cross-attention readout (the alternative)

Encoder consumes per-event tokens, then a `LocalCrossAttention` pool
reads out at FPS-derived centroid queries. The predictor's job is then
"pool-at-target ← pool-at-context" — same statistic at different
locations. Includes a tokenizer skip-connection at the pool input to
guarantee per-event variation even if the encoder partially collapses.

- BigBird encoder over 400 per-pixel events
- Cross-attention pool at FPS centroids
- Symmetric pool-at-context / pool-at-target predictor
- Same DINO-style target centering
- See `xattn/README.md` for the full architecture.

## Which one to use as the baseline

| | pbb_v2 | xattn |
|---|---|---|
| Per-token entropy | high (24 bits, per-pixel) | moderate (post-pool) |
| Token count seen by encoder | 100 context tokens | 100 context tokens |
| Information loss in aggregation | **none** (direct gather) | some (cross-attn readout) |
| Empirical linear-probe accuracy on CIFAR-10 | **stronger baseline** | weaker than pbb_v2 |
| Generalization to event cameras | works when token entropy is high enough | meant for sparser modalities |

**Default recommendation: start with `pbb_v2/`.** It's the cleaner
baseline (no aggregation pitfalls) and is what the main repo's
downstream notebooks build on. The `xattn/` variant is included because
the cross-attention pool design is what scales to genuinely sparse
modalities (event cameras, point clouds, large-N point sets); on
CIFAR-10 specifically it loses a few points to pbb_v2's per-pixel
direct-gather approach.

## Running

Each subfolder is **independent** and **copy-paste portable**. Drop
either folder anywhere with PyTorch + torchvision installed and open
its notebook:

```bash
cd pbb_v2 && jupyter notebook pbb_cifar10.ipynb
# or
cd xattn  && jupyter notebook xattn_cifar10.ipynb
```

Both notebooks resume training from a local `./checkpoints_*/` directory;
delete the directory or set `RESUME = False` in cell 5 to start fresh.

## Layout

```
CIFAR10_JEPA_baselines/
├── README.md                      ← this file
├── pbb_v2/                        ← self-contained PBB v2
│   ├── README.md
│   ├── pbb_core.py                ← Tokenizer, BigBird, predictor, target centering
│   └── pbb_cifar10.ipynb
└── xattn/                         ← cross-attention readout baseline
    ├── README.md
    ├── omnibird/                  ← package the notebook imports from
    │   ├── __init__.py
    │   ├── attention.py
    │   ├── config.py
    │   ├── data.py
    │   ├── jepa.py
    │   ├── model.py
    │   ├── probe.py
    │   ├── serialization.py
    │   └── utils.py
    └── xattn_cifar10.ipynb
```
