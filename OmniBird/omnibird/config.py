"""Default hyperparameters for OmniBird — multimodal extension of PointBigBird-JEPA.

The single-modality (event-only) setup mirrors the PointBigBird v2 sparse-input
recipe, adapted for 3-D event coordinates (x, y, t). The multimodal setup adds
an ICMR-style cross-modal refinement on top (see icmr.py).
"""

from dataclasses import dataclass, field
from typing import Tuple


@dataclass
class OmniBirdConfig:
    # ── Event-camera coords (single-modality default) ──────────────────────
    coord_dim: int = 3            # events: (x, y, t)
    signal_dim: int = 1           # events: polarity (+/-1, encoded as scalar)
    side: int = 32                # grid resolution per axis for serialization (32^3 = 32768 cells)

    # ── Per-sample budget ──────────────────────────────────────────────────
    # n_events_total: cap events per sample.
    #   > 0  -> sample to exactly this size, pad with zeros if clip is shorter
    #   <= 0 -> use all real events; pad to n_events_max (key_padding_mask handles tail)
    n_events_total: int = 2048
    n_events_max:   int = 16384   # only used when n_events_total <= 0
    n_ctx: int = 1024             # context: ~50% of available events
    n_tgt_per_block: int = 64
    n_pred_blocks: int = 4
    n_tgt: int = 256              # = n_pred_blocks * n_tgt_per_block

    # ── Tokenizer ──────────────────────────────────────────────────────────
    fourier_dim: int = 96         # γ output = 2 * fourier_dim per coord_dim
    fourier_scale: float = 15.0

    # ── Encoder ────────────────────────────────────────────────────────────
    d_model: int = 256
    n_heads: int = 8
    dim_head: int = 32
    n_layers_enc: int = 6
    ffn_mult: int = 4

    # ── Predictor ──────────────────────────────────────────────────────────
    d_pred: int = 192
    n_heads_pred: int = 6
    dim_head_pred: int = 32
    n_layers_pred: int = 4
    predictor_pos_symmetric: bool = True

    # ── Sparse attention ───────────────────────────────────────────────────
    # attention_type:
    #   "bigbird" — BigBird block-sparse: window + globals + random key blocks
    #               per query block. Compute ~ O(N · K_attended · Dh).
    #   "grouped" — Dense self-attention WITHIN non-overlapping windows of
    #               `group_size`. Compute ~ O(N · group_size · Dh).
    #               Receptive field stays global through depth because each
    #               encoder layer picks a different curve, so group composition
    #               changes layer-to-layer.
    attention_type: str = "bigbird"
    block_size: int = 8           # used only when attention_type=="bigbird"
    window: int = 1
    n_random: int = 2
    n_global: int = 2
    group_size: int = 16          # used only when attention_type=="grouped"

    # ── Serialization ──────────────────────────────────────────────────────
    serial_orders: Tuple[str, ...] = ('z', 'z_rev', 'hilbert', 'hilbert_rev')
    reinject_pos: bool = False    # input-pos-only by default
    disjoint_targets: bool = True

    # Spatial separation margin (in normalized [-1, 1]^coord_dim units) between
    # context events and target events. After target blocks are picked, any
    # event within `context_target_margin` Euclidean distance of ANY target
    # event is forbidden from the context. Creates a true buffer zone that
    # forces the predictor to extrapolate across non-trivial spatial extent
    # rather than interpolate from immediately-adjacent context.
    #   0.0  -> no margin, only event-level disjointness (legacy default)
    #   0.03-0.05 -> typical for event clouds in [-1, 1]^3
    context_target_margin: float = 0.0

    # ── Patch-based mode (Point-MAE-aligned full) ──────────────────────────
    # When True, the dataset organizes events into fixed-size patches via
    # Hilbert-curve sort + reshape, and the encoder runs on patch tokens
    # produced by a mini-PointNet. Multi-block masking happens at the patch
    # level: 4 target blocks of `patches_per_block` patches each.
    patch_mode: bool = False
    patch_size: int = 32            # events per patch (K)
    patches_per_block: int = 16     # target patches per block
    ctx_max_patches: int = 192      # max context patches per sample
    patch_curve: str = "hilbert"    # which curve to sort by for patching

    # ── JEPA ───────────────────────────────────────────────────────────────
    loss_type: str = "cosine"     # canonical v2 default
    use_centering: bool = False   # canonical i-JEPA: no DINO centering
    ema_start: float = 0.999
    ema_end: float = 0.9999

    # ── Probe ──────────────────────────────────────────────────────────────
    probe_use_attn_pool: bool = True
    probe_interval: int = 20
    probe_epochs: int = 3
    probe_patience: int = 5

    # ── Optim ──────────────────────────────────────────────────────────────
    batch_size: int = 32
    epochs: int = 100
    lr: float = 2e-4
    weight_decay: float = 1e-4
    log_every: int = 50
    ckpt_dir: str = "./checkpoints_omnibird"

    # ── Multimodal (Phase 2 — see icmr.py) ─────────────────────────────────
    multimodal: bool = False                       # turn on cross-modal ICMR
    modalities: Tuple[str, ...] = ('events',)      # ('events', 'rgb') when multimodal
    icmr_iters: int = 2                            # ICMR refinement iterations
    icmr_n_latents: int = 256                      # shared latent set size
