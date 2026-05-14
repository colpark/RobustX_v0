"""Default hyperparameters for PointBigBird-JEPA on CIFAR-10."""

from dataclasses import dataclass, field
from typing import Tuple


@dataclass
class PBBConfig:
    # Image / data
    image_size: int = 32
    n_pix: int = 1024
    pool_frac: float = 0.40
    k_pool: int = 410           # = round(pool_frac * n_pix)
    k_half: int = 205           # = k_pool // 2; test-time fixed context size
    pool_seed: int = 12345

    # v8 multi-block masking (i-JEPA recipe)
    k_ctx: int = 100            # train context block size (pool pixels)
    k_tgt: int = 50             # per-target-block size      (pool pixels)
    n_pred: int = 4             # number of target blocks per image
    n_tgt: int = 200            # = n_pred * k_tgt

    # Tokenization
    rgb_channels: int = 3
    fourier_dim: int = 96       # γ output is 2*fourier_dim
    pos_dim: int = 192          # = 2 * fourier_dim
    fourier_scale: float = 15.0

    # Encoder
    d_model: int = 256
    n_heads: int = 8
    dim_head: int = 32          # → inner = 256
    n_layers_enc: int = 6
    ffn_mult: int = 4

    # Predictor
    d_pred: int = 192
    n_heads_pred: int = 6
    dim_head_pred: int = 32
    n_layers_pred: int = 4

    # BigBird sparse attention (block-sparse, fixed pattern)
    # Each query block attends to: (2*window + 1) window blocks + n_global + n_random
    block_size: int = 32
    window: int = 1
    n_random: int = 2
    n_global: int = 2           # blocks 0 and (num_blocks - 1) are globals

    # Serialization — 4 orderings, one sampled per encoder layer
    serial_orders: Tuple[str, ...] = ('z', 'z_rev', 'hilbert', 'hilbert_rev')

    # Re-inject positional embedding as a residual at every encoder layer.
    # Required for per-layer reshuffling to actually motivate the model to
    # use *true* spatial position (otherwise it can latch onto current-layer
    # sequence position, which changes with the curve).
    reinject_pos: bool = True

    # v2 improvements (default OFF for back-compat — toggled on in the v2 notebook):
    # - disjoint_targets:           target blocks disjoint from each other (not just from ctx)
    # - predictor_pos_symmetric:    predictor adds γ(ctx_coords) to ctx tokens too
    # - probe_use_attn_pool:        probe pools via a learnable AttnPoolHead instead of mean
    disjoint_targets: bool = False
    predictor_pos_symmetric: bool = False
    probe_use_attn_pool: bool = False

    # Test-time input size:
    #   test_full_pool=False (default): test loader serves K_HALF=205 pool pixels (legacy 20% contract)
    #   test_full_pool=True           : test loader serves all K_POOL=410 pool pixels (full 40% budget)
    # When True, the probe / test inputs match the target encoder's training distribution exactly
    # (the target encoder always saw the full pool during JEPA training).
    test_full_pool: bool = False

    # JEPA
    ema_start: float = 0.999
    # ema_end: 1.000 freezes the target encoder by end of training (legacy).
    # 0.9999 keeps it slowly moving forever → avoids late-training "predictor
    # saturation" failure mode where features drift after EMA freezes.
    ema_end: float = 0.9999
    center_momentum: float = 0.9
    # JEPA distance metric. "smooth_l1" (legacy) penalizes magnitude differences;
    # "cosine" uses 1 - cos(h_pred, h_tgt), which is direction-only and tends to
    # be more robust over long training (no magnitude pull on h_pred toward h_tgt).
    loss_type: str = "smooth_l1"

    # DINO-style centering of target features. NOT part of canonical i-JEPA —
    # we used to inherit it from the v1–v8 OmniField-JEPA setup. False matches
    # the i-JEPA spec (loss = smooth_L1(h_pred, LayerNorm(h_tgt))) and avoids
    # the late-training "center drift" failure mode.
    use_centering: bool = False

    # Optim
    batch_size: int = 64
    epochs: int = 100
    lr: float = 2e-4
    weight_decay: float = 1e-4
    probe_interval: int = 20
    probe_epochs: int = 3
    # Early-stop training if the embedded probe accuracy does not improve for
    # this many consecutive probe checks. 0 disables (legacy behavior).
    probe_patience: int = 0

    # Logging
    log_every: int = 50
    ckpt_dir: str = "./checkpoints_pbb"
