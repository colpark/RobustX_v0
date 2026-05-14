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

    # JEPA
    ema_start: float = 0.999
    ema_end: float = 1.000
    center_momentum: float = 0.9

    # Optim
    batch_size: int = 64
    epochs: int = 100
    lr: float = 2e-4
    weight_decay: float = 1e-4
    probe_interval: int = 20
    probe_epochs: int = 3

    # Logging
    log_every: int = 50
    ckpt_dir: str = "./checkpoints_pbb"
