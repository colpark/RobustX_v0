"""PointBigBird-JEPA package."""

from .config import PBBConfig
from .serialization import (
    morton_code_2d, hilbert_code_2d,
    precompute_grid_orderings, subset_perm, invert_perm,
    precompute_subset_orderings,
)
from .attention import MultiHeadAttention, BigBirdSparseAttention, make_attention
from .model import (
    GaussianFourierFeatures, Tokenizer, FeedForward,
    EncoderBlock, PBBEncoder,
    PredictorBlock, PBBPredictor,
)
from .data import PBBChunkCIFAR10, build_loaders, orderings_from_batch, CIFAR_CLASSES
from .jepa import (
    ema_update, make_momentum_schedule, TargetCenter,
    gather_target_features, jepa_loss, diag_dict, fmt_diag,
)
from .utils import save_atomic, ensure_dir, count_params, short_params

__all__ = [
    "PBBConfig",
    # serialization
    "morton_code_2d", "hilbert_code_2d", "precompute_grid_orderings",
    "subset_perm", "invert_perm", "precompute_subset_orderings",
    # attention
    "MultiHeadAttention", "BigBirdSparseAttention", "make_attention",
    # model
    "GaussianFourierFeatures", "Tokenizer", "FeedForward",
    "EncoderBlock", "PBBEncoder", "PredictorBlock", "PBBPredictor",
    # data
    "PBBChunkCIFAR10", "build_loaders", "orderings_from_batch", "CIFAR_CLASSES",
    # jepa
    "ema_update", "make_momentum_schedule", "TargetCenter",
    "gather_target_features", "jepa_loss", "diag_dict", "fmt_diag",
    # utils
    "save_atomic", "ensure_dir", "count_params", "short_params",
]
