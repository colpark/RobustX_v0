"""OmniBird — multimodal extension of PointBigBird-JEPA for event cameras."""

from .config import OmniBirdConfig
from .serialization import (
    morton_code_2d, hilbert_code_2d, morton_code_3d, hilbert_code_3d,
    precompute_grid_orderings, subset_perm, invert_perm, quantize_coords,
)
from .attention import MultiHeadAttention, BigBirdSparseAttention, make_attention
from .model import (
    GaussianFourierFeatures, Tokenizer, FeedForward,
    EncoderBlock, OmniBirdEncoder,
    PredictorBlock, OmniBirdPredictor,
)
from .data import OmniBirdEventDataset, build_loaders, orderings_from_batch
from .jepa import (
    ema_update, make_momentum_schedule, TargetCenter,
    gather_target_features, jepa_loss, diag_dict, fmt_diag,
)
from .probe import LinearProbe, AttnPoolHead, extract_z, quick_probe
from .icmr import ICMR, ModalityCrossAttn
from .utils import save_atomic, ensure_dir, count_params, short_params

__all__ = [
    "OmniBirdConfig",
    # serialization
    "morton_code_2d", "hilbert_code_2d", "morton_code_3d", "hilbert_code_3d",
    "precompute_grid_orderings", "subset_perm", "invert_perm", "quantize_coords",
    # attention
    "MultiHeadAttention", "BigBirdSparseAttention", "make_attention",
    # model
    "GaussianFourierFeatures", "Tokenizer", "FeedForward",
    "EncoderBlock", "OmniBirdEncoder", "PredictorBlock", "OmniBirdPredictor",
    # data
    "OmniBirdEventDataset", "build_loaders", "orderings_from_batch",
    # jepa
    "ema_update", "make_momentum_schedule", "TargetCenter",
    "gather_target_features", "jepa_loss", "diag_dict", "fmt_diag",
    # probe
    "LinearProbe", "AttnPoolHead", "extract_z", "quick_probe",
    # multimodal (Phase 2)
    "ICMR", "ModalityCrossAttn",
    # utils
    "save_atomic", "ensure_dir", "count_params", "short_params",
]
