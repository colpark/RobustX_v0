"""Dataset adapters for OmniBird (event-camera + multimodal)."""

from .synthetic  import SyntheticEventDataset, build_synthetic_loaders
from .eventscape import EventScapeDataset

__all__ = ["SyntheticEventDataset", "build_synthetic_loaders", "EventScapeDataset"]
