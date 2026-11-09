"""Small helpers shared across the package: seeding and device selection."""

from __future__ import annotations

import random

import numpy as np
import torch


def set_seed(seed: int) -> None:
    """Seed Python, NumPy and PyTorch RNGs for reproducible runs."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_device(preferred: str | None = None) -> torch.device:
    """Pick the best available device: CUDA → MPS → CPU.

    Args:
        preferred: Optional explicit device string (e.g. ``"cpu"``) that overrides auto-detection.
    """
    if preferred:
        return torch.device(preferred)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")
