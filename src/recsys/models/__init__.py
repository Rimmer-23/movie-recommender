"""Model registry: build a model from a config or restore it from a checkpoint."""

from __future__ import annotations

from pathlib import Path

import torch

from recsys.config import ExperimentConfig, from_dict
from recsys.models.base import Recommender, TorchRecommender
from recsys.models.popular import PopularRecommender

MODEL_REGISTRY: dict[str, type[Recommender]] = {cls.name: cls for cls in (PopularRecommender,)}

__all__ = [
    "MODEL_REGISTRY",
    "PopularRecommender",
    "Recommender",
    "TorchRecommender",
    "build_model",
    "load_model",
]


def build_model(config: ExperimentConfig, n_users: int, n_items: int) -> Recommender:
    """Instantiate an untrained model described by ``config``."""
    if config.model not in MODEL_REGISTRY:
        raise ValueError(f"Unknown model '{config.model}'. Available: {sorted(MODEL_REGISTRY)}")
    cls = MODEL_REGISTRY[config.model]
    return cls(
        n_users=n_users,
        n_items=n_items,
        params=from_dict(cls.Params, config.params),
        train_config=config.train,
        seed=config.seed,
    )


def load_model(path: Path, device: str | None = None) -> Recommender:
    """Load any saved model; the class is resolved from the checkpoint's ``name``."""
    checkpoint = torch.load(Path(path), map_location="cpu", weights_only=True)
    cls = MODEL_REGISTRY[checkpoint["name"]]
    return cls.from_checkpoint(checkpoint, device=device)
