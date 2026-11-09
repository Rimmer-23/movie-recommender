"""Experiment configuration: YAML files in ``configs/`` loaded into dataclasses."""

from __future__ import annotations

from dataclasses import dataclass, field, fields
from pathlib import Path
from typing import Any, TypeVar

import yaml

T = TypeVar("T")

DEFAULT_DATA_DIR = Path("data") / "processed"
DEFAULT_ARTIFACTS_DIR = Path("artifacts")
DEFAULT_CONFIG_DIR = Path("configs")
MODEL_ORDER = ("popular", "mf", "two_tower", "sasrec")


def checkpoint_path(artifacts_dir: Path, model_name: str) -> Path:
    """Where a trained model is stored: ``artifacts/models/<name>.pt``."""
    return Path(artifacts_dir) / "models" / f"{model_name}.pt"


def results_path(artifacts_dir: Path) -> Path:
    """JSON with validation/test metrics of all models and the selected best model."""
    return Path(artifacts_dir) / "results.json"


@dataclass
class TrainConfig:
    """Optimisation and early-stopping settings shared by all trainable models.

    Attributes:
        epochs: Maximum number of passes over the training data.
        batch_size: Training batch size.
        lr: Adam learning rate.
        weight_decay: Adam weight decay (L2 on all parameters).
        grad_clip: Max global gradient norm, or ``None`` to disable clipping.
        eval_every: Run validation every N epochs.
        patience: Stop after this many validation rounds without improvement.
        monitor: Validation metric used for early stopping and model selection.
        eval_batch_size: Users per batch during full-ranking evaluation.
        device: Force a device (``"cpu"``, ``"cuda"``, ``"mps"``); auto-detected if ``None``.
    """

    epochs: int = 50
    batch_size: int = 1024
    lr: float = 1e-3
    weight_decay: float = 0.0
    grad_clip: float | None = None
    eval_every: int = 1
    patience: int = 5
    monitor: str = "NDCG@10"
    eval_batch_size: int = 1024
    device: str | None = None


@dataclass
class ExperimentConfig:
    """One training run: which model, its hyperparameters and how to train it."""

    model: str
    seed: int = 42
    params: dict[str, Any] = field(default_factory=dict)
    train: TrainConfig = field(default_factory=TrainConfig)


def from_dict(cls: type[T], values: dict[str, Any] | None) -> T:
    """Instantiate dataclass ``cls`` from a dict, rejecting unknown keys (catches typos)."""
    values = dict(values or {})
    known = {f.name for f in fields(cls)}  # type: ignore[arg-type]
    unknown = set(values) - known
    if unknown:
        raise ValueError(f"Unknown {cls.__name__} fields: {sorted(unknown)}")
    return cls(**values)


def load_config(path: Path) -> ExperimentConfig:
    """Load an experiment config from YAML.

    Expected layout::

        model: sasrec
        seed: 42
        params: {...}   # model-specific, validated by the model's Params dataclass
        train: {...}    # TrainConfig fields
    """
    raw = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
    if "model" not in raw:
        raise ValueError(f"{path}: missing required key 'model'")
    return ExperimentConfig(
        model=raw["model"],
        seed=int(raw.get("seed", 42)),
        params=raw.get("params") or {},
        train=from_dict(TrainConfig, raw.get("train")),
    )
