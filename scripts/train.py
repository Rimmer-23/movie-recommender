"""Train a model (or all of them) with validation-based early stopping.

Usage:
    python scripts/train.py --model sasrec
    python scripts/train.py --model all
    python scripts/train.py --model mf --config configs/mf.yaml --epochs 3

Checkpoints go to ``artifacts/models/<model>.pt`` and TensorBoard logs to
``artifacts/runs/<model>/`` (``tensorboard --logdir artifacts/runs``).
"""

from __future__ import annotations

import argparse
import json
import logging
import shutil
import time
from pathlib import Path

from recsys.config import (
    DEFAULT_ARTIFACTS_DIR,
    DEFAULT_CONFIG_DIR,
    DEFAULT_DATA_DIR,
    MODEL_ORDER,
    checkpoint_path,
    load_config,
)
from recsys.data import SplitData, load_dataset
from recsys.evaluate import evaluate
from recsys.models import build_model
from recsys.utils import set_seed

logger = logging.getLogger("train")


def train_one(
    model_name: str,
    data: SplitData,
    config_path: Path,
    artifacts_dir: Path,
    epochs: int | None = None,
    device: str | None = None,
) -> dict[str, float]:
    """Train ``model_name`` from its config, save the checkpoint, return valid metrics."""
    config = load_config(config_path)
    if config.model != model_name:
        raise ValueError(f"{config_path} configures '{config.model}', not '{model_name}'")
    if epochs is not None:
        config.train.epochs = epochs
    if device is not None:
        config.train.device = device

    set_seed(config.seed)
    model = build_model(config, n_users=data.n_users, n_items=data.n_items)
    log_dir = artifacts_dir / "runs" / model_name
    shutil.rmtree(log_dir, ignore_errors=True)  # one run per model keeps curves comparable

    logger.info("Training %s on %s (config: %s)", model_name, model.device, config_path)
    start = time.perf_counter()
    model.fit(data, log_dir=log_dir)
    elapsed = time.perf_counter() - start

    path = checkpoint_path(artifacts_dir, model_name)
    model.save(path)
    metrics = evaluate(model, data, "valid")
    logger.info("Saved %s (%.0fs). Valid: %s", path, elapsed, json.dumps(_round(metrics)))
    return metrics


def _round(metrics: dict[str, float]) -> dict[str, float]:
    return {k: round(v, 4) for k, v in metrics.items()}


def main() -> None:
    parser = argparse.ArgumentParser(description="Train recommender models.")
    parser.add_argument("--model", required=True, choices=[*MODEL_ORDER, "all"])
    parser.add_argument("--config", type=Path, help="defaults to configs/<model>.yaml")
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--artifacts-dir", type=Path, default=DEFAULT_ARTIFACTS_DIR)
    parser.add_argument("--epochs", type=int, help="override the configured max epochs")
    parser.add_argument("--device", help="force a device, e.g. cpu / cuda / mps")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s | %(message)s")
    if args.model == "all" and args.config is not None:
        parser.error("--config cannot be combined with --model all")

    data = load_dataset(args.data_dir)
    names = MODEL_ORDER if args.model == "all" else (args.model,)
    for name in names:
        config_path = args.config or DEFAULT_CONFIG_DIR / f"{name}.yaml"
        train_one(name, data, config_path, args.artifacts_dir, args.epochs, args.device)


if __name__ == "__main__":
    main()
