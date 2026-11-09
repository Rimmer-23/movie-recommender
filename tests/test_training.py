"""Tiny end-to-end training runs of every trainable model on the synthetic dataset."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pytest
import torch

from recsys.config import ExperimentConfig, TrainConfig
from recsys.data import SplitData
from recsys.evaluate import evaluate
from recsys.models import TorchRecommender, build_model, load_model

TINY_PARAMS: dict[str, dict[str, Any]] = {
    "mf": {"embedding_dim": 8},
    "two_tower": {"id_dim": 8, "feature_dim": 4, "hidden_dims": [16], "output_dim": 8},
    "sasrec": {"hidden_dim": 16, "max_len": 10, "n_heads": 2, "dropout": 0.1},
}
TRAINABLE = list(TINY_PARAMS)


def tiny_model(name: str, split: SplitData, epochs: int = 2) -> TorchRecommender:
    config = ExperimentConfig(
        model=name,
        seed=0,
        params=TINY_PARAMS[name],
        train=TrainConfig(epochs=epochs, batch_size=32, lr=1e-2, patience=10, device="cpu"),
    )
    model = build_model(config, split.n_users, split.n_items)
    assert isinstance(model, TorchRecommender)
    return model


@pytest.mark.parametrize("name", TRAINABLE)
def test_fit_recommend_save_load(name: str, split: SplitData, tmp_path: Path) -> None:
    model = tiny_model(name, split).fit(split)

    assert len(model.training_log) == 2
    assert all(np.isfinite(r["loss"]) for r in model.training_log)
    assert "valid_NDCG@10" in model.training_log[-1]

    users = np.arange(split.n_users)
    scores = model.score(users[:5])
    assert scores.shape == (5, split.n_items + 1)
    items, item_scores = model.recommend(users, k=5)
    assert items.shape == item_scores.shape == (split.n_users, 5)
    for user, row in zip(users, items, strict=True):
        assert not set(row) & set(split.history("valid")[user])

    metrics = evaluate(model, split, "test")
    assert all(0.0 <= v <= 1.0 for v in metrics.values())

    path = tmp_path / f"{name}.pt"
    model.save(path)
    restored = load_model(path, device="cpu")
    for m in (model, restored):
        m.set_history(split.history("full"))
    torch.testing.assert_close(model.score(users[:8]), restored.score(users[:8]))
    np.testing.assert_array_equal(model.recommend(users, k=5)[0], restored.recommend(users, k=5)[0])


@pytest.mark.parametrize("name", TRAINABLE)
def test_training_reduces_loss(name: str, split: SplitData) -> None:
    model = tiny_model(name, split, epochs=8).fit(split)
    losses = [r["loss"] for r in model.training_log]
    assert losses[-1] < losses[0]


@pytest.mark.parametrize("name", TRAINABLE)
def test_training_is_reproducible(name: str, split: SplitData) -> None:
    first = tiny_model(name, split).fit(split)
    second = tiny_model(name, split).fit(split)
    assert [r["loss"] for r in first.training_log] == [r["loss"] for r in second.training_log]


def test_early_stopping_restores_best_epoch(split: SplitData) -> None:
    model = tiny_model("mf", split, epochs=30)
    model.train_config.patience = 2
    model.fit(split)
    log = model.training_log
    best = max(log, key=lambda r: r["valid_NDCG@10"])
    assert model.best_epoch == best["epoch"]
    assert len(log) <= 30
    # The restored weights reproduce the best validation score.
    assert evaluate(model, split, "valid")["NDCG@10"] == pytest.approx(best["valid_NDCG@10"])
