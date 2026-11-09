"""Model interface: output shapes, seen-item exclusion and save/load round trips."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from recsys.config import ExperimentConfig
from recsys.data import SplitData
from recsys.models import MODEL_REGISTRY, PopularRecommender, build_model, load_model


def test_popular_ranks_by_train_count(split: SplitData) -> None:
    model = PopularRecommender(split.n_users, split.n_items).fit(split)
    counts = np.bincount(np.concatenate(split.history("valid")), minlength=split.n_items + 1)

    scores = model.score(np.array([0, 1, 2]))
    assert scores.shape == (3, split.n_items + 1)

    items, item_scores = model.recommend([0], k=5, exclude_seen=False)
    assert items.shape == item_scores.shape == (1, 5)
    expected = np.argsort(-counts[1:], kind="stable")[:5] + 1
    np.testing.assert_array_equal(np.sort(counts[items[0]]), np.sort(counts[expected]))
    assert np.all(np.diff(item_scores[0]) <= 0)


def test_popular_excludes_seen_items(split: SplitData) -> None:
    model = PopularRecommender(split.n_users, split.n_items).fit(split)
    users = np.arange(split.n_users)
    items, _ = model.recommend(users, k=5, exclude_seen=True)
    for user, row in zip(users, items, strict=True):
        assert not set(row) & set(split.history("valid")[user])
        assert 0 not in row


def test_recommend_requires_history(split: SplitData) -> None:
    model = PopularRecommender(split.n_users, split.n_items)
    with pytest.raises(RuntimeError):
        model.recommend([0], k=3, exclude_seen=True)
    items, _ = model.recommend([0], k=3, exclude_seen=False)  # fine without history
    assert items.shape == (1, 3)


def test_k_is_capped_at_catalogue_size(split: SplitData) -> None:
    model = PopularRecommender(split.n_users, split.n_items).fit(split)
    items, _ = model.recommend([0], k=10_000, exclude_seen=False)
    assert items.shape == (1, split.n_items)


@pytest.mark.parametrize("name", ["popular"])
def test_save_load_roundtrip(split: SplitData, tmp_path: Path, name: str) -> None:
    model = build_model(ExperimentConfig(model=name), split.n_users, split.n_items).fit(split)
    path = tmp_path / f"{name}.pt"
    model.save(path)
    restored = load_model(path)
    assert type(restored) is MODEL_REGISTRY[name]
    restored.set_history(split.history("valid"))
    users = np.arange(min(8, split.n_users))
    np.testing.assert_array_equal(model.recommend(users, k=5)[0], restored.recommend(users, k=5)[0])


def test_unknown_model_name() -> None:
    with pytest.raises(ValueError):
        build_model(ExperimentConfig(model="nope"), 10, 10)
