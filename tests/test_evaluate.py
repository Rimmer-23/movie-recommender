"""The shared full-ranking evaluation loop, checked with hand-crafted scorers."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pytest
import torch

from recsys.data import SplitData
from recsys.evaluate import evaluate, rank_items
from recsys.models.base import Recommender


class TargetScorer(Recommender):
    """Scores the current-stage target (and optionally the history) above everything else."""

    name = "target_scorer"
    display_name = "Target scorer"

    def __init__(self, data: SplitData, boost_history: bool, invert: bool = False) -> None:
        super().__init__(n_users=data.n_users, n_items=data.n_items)
        self.data, self.boost_history, self.invert = data, boost_history, invert

    def fit(self, data: SplitData, log_dir: Path | None = None) -> TargetScorer:
        return self

    def score(self, user_ids: np.ndarray) -> torch.Tensor:
        scores = torch.zeros(len(user_ids), self.n_items + 1)
        for row, user in enumerate(user_ids):
            history = self.history[user]
            # The next item after the given history is the target for this stage.
            target = self.data.sequences[user][len(history)]
            scores[row, target] = -10.0 if self.invert else 10.0
            if self.boost_history:
                scores[row, history] = 100.0
        return scores

    def get_state(self) -> dict[str, Any]:
        return {}

    def set_state(self, state: dict[str, Any]) -> None:
        pass


@pytest.mark.parametrize("stage", ["valid", "test"])
def test_perfect_model_scores_one(split: SplitData, stage: str) -> None:
    metrics = evaluate(TargetScorer(split, boost_history=False), split, stage)
    assert all(v == pytest.approx(1.0) for v in metrics.values())


@pytest.mark.parametrize("stage", ["valid", "test"])
def test_seen_items_are_excluded(split: SplitData, stage: str) -> None:
    # History items get the highest scores; if they were not excluded the target would
    # drop out of the top ranks for users with long histories.
    metrics = evaluate(TargetScorer(split, boost_history=True), split, stage)
    assert metrics["Recall@10"] == pytest.approx(1.0)
    assert metrics["NDCG@10"] == pytest.approx(1.0)


def test_worst_model_scores_zero(split: SplitData) -> None:
    metrics = evaluate(TargetScorer(split, boost_history=False, invert=True), split, "test")
    assert metrics["Recall@10"] == metrics["NDCG@10"] == 0.0


def test_rank_items_batches_consistently(split: SplitData) -> None:
    model = TargetScorer(split, boost_history=False)
    model.set_history(split.history("valid"))
    users = np.arange(split.n_users)
    full = rank_items(model, users, k=5, batch_size=1000)
    batched = rank_items(model, users, k=5, batch_size=7)
    np.testing.assert_array_equal(full, batched)
    assert full.shape == (split.n_users, 5)
    for user, row in zip(users, full, strict=True):
        assert not set(row) & set(split.history("valid")[user])
        assert 0 not in row
