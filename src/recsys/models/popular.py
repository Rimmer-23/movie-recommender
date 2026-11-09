"""Most Popular baseline: rank items by their number of train interactions."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import torch

from recsys.data import SplitData
from recsys.models.base import Recommender


class PopularRecommender(Recommender):
    """Non-personalised baseline: every user gets the globally most frequent items.

    Also used by the API as the fallback for unknown users.
    """

    name = "popular"
    display_name = "Most Popular"

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.counts = torch.zeros(self.n_items + 1)  # (n_items + 1,)

    def fit(self, data: SplitData, log_dir: Path | None = None) -> PopularRecommender:
        train_history = data.history("valid")
        self.fit_items(np.concatenate(train_history))
        self.set_history(train_history)
        return self

    def fit_items(self, items: np.ndarray) -> PopularRecommender:
        """Count interactions from a flat array of item ids."""
        counts = np.bincount(items, minlength=self.n_items + 1).astype(np.float32)
        self.counts = torch.from_numpy(counts)
        return self

    def score(self, user_ids: np.ndarray) -> torch.Tensor:
        return self.counts.expand(len(user_ids), -1).clone()  # (batch, n_items + 1)

    def get_state(self) -> dict[str, Any]:
        return {"counts": self.counts}

    def set_state(self, state: dict[str, Any]) -> None:
        self.counts = state["counts"].float()
