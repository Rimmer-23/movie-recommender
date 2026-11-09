"""Full-ranking evaluation shared by every model.

For each user the model scores *all* items, items already in the user's history are
removed, and the held-out item's position in the remaining ranking determines the metrics.
No negative sampling is involved.
"""

from __future__ import annotations

from collections.abc import Iterable
from typing import TYPE_CHECKING, Literal

import numpy as np

from recsys.data import SplitData
from recsys.metrics import DEFAULT_KS, compute_metrics

if TYPE_CHECKING:
    from recsys.models.base import Recommender


def rank_items(
    model: Recommender, user_ids: np.ndarray, k: int, batch_size: int = 1024
) -> np.ndarray:
    """Top-``k`` items (seen items excluded) for ``user_ids``, shape ``(n_users, k)``."""
    batches = [
        model.recommend(user_ids[start : start + batch_size], k=k, exclude_seen=True)[0]
        for start in range(0, len(user_ids), batch_size)
    ]
    return np.concatenate(batches, axis=0)


def evaluate(
    model: Recommender,
    data: SplitData,
    stage: Literal["valid", "test"],
    ks: Iterable[int] = DEFAULT_KS,
    batch_size: int = 1024,
) -> dict[str, float]:
    """Leave-one-out full-ranking metrics on the validation or test target.

    The model's history is set to what is known before the target: train items for
    ``valid``, train + validation item for ``test``.
    """
    ks = tuple(ks)
    model.set_history(data.history(stage))
    users = np.arange(data.n_users)
    ranked = rank_items(model, users, k=max(ks), batch_size=batch_size)
    return compute_metrics(ranked, data.targets(stage), ks)
