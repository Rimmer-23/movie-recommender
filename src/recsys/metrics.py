"""Top-k ranking metrics: Recall@k and NDCG@k.

All functions take

* ``ranked`` — integer array ``(n_users, K)`` with recommended item ids, best first;
* ``targets`` — relevant items per user, either ``(n_users,)`` for a single held-out item
  (leave-one-out) or ``(n_users, T)`` padded with negative values for variable-length sets.

With a single relevant item, Recall@k equals HitRate@k and NDCG@k is ``1 / log2(rank + 1)``
when the item is ranked within the top k (1-based rank), else 0.
"""

from __future__ import annotations

from collections.abc import Iterable

import numpy as np

DEFAULT_KS: tuple[int, ...] = (10, 20)


def _as_2d(targets: np.ndarray) -> np.ndarray:
    targets = np.asarray(targets)
    return targets[:, None] if targets.ndim == 1 else targets


def _hits(ranked: np.ndarray, targets: np.ndarray, k: int) -> np.ndarray:
    """Boolean matrix ``(n_users, k)``: is the item at each position relevant?"""
    ranked = np.asarray(ranked)[:, :k]
    targets = _as_2d(targets)
    if ranked.shape[0] != targets.shape[0]:
        raise ValueError(f"ranked has {ranked.shape[0]} rows but targets has {targets.shape[0]}")
    valid = targets >= 0  # negative ids are padding
    return ((ranked[:, :, None] == targets[:, None, :]) & valid[:, None, :]).any(axis=-1)


def _n_relevant(targets: np.ndarray) -> np.ndarray:
    return (_as_2d(targets) >= 0).sum(axis=1)


def recall_at_k(ranked: np.ndarray, targets: np.ndarray, k: int) -> float:
    """Mean over users of ``|top-k ∩ relevant| / |relevant|``.

    Users without relevant items are ignored.
    """
    hits = _hits(ranked, targets, k).sum(axis=1)
    n_rel = _n_relevant(targets)
    mask = n_rel > 0
    if not mask.any():
        return 0.0
    return float(np.mean(hits[mask] / n_rel[mask]))


def ndcg_at_k(ranked: np.ndarray, targets: np.ndarray, k: int) -> float:
    """Mean over users of binary-relevance NDCG@k.

    ``DCG = Σ_{i<k} rel_i / log2(i + 2)`` and IDCG places ``min(|relevant|, k)`` hits on top.
    Users without relevant items are ignored.
    """
    hits = _hits(ranked, targets, k)
    discounts = 1.0 / np.log2(np.arange(2, hits.shape[1] + 2))  # (k,)
    dcg = (hits * discounts).sum(axis=1)
    n_rel = np.minimum(_n_relevant(targets), hits.shape[1])
    idcg = np.cumsum(discounts)[np.maximum(n_rel - 1, 0)] * (n_rel > 0)
    mask = n_rel > 0
    if not mask.any():
        return 0.0
    return float(np.mean(dcg[mask] / idcg[mask]))


def compute_metrics(
    ranked: np.ndarray, targets: np.ndarray, ks: Iterable[int] = DEFAULT_KS
) -> dict[str, float]:
    """Recall@k and NDCG@k for every ``k`` in ``ks``, e.g. ``{"Recall@10": ..., ...}``."""
    results: dict[str, float] = {}
    for k in ks:
        if k > np.asarray(ranked).shape[1]:
            raise ValueError(f"k={k} exceeds the number of ranked items")
        results[f"Recall@{k}"] = recall_at_k(ranked, targets, k)
        results[f"NDCG@{k}"] = ndcg_at_k(ranked, targets, k)
    return results
