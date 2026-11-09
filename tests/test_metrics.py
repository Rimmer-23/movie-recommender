"""Correctness of Recall@k / NDCG@k against hand-computed values."""

from __future__ import annotations

import math

import numpy as np
import pytest

from recsys.metrics import compute_metrics, ndcg_at_k, recall_at_k

RANKED = np.array(
    [
        [5, 3, 9, 1],  # target 5 at rank 1
        [2, 4, 6, 8],  # target 6 at rank 3
        [7, 1, 2, 3],  # target 9 missing
    ]
)
TARGETS = np.array([5, 6, 9])


def test_recall_single_target() -> None:
    assert recall_at_k(RANKED, TARGETS, k=1) == pytest.approx(1 / 3)
    assert recall_at_k(RANKED, TARGETS, k=2) == pytest.approx(1 / 3)
    assert recall_at_k(RANKED, TARGETS, k=3) == pytest.approx(2 / 3)
    assert recall_at_k(RANKED, TARGETS, k=4) == pytest.approx(2 / 3)


def test_ndcg_single_target() -> None:
    # rank 1 → 1/log2(2) = 1, rank 3 → 1/log2(4) = 0.5, miss → 0
    assert ndcg_at_k(RANKED, TARGETS, k=1) == pytest.approx(1 / 3)
    assert ndcg_at_k(RANKED, TARGETS, k=3) == pytest.approx((1 + 0.5 + 0) / 3)
    assert ndcg_at_k(RANKED, TARGETS, k=2) == pytest.approx(1 / 3)


def test_multiple_targets_with_padding() -> None:
    ranked = np.array([[1, 2, 3, 4], [4, 3, 2, 1]])
    targets = np.array([[2, 4, -1], [9, -1, -1]])  # user 0: {2, 4}, user 1: {9}
    # user 0: hits at ranks 2 and 4
    assert recall_at_k(ranked, targets, k=2) == pytest.approx((0.5 + 0.0) / 2)
    assert recall_at_k(ranked, targets, k=4) == pytest.approx((1.0 + 0.0) / 2)
    dcg = 1 / math.log2(3) + 1 / math.log2(5)
    idcg = 1 / math.log2(2) + 1 / math.log2(3)
    assert ndcg_at_k(ranked, targets, k=4) == pytest.approx((dcg / idcg + 0.0) / 2)


def test_perfect_and_empty_rankings() -> None:
    ranked = np.array([[3, 1, 2]])
    assert ndcg_at_k(ranked, np.array([3]), k=3) == pytest.approx(1.0)
    assert recall_at_k(ranked, np.array([3]), k=1) == pytest.approx(1.0)
    assert ndcg_at_k(ranked, np.array([[3, 1, 2]]), k=3) == pytest.approx(1.0)
    # Users without relevant items are ignored rather than counted as zeros.
    assert recall_at_k(ranked, np.array([[-1]]), k=3) == 0.0


def test_compute_metrics_keys_and_bounds() -> None:
    metrics = compute_metrics(RANKED, TARGETS, ks=(1, 4))
    assert set(metrics) == {"Recall@1", "NDCG@1", "Recall@4", "NDCG@4"}
    assert all(0.0 <= v <= 1.0 for v in metrics.values())
    assert metrics["NDCG@4"] <= metrics["Recall@4"]
    with pytest.raises(ValueError):
        compute_metrics(RANKED, TARGETS, ks=(10,))


def test_mismatched_shapes_raise() -> None:
    with pytest.raises(ValueError):
        recall_at_k(RANKED, np.array([1, 2]), k=2)
