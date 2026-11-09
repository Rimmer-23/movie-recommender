"""Serving layer shared by the API and the batch export.

Loads the selected model and the catalogue, maps raw MovieLens IDs to internal indices,
uses each user's *full* history (train + valid + test) so recommendations reflect
everything the user has watched, and falls back to global popularity for unknown users.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from recsys.config import (
    DEFAULT_ARTIFACTS_DIR,
    DEFAULT_DATA_DIR,
    checkpoint_path,
    results_path,
)
from recsys.data import SplitData, load_dataset
from recsys.models import PopularRecommender, Recommender, load_model

RECOMMENDATIONS_SCHEMA = pa.schema(
    [
        pa.field("user_id", pa.int32(), nullable=False),
        pa.field("rank", pa.int16(), nullable=False),
        pa.field("movie_id", pa.int32(), nullable=False),
        pa.field("title", pa.string(), nullable=False),
        pa.field("score", pa.float32(), nullable=False),
        pa.field("model", pa.string(), nullable=False),
        pa.field("generated_at", pa.timestamp("us", tz="UTC"), nullable=False),
    ]
)
"""Schema of the batch export, i.e. the table delivered to the warehouse."""


@dataclass(frozen=True)
class Recommendation:
    rank: int
    movie_id: int
    title: str
    score: float


@dataclass(frozen=True)
class RecommendationResult:
    user_id: int
    model: str
    fallback: bool
    items: list[Recommendation]


def best_model_name(artifacts_dir: Path) -> str:
    """Name of the model selected by ``scripts/evaluate.py`` (best validation NDCG@10)."""
    path = results_path(artifacts_dir)
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found. Run `python scripts/evaluate.py` to select the best model, "
            "or set RECSYS_MODEL."
        )
    return json.loads(path.read_text(encoding="utf-8"))["best_model"]


class RecommendationService:
    """Top-k recommendations by raw MovieLens ``user_id`` with a popularity fallback."""

    def __init__(self, model: Recommender, data: SplitData) -> None:
        if (model.n_users, model.n_items) != (data.n_users, data.n_items):
            raise ValueError(
                f"Model was trained on {model.n_users} users / {model.n_items} items but the "
                f"data has {data.n_users} / {data.n_items}; retrain or point to the right data."
            )
        self.model = model
        self.model_name = model.name
        history = data.history("full")
        self.model.set_history(history)
        self.popular = PopularRecommender(data.n_users, data.n_items).fit_items(
            np.concatenate(history)
        )
        self.user_index: dict[int, int] = dict(
            zip(data.users["user_id"].tolist(), data.users["user_idx"].tolist(), strict=True)
        )
        self.user_ids = data.users.sort_values("user_idx")["user_id"].to_numpy()
        # Lookup tables indexed by internal item id (row 0 = padding).
        self.movie_ids = np.zeros(data.n_items + 1, dtype=np.int64)
        self.movie_ids[data.items["item_idx"]] = data.items["movie_id"]
        self.titles = np.full(data.n_items + 1, "", dtype=object)
        self.titles[data.items["item_idx"]] = data.items["title"]

    @classmethod
    def from_artifacts(
        cls,
        artifacts_dir: Path = DEFAULT_ARTIFACTS_DIR,
        data_dir: Path = DEFAULT_DATA_DIR,
        model_name: str | None = None,
        device: str | None = "cpu",
    ) -> RecommendationService:
        """Load the processed data and a trained model (the best one by default)."""
        model_name = model_name or os.getenv("RECSYS_MODEL") or best_model_name(artifacts_dir)
        model = load_model(checkpoint_path(artifacts_dir, model_name), device=device)
        return cls(model, load_dataset(data_dir))

    @property
    def n_users(self) -> int:
        return len(self.user_ids)

    @property
    def n_items(self) -> int:
        return len(self.movie_ids) - 1

    def recommend(self, user_id: int, k: int = 10) -> RecommendationResult:
        """Top-``k`` unseen movies for a raw MovieLens user id (popular items if unknown)."""
        user_idx = self.user_index.get(user_id)
        if user_idx is None:
            items, scores = self.popular.recommend([0], k=k, exclude_seen=False)
            model_name, fallback = self.popular.name, True
        else:
            items, scores = self.model.recommend([user_idx], k=k, exclude_seen=True)
            model_name, fallback = self.model_name, False
        recommendations = [
            Recommendation(
                rank=rank,
                movie_id=int(self.movie_ids[item]),
                title=str(self.titles[item]),
                score=float(score),
            )
            for rank, (item, score) in enumerate(zip(items[0], scores[0], strict=True), start=1)
            if np.isfinite(score)
        ]
        return RecommendationResult(user_id, model_name, fallback, recommendations)

    def recommend_all(
        self, k: int = 10, batch_size: int = 1024, generated_at: datetime | None = None
    ) -> pa.Table:
        """Top-``k`` for every known user as an Arrow table with :data:`RECOMMENDATIONS_SCHEMA`."""
        generated_at = generated_at or datetime.now(UTC).replace(microsecond=0)
        frames = []
        for start in range(0, self.n_users, batch_size):
            users = np.arange(start, min(start + batch_size, self.n_users))
            items, scores = self.model.recommend(users, k=k, exclude_seen=True)  # (batch, k)
            n_users, n_ranks = items.shape
            frames.append(
                pd.DataFrame(
                    {
                        "user_id": np.repeat(self.user_ids[users], n_ranks),
                        "rank": np.tile(np.arange(1, n_ranks + 1), n_users),
                        "movie_id": self.movie_ids[items.ravel()],
                        "title": self.titles[items.ravel()],
                        "score": scores.ravel(),
                    }
                )
            )
        frame = pd.concat(frames, ignore_index=True)
        frame = frame[np.isfinite(frame["score"])]
        frame["model"] = self.model_name
        frame["generated_at"] = pd.Timestamp(generated_at)
        return pa.Table.from_pandas(frame, schema=RECOMMENDATIONS_SCHEMA, preserve_index=False)


def write_recommendations(table: pa.Table, path: Path) -> None:
    """Write the export table to parquet."""
    path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(table, path)
