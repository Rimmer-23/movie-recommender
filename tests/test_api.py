"""API endpoints (FastAPI TestClient) and the serving layer behind them."""

from __future__ import annotations

from collections.abc import Iterator
from datetime import UTC, datetime
from pathlib import Path

import pyarrow.parquet as pq
import pytest
from fastapi.testclient import TestClient

from recsys.api import create_app
from recsys.config import ExperimentConfig, TrainConfig
from recsys.data import SplitData
from recsys.models import build_model
from recsys.serving import RECOMMENDATIONS_SCHEMA, RecommendationService, write_recommendations


@pytest.fixture(scope="module")
def service(split: SplitData) -> RecommendationService:
    config = ExperimentConfig(
        model="mf",
        seed=0,
        params={"embedding_dim": 8},
        train=TrainConfig(epochs=2, batch_size=32, lr=1e-2, device="cpu"),
    )
    model = build_model(config, split.n_users, split.n_items).fit(split)
    return RecommendationService(model, split)


@pytest.fixture(scope="module")
def client(service: RecommendationService) -> Iterator[TestClient]:
    with TestClient(create_app(service)) as test_client:
        yield test_client


def test_health(client: TestClient, split: SplitData) -> None:
    response = client.get("/health")
    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "ok"
    assert body["model"] == "mf"
    assert body["n_users"] == split.n_users and body["n_items"] == split.n_items


def test_recommend_known_user(client: TestClient, split: SplitData) -> None:
    user = split.users.iloc[3]
    response = client.get(f"/recommend/{user.user_id}", params={"k": 5})
    assert response.status_code == 200
    body = response.json()
    assert body["user_id"] == user.user_id
    assert body["model"] == "mf" and body["fallback"] is False
    recs = body["recommendations"]
    assert [r["rank"] for r in recs] == [1, 2, 3, 4, 5]
    scores = [r["score"] for r in recs]
    assert scores == sorted(scores, reverse=True)

    titles = split.items.set_index("movie_id")["title"]
    item_to_movie = split.items.set_index("item_idx")["movie_id"]
    watched = set(item_to_movie.loc[split.history("full")[user.user_idx]])
    for rec in recs:
        assert rec["title"] == titles[rec["movie_id"]]
        assert rec["movie_id"] not in watched  # everything the user has seen is excluded


def test_unknown_user_gets_popular_fallback(client: TestClient, split: SplitData) -> None:
    response = client.get("/recommend/999999", params={"k": 3})
    assert response.status_code == 200
    body = response.json()
    assert body["fallback"] is True and body["model"] == "popular"
    assert len(body["recommendations"]) == 3
    item_counts = {}
    for history in split.history("full"):
        for item in history:
            item_counts[item] = item_counts.get(item, 0) + 1
    top_count = max(item_counts.values())
    assert body["recommendations"][0]["score"] == top_count


def test_default_k_is_ten(client: TestClient, split: SplitData) -> None:
    user_id = int(split.users.iloc[0].user_id)
    body = client.get(f"/recommend/{user_id}").json()
    assert len(body["recommendations"]) == min(10, split.n_items - len(split.sequences[0]))


@pytest.mark.parametrize("k", [0, -1, 101])
def test_invalid_k_is_rejected(client: TestClient, k: int) -> None:
    assert client.get("/recommend/1", params={"k": k}).status_code == 422


def test_non_integer_user_is_rejected(client: TestClient) -> None:
    assert client.get("/recommend/abc").status_code == 422


def test_batch_export_table(service: RecommendationService, split: SplitData, tmp_path: Path):
    generated_at = datetime(2026, 1, 1, tzinfo=UTC)
    table = service.recommend_all(k=4, batch_size=7, generated_at=generated_at)
    assert table.schema == RECOMMENDATIONS_SCHEMA
    frame = table.to_pandas()
    assert frame["user_id"].nunique() == split.n_users
    assert (frame.groupby("user_id")["rank"].apply(list) == [[1, 2, 3, 4]] * split.n_users).all()
    assert (frame["model"] == "mf").all()
    assert (frame["generated_at"] == generated_at).all()
    # Same lists as the online endpoint.
    first_user = int(split.users.iloc[0].user_id)
    online = [r.movie_id for r in service.recommend(first_user, k=4).items]
    assert frame[frame["user_id"] == first_user]["movie_id"].tolist() == online

    path = tmp_path / "recs.parquet"
    write_recommendations(table, path)
    assert pq.read_table(path).equals(table)


def test_service_rejects_mismatched_data(split: SplitData) -> None:
    config = ExperimentConfig(model="popular")
    model = build_model(config, split.n_users + 1, split.n_items)
    with pytest.raises(ValueError):
        RecommendationService(model, split)
