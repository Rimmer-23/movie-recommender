"""Generate a tiny synthetic dataset and a trained model for smoke-testing the API.

CI has neither MovieLens (its licence forbids redistribution) nor checkpoints, so the
Docker job runs this script inside the freshly built image and then serves the result.
It needs only the runtime dependencies.

Usage:
    python scripts/make_smoke_artifacts.py [--output smoke]

Writes ``<output>/data/processed`` and ``<output>/artifacts`` in the layout the API
expects, plus ``<output>/known_user_id.txt`` with one valid raw user id to query.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from recsys.config import ExperimentConfig, TrainConfig, checkpoint_path, results_path
from recsys.data import build_split, preprocess
from recsys.models import build_model

N_USERS, N_MOVIES, SEED = 50, 40, 0
GENRES = ["Action", "Comedy", "Drama", "Horror", "Romance", "Sci-Fi"]


def synthetic_movielens() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Random ``ratings``, ``movies`` and ``users`` tables in the raw MovieLens 1M schema."""
    rng = np.random.default_rng(SEED)
    movie_ids = np.arange(1, N_MOVIES + 1) * 10
    user_ids = np.arange(1, N_USERS + 1)
    movies = pd.DataFrame(
        {
            "movie_id": movie_ids,
            "title": [f"Smoke Movie {i} (1999)" for i in range(N_MOVIES)],
            "genres": rng.choice(GENRES, size=N_MOVIES),
        }
    )
    users = pd.DataFrame(
        {
            "user_id": user_ids,
            "gender": rng.choice(["F", "M"], size=N_USERS),
            "age": rng.choice([18, 25, 35], size=N_USERS),
            "occupation": rng.integers(0, 5, size=N_USERS),
            "zip": "00000",
        }
    )
    rows = [
        (user, movie, 5, 978_300_000 + 60 * step)
        for user in user_ids
        for step, movie in enumerate(rng.choice(movie_ids, size=10, replace=False))
    ]
    ratings = pd.DataFrame(rows, columns=["user_id", "movie_id", "rating", "timestamp"])
    return ratings, movies, users


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--output", type=Path, default=Path("smoke"))
    args = parser.parse_args()

    processed = preprocess(*synthetic_movielens())
    processed.save(args.output / "data" / "processed")
    split = build_split(processed)

    config = ExperimentConfig(
        model="mf",
        seed=SEED,
        params={"embedding_dim": 8},
        train=TrainConfig(epochs=2, batch_size=64, lr=1e-2, device="cpu"),
    )
    model = build_model(config, split.n_users, split.n_items).fit(split)
    artifacts_dir = args.output / "artifacts"
    model.save(checkpoint_path(artifacts_dir, model.name))
    # The API picks its model from results.json, exactly as after scripts/evaluate.py.
    results_path(artifacts_dir).write_text(json.dumps({"best_model": model.name}), encoding="utf-8")

    known_user = int(processed.users["user_id"].iloc[0])
    (args.output / "known_user_id.txt").write_text(f"{known_user}\n", encoding="utf-8")
    print(f"Wrote {args.output}: {split.n_users} users, {split.n_items} items, model {model.name}")


if __name__ == "__main__":
    main()
