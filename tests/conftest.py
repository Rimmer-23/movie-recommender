"""Shared fixtures: a small synthetic dataset in the raw MovieLens 1M schema.

Tests never need the real dataset, so they run in CI without downloading anything.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from recsys.data import ProcessedData, SplitData, build_split, preprocess

GENRES = ["Action", "Comedy", "Drama", "Horror", "Romance", "Sci-Fi"]


def make_raw_movielens(
    n_users: int = 40, n_movies: int = 30, seed: int = 0
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Synthetic ``ratings``, ``movies`` and ``users`` tables with non-contiguous raw IDs."""
    rng = np.random.default_rng(seed)
    movie_ids = np.arange(1, n_movies + 1) * 10  # 10, 20, ... (gaps test re-indexing)
    movies = pd.DataFrame(
        {
            "movie_id": movie_ids,
            "title": [f"Movie {i} (1999)" for i in range(n_movies)],
            "genres": [
                "|".join(rng.choice(GENRES, size=rng.integers(1, 4), replace=False))
                for _ in range(n_movies)
            ],
        }
    )
    user_ids = 100 + 3 * np.arange(n_users)
    users = pd.DataFrame(
        {
            "user_id": user_ids,
            "gender": rng.choice(["F", "M"], size=n_users),
            "age": rng.choice([1, 18, 25, 35, 45, 50, 56], size=n_users),
            "occupation": rng.integers(0, 21, size=n_users),
            "zip": ["00000"] * n_users,
        }
    )
    popularity = 1.0 / np.arange(1, n_movies + 1)
    popularity /= popularity.sum()
    rows = []
    for user in user_ids:
        n = int(rng.integers(5, 16))
        chosen = rng.choice(movie_ids, size=n, replace=False, p=popularity)
        # Non-decreasing timestamps with occasional ties, like bulk-rating sessions.
        timestamps = 978_300_000 + np.cumsum(rng.integers(0, 3, size=n)) * 60
        rows += [
            (user, movie, int(rng.integers(1, 6)), ts)
            for movie, ts in zip(chosen, timestamps, strict=True)
        ]
    ratings = pd.DataFrame(rows, columns=["user_id", "movie_id", "rating", "timestamp"])
    # Shuffle so preprocessing has to do the sorting itself.
    ratings = ratings.sample(frac=1.0, random_state=seed).reset_index(drop=True)
    return ratings, movies, users


@pytest.fixture(scope="session")
def raw_tables() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    return make_raw_movielens()


@pytest.fixture(scope="session")
def processed(raw_tables: tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]) -> ProcessedData:
    return preprocess(*raw_tables)


@pytest.fixture(scope="session")
def split(processed: ProcessedData) -> SplitData:
    return build_split(processed)
