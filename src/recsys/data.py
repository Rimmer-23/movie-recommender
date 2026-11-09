"""Data pipeline for MovieLens 1M.

Raw ``.dat`` files → preprocessing (implicit feedback, chronological order, contiguous IDs)
→ leave-one-out split by time → PyTorch ``Dataset`` objects for each model family.

ID conventions
--------------
* Users are re-indexed to ``0 .. n_users - 1``.
* Items are re-indexed to ``1 .. n_items``; index ``0`` is reserved for padding, so every
  item-score tensor has ``n_items + 1`` columns and column 0 is never recommended.

Split protocol (identical for every model)
------------------------------------------
For each user, interactions are sorted by timestamp (ties keep the original file order):
the last one is the **test** target, the second-to-last the **validation** target and
everything before is **train**.
"""

from __future__ import annotations

import csv
import io
import ssl
import urllib.request
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import numpy as np
import pandas as pd
import torch
from torch.utils.data import BatchSampler, DataLoader, Dataset, RandomSampler, SequentialSampler

ML1M_URL = "https://files.grouplens.org/datasets/movielens/ml-1m.zip"
PAD_ITEM = 0

Stage = Literal["valid", "test", "full"]
"""Which part of the timeline a model sees: ``valid`` → train only, ``test`` → train + valid,
``full`` → everything (used for serving and batch export)."""


# --------------------------------------------------------------------------------------------
# Download & raw files
# --------------------------------------------------------------------------------------------


def download_movielens(raw_dir: Path, url: str = ML1M_URL, force: bool = False) -> Path:
    """Download and unzip MovieLens 1M into ``raw_dir``.

    Returns:
        Path to the extracted ``ml-1m`` directory.
    """
    target = raw_dir / "ml-1m"
    if (target / "ratings.dat").exists() and not force:
        return target
    raw_dir.mkdir(parents=True, exist_ok=True)
    zip_path = raw_dir / "ml-1m.zip"
    if not zip_path.exists() or force:
        _stream_to_file(url, zip_path)
    with zipfile.ZipFile(zip_path) as archive:
        for member in archive.namelist():
            if member.startswith(("/", "\\")) or ".." in Path(member).parts:
                raise ValueError(f"Unsafe path in archive: {member}")
        archive.extractall(raw_dir)
    return target


def _stream_to_file(url: str, path: Path, chunk_size: int = 1 << 16) -> None:
    """Download ``url`` to ``path`` in chunks, printing progress; atomic via a ``.part`` file."""
    partial = path.with_suffix(path.suffix + ".part")
    with (
        urllib.request.urlopen(url, timeout=120, context=_ssl_context()) as response,
        partial.open("wb") as out,
    ):
        total = int(response.headers.get("Content-Length", 0))
        done = 0
        while chunk := response.read(chunk_size):
            out.write(chunk)
            done += len(chunk)
            if total:
                print(f"\rDownloading {url}: {done / total:6.1%}", end="", flush=True)
    print()
    partial.replace(path)


def _ssl_context() -> ssl.SSLContext:
    """Default TLS context, using certifi's CA bundle when available.

    Some Python builds (notably python.org installers on Windows/macOS) ship without access
    to the system certificate store, which breaks HTTPS downloads.
    """
    try:
        import certifi
    except ImportError:
        return ssl.create_default_context()
    return ssl.create_default_context(cafile=certifi.where())


def _read_dat(path: Path, columns: list[str]) -> pd.DataFrame:
    """Read a MovieLens ``::``-separated file with the fast C parser."""
    text = path.read_text(encoding="latin-1").replace("::", "\t")
    return pd.read_csv(
        io.StringIO(text), sep="\t", names=columns, header=None, quoting=csv.QUOTE_NONE
    )


def read_raw_movielens(ml_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Read ``ratings.dat``, ``movies.dat`` and ``users.dat`` from an ``ml-1m`` directory."""
    ratings = _read_dat(ml_dir / "ratings.dat", ["user_id", "movie_id", "rating", "timestamp"])
    movies = _read_dat(ml_dir / "movies.dat", ["movie_id", "title", "genres"])
    users = _read_dat(ml_dir / "users.dat", ["user_id", "gender", "age", "occupation", "zip"])
    return ratings, movies, users


# --------------------------------------------------------------------------------------------
# Preprocessing
# --------------------------------------------------------------------------------------------


@dataclass
class ProcessedData:
    """Re-indexed interaction log plus item and user tables.

    Attributes:
        interactions: ``user_idx, item_idx, rating, timestamp`` sorted by user, then time.
        items: ``item_idx, movie_id, title, genres`` (``item_idx`` in ``1 .. n_items``).
        users: ``user_idx, user_id, gender, age, occupation`` (``user_idx`` in ``0 .. n-1``).
    """

    interactions: pd.DataFrame
    items: pd.DataFrame
    users: pd.DataFrame

    _FILES = ("interactions", "items", "users")

    def save(self, directory: Path) -> None:
        """Write the three tables as parquet files into ``directory``."""
        directory.mkdir(parents=True, exist_ok=True)
        for name in self._FILES:
            getattr(self, name).to_parquet(directory / f"{name}.parquet", index=False)

    @classmethod
    def load(cls, directory: Path) -> ProcessedData:
        """Load tables previously written by :meth:`save`."""
        missing = [n for n in cls._FILES if not (directory / f"{n}.parquet").exists()]
        if missing:
            raise FileNotFoundError(
                f"Processed data not found in {directory} (missing {missing}). "
                "Run `python scripts/download_data.py` first."
            )
        tables = {n: pd.read_parquet(directory / f"{n}.parquet") for n in cls._FILES}
        return cls(**tables)


def preprocess(
    ratings: pd.DataFrame,
    movies: pd.DataFrame,
    users: pd.DataFrame,
    min_user_interactions: int = 3,
) -> ProcessedData:
    """Turn raw MovieLens tables into an implicit-feedback log with contiguous IDs.

    Every rating counts as a positive interaction regardless of its value. Users with fewer
    than ``min_user_interactions`` events are dropped (leave-one-out needs at least 3).
    Each user's history is sorted by timestamp; ties keep the original file order.
    """
    log = ratings[["user_id", "movie_id", "rating", "timestamp"]].copy()
    log = log.drop_duplicates(subset=["user_id", "movie_id"], keep="last")
    counts = log.groupby("user_id")["movie_id"].transform("size")
    log = log[counts >= min_user_interactions]

    user_ids = np.sort(log["user_id"].unique())
    movie_ids = np.sort(log["movie_id"].unique())
    user_index = pd.Series(np.arange(len(user_ids)), index=user_ids)
    item_index = pd.Series(np.arange(1, len(movie_ids) + 1), index=movie_ids)

    log = log.sort_values(["user_id", "timestamp"], kind="stable")
    interactions = pd.DataFrame(
        {
            "user_idx": user_index.loc[log["user_id"]].to_numpy(),
            "item_idx": item_index.loc[log["movie_id"]].to_numpy(),
            "rating": log["rating"].to_numpy(),
            "timestamp": log["timestamp"].to_numpy(),
        }
    )

    items = (
        movies[movies["movie_id"].isin(movie_ids)]
        .assign(item_idx=lambda df: item_index.loc[df["movie_id"]].to_numpy())
        .sort_values("item_idx")[["item_idx", "movie_id", "title", "genres"]]
        .reset_index(drop=True)
    )
    users_table = (
        users[users["user_id"].isin(user_ids)]
        .assign(user_idx=lambda df: user_index.loc[df["user_id"]].to_numpy())
        .sort_values("user_idx")[["user_idx", "user_id", "gender", "age", "occupation"]]
        .reset_index(drop=True)
    )
    return ProcessedData(interactions=interactions, items=items, users=users_table)


# --------------------------------------------------------------------------------------------
# Leave-one-out split
# --------------------------------------------------------------------------------------------


@dataclass
class SplitData:
    """Chronological per-user sequences with a leave-one-out-by-time split.

    ``sequences[u]`` holds every item user ``u`` interacted with, oldest first. The split is
    implicit: ``sequences[u][:-2]`` is train, ``[-2]`` is the validation target and ``[-1]``
    is the test target.
    """

    sequences: list[np.ndarray]
    n_items: int
    items: pd.DataFrame
    users: pd.DataFrame

    @property
    def n_users(self) -> int:
        return len(self.sequences)

    def history(self, stage: Stage) -> list[np.ndarray]:
        """Items a model may use as input when predicting the ``stage`` target.

        ``valid`` → train items, ``test`` → train + validation item, ``full`` → all items.
        """
        cut = {"valid": -2, "test": -1, "full": None}[stage]
        return [seq[:cut] for seq in self.sequences]

    def targets(self, stage: Literal["valid", "test"]) -> np.ndarray:
        """Held-out item per user for the given stage, shape ``(n_users,)``."""
        position = {"valid": -2, "test": -1}[stage]
        return np.array([seq[position] for seq in self.sequences], dtype=np.int64)

    def train_pairs(self) -> tuple[np.ndarray, np.ndarray]:
        """All train ``(user, item)`` interactions as two aligned arrays."""
        train = self.history("valid")
        users = np.repeat(np.arange(self.n_users), [len(s) for s in train])
        items = np.concatenate(train).astype(np.int64)
        return users.astype(np.int64), items

    def item_genre_matrix(self) -> tuple[np.ndarray, int]:
        """Genre ids per item, left-aligned and 0-padded.

        Returns:
            ``(genres, n_genres)`` where ``genres`` has shape ``(n_items + 1, max_genres)``
            and genre ids run from 1 to ``n_genres`` (0 = padding).
        """
        genre_lists = self.items["genres"].str.split("|")
        vocab = {g: i + 1 for i, g in enumerate(sorted({g for gs in genre_lists for g in gs}))}
        max_genres = max(len(gs) for gs in genre_lists)
        matrix = np.zeros((self.n_items + 1, max_genres), dtype=np.int64)
        for item_idx, genres in zip(self.items["item_idx"], genre_lists, strict=True):
            matrix[item_idx, : len(genres)] = [vocab[g] for g in genres]
        return matrix, len(vocab)

    def user_feature_matrix(self) -> tuple[np.ndarray, list[int]]:
        """Categorical user features ``gender, age, occupation`` encoded as ids ≥ 1.

        Returns:
            ``(features, cardinalities)`` with ``features`` of shape ``(n_users, 3)`` and the
            embedding table size needed for each column (including the 0 "unknown" id).
        """
        features = np.zeros((self.n_users, 3), dtype=np.int64)
        cardinalities = []
        users = self.users.set_index("user_idx").reindex(range(self.n_users))
        for col, name in enumerate(["gender", "age", "occupation"]):
            codes, uniques = pd.factorize(users[name], sort=True)
            features[:, col] = codes + 1  # factorize marks missing values as -1 → 0
            cardinalities.append(len(uniques) + 1)
        return features, cardinalities


def build_split(processed: ProcessedData) -> SplitData:
    """Group the (already time-sorted) log into per-user sequences."""
    log = processed.interactions
    n_users = int(log["user_idx"].max()) + 1
    boundaries = np.cumsum(np.bincount(log["user_idx"].to_numpy(), minlength=n_users))[:-1]
    sequences = np.split(log["item_idx"].to_numpy(dtype=np.int64), boundaries)
    return SplitData(
        sequences=sequences,
        n_items=int(processed.items["item_idx"].max()),
        items=processed.items,
        users=processed.users,
    )


def load_dataset(processed_dir: Path) -> SplitData:
    """Load processed MovieLens tables and build the leave-one-out split."""
    return build_split(ProcessedData.load(processed_dir))


def dataset_stats(processed: ProcessedData) -> dict[str, float]:
    """Summary statistics of the interaction log and its split."""
    log = processed.interactions
    n_users = log["user_idx"].nunique()
    n_items = log["item_idx"].nunique()
    per_user = log.groupby("user_idx").size()
    per_item = log.groupby("item_idx").size()
    timestamps = pd.to_datetime(log["timestamp"], unit="s")
    return {
        "users": n_users,
        "items": n_items,
        "interactions": len(log),
        "density_%": 100.0 * len(log) / (n_users * n_items),
        "per_user_min": int(per_user.min()),
        "per_user_median": float(per_user.median()),
        "per_user_mean": float(per_user.mean()),
        "per_user_max": int(per_user.max()),
        "per_item_min": int(per_item.min()),
        "per_item_median": float(per_item.median()),
        "per_item_mean": float(per_item.mean()),
        "per_item_max": int(per_item.max()),
        "first_event": str(timestamps.min().date()),
        "last_event": str(timestamps.max().date()),
        "train_interactions": len(log) - 2 * n_users,
        "valid_interactions": n_users,
        "test_interactions": n_users,
    }


# --------------------------------------------------------------------------------------------
# PyTorch datasets
# --------------------------------------------------------------------------------------------


def pad_left(sequence: np.ndarray, max_len: int) -> np.ndarray:
    """Keep the ``max_len`` most recent items and left-pad with ``PAD_ITEM``."""
    out = np.full(max_len, PAD_ITEM, dtype=np.int64)
    tail = sequence[-max_len:]
    if len(tail):
        out[-len(tail) :] = tail
    return out


class BatchIndexDataset(Dataset):
    """Dataset whose ``__getitem__`` receives a *list* of indices and returns a whole batch.

    Combined with :func:`make_batch_loader` this keeps the standard ``Dataset``/``DataLoader``
    API while building each batch with vectorised NumPy ops instead of collating thousands
    of per-example Python tuples (≈1M train interactions per epoch).
    """

    def __len__(self) -> int:
        raise NotImplementedError

    def __getitem__(self, indices: list[int]) -> dict[str, torch.Tensor]:
        raise NotImplementedError


def make_batch_loader(
    dataset: BatchIndexDataset, batch_size: int, shuffle: bool, seed: int
) -> DataLoader:
    """``DataLoader`` that hands lists of indices to a :class:`BatchIndexDataset`."""
    generator = torch.Generator().manual_seed(seed)
    base = RandomSampler(dataset, generator=generator) if shuffle else SequentialSampler(dataset)
    sampler = BatchSampler(base, batch_size=batch_size, drop_last=False)
    return DataLoader(dataset, sampler=sampler, batch_size=None)


class InteractionDataset(BatchIndexDataset):
    """Positive ``(user, item)`` pairs, e.g. for in-batch-negative training."""

    def __init__(self, users: np.ndarray, items: np.ndarray) -> None:
        self.users = torch.as_tensor(users, dtype=torch.long)
        self.items = torch.as_tensor(items, dtype=torch.long)

    def __len__(self) -> int:
        return len(self.users)

    def __getitem__(self, indices: list[int]) -> dict[str, torch.Tensor]:
        idx = torch.as_tensor(indices, dtype=torch.long)
        return {"users": self.users[idx], "items": self.items[idx]}


class BPRDataset(InteractionDataset):
    """``(user, positive, negative)`` triplets with uniform negative sampling.

    Negatives are drawn uniformly from ``1 .. n_items`` and re-drawn (a few times) if the
    user has interacted with them in train, so they are true negatives in all but rare cases.
    """

    def __init__(
        self,
        users: np.ndarray,
        items: np.ndarray,
        n_items: int,
        seed: int,
        max_resample: int = 10,
    ) -> None:
        super().__init__(users, items)
        self.n_items = n_items
        self.max_resample = max_resample
        self.rng = np.random.default_rng(seed)
        # Sorted (user, item) keys allow O(log n) membership tests with searchsorted.
        self._seen = np.unique(users.astype(np.int64) * (n_items + 1) + items)

    def _is_seen(self, users: np.ndarray, items: np.ndarray) -> np.ndarray:
        keys = users * (self.n_items + 1) + items
        pos = np.minimum(np.searchsorted(self._seen, keys), len(self._seen) - 1)
        return self._seen[pos] == keys

    def sample_negatives(self, users: np.ndarray) -> np.ndarray:
        """Draw one negative item per user."""
        negatives = self.rng.integers(1, self.n_items + 1, size=len(users))
        for _ in range(self.max_resample):
            clash = self._is_seen(users, negatives)
            if not clash.any():
                break
            negatives[clash] = self.rng.integers(1, self.n_items + 1, size=int(clash.sum()))
        return negatives

    def __getitem__(self, indices: list[int]) -> dict[str, torch.Tensor]:
        batch = super().__getitem__(indices)
        negatives = self.sample_negatives(batch["users"].numpy())
        return {
            "users": batch["users"],
            "pos_items": batch["items"],
            "neg_items": torch.from_numpy(negatives),
        }


class NextItemDataset(Dataset):
    """One left-padded training sequence per user for next-item prediction (SASRec).

    For a train sequence ``[i1, ..., iT]`` the input is ``[i1, ..., i(T-1)]`` and the target
    is the same sequence shifted by one, ``[i2, ..., iT]``; both are cut to the most recent
    ``max_len`` positions and left-padded with 0.
    """

    def __init__(self, sequences: list[np.ndarray], max_len: int) -> None:
        self.sequences = [s for s in sequences if len(s) >= 2]
        self.max_len = max_len

    def __len__(self) -> int:
        return len(self.sequences)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        seq = self.sequences[index]
        return {
            "inputs": torch.from_numpy(pad_left(seq[:-1], self.max_len)),  # (max_len,)
            "targets": torch.from_numpy(pad_left(seq[1:], self.max_len)),  # (max_len,)
        }
