"""Preprocessing, ID re-indexing, leave-one-out split (no leakage) and torch datasets."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from recsys.data import (
    BPRDataset,
    InteractionDataset,
    NextItemDataset,
    ProcessedData,
    SplitData,
    build_split,
    dataset_stats,
    make_batch_loader,
    pad_left,
    preprocess,
    read_raw_movielens,
)


def test_ids_are_contiguous(processed: ProcessedData) -> None:
    log = processed.interactions
    n_users, n_items = log["user_idx"].nunique(), log["item_idx"].nunique()
    assert sorted(log["user_idx"].unique()) == list(range(n_users))
    assert sorted(log["item_idx"].unique()) == list(range(1, n_items + 1))  # 0 = padding
    assert processed.items["item_idx"].tolist() == list(range(1, n_items + 1))
    assert processed.users["user_idx"].tolist() == list(range(n_users))


def test_reindexing_preserves_raw_ids(raw_tables, processed: ProcessedData) -> None:
    ratings, _, _ = raw_tables
    item_to_movie = processed.items.set_index("item_idx")["movie_id"]
    idx_to_user = processed.users.set_index("user_idx")["user_id"]
    restored = pd.DataFrame(
        {
            "user_id": idx_to_user.loc[processed.interactions["user_idx"]].to_numpy(),
            "movie_id": item_to_movie.loc[processed.interactions["item_idx"]].to_numpy(),
            "timestamp": processed.interactions["timestamp"].to_numpy(),
        }
    )
    key = ["user_id", "movie_id", "timestamp"]
    expected = ratings[key].sort_values(key).reset_index(drop=True)
    pd.testing.assert_frame_equal(restored.sort_values(key).reset_index(drop=True), expected)


def test_histories_are_chronological(processed: ProcessedData) -> None:
    diffs = processed.interactions.groupby("user_idx")["timestamp"].diff().dropna()
    assert (diffs >= 0).all()


def test_min_user_interactions_filter(raw_tables) -> None:
    ratings, movies, users = raw_tables
    first_user = ratings["user_id"].iloc[0]
    tiny = pd.concat(
        [ratings[ratings["user_id"] != first_user], ratings[ratings["user_id"] == first_user][:2]]
    )
    processed = preprocess(tiny, movies, users, min_user_interactions=3)
    assert first_user not in set(processed.users["user_id"])


def test_split_has_no_leakage(split: SplitData, processed: ProcessedData) -> None:
    valid_targets, test_targets = split.targets("valid"), split.targets("test")
    train = split.history("valid")
    timestamps = processed.interactions.groupby("user_idx")["timestamp"].apply(np.asarray)
    for user in range(split.n_users):
        assert valid_targets[user] not in train[user]
        assert test_targets[user] not in train[user]
        assert valid_targets[user] != test_targets[user]
        ts = timestamps[user]
        # Every train event happens no later than the validation event, which is no later
        # than the test event.
        assert ts[:-2].max() <= ts[-2] <= ts[-1]
    n_train = sum(len(h) for h in train)
    assert n_train + 2 * split.n_users == len(processed.interactions)


def test_history_stages_grow_by_one(split: SplitData) -> None:
    for stage_a, stage_b in [("valid", "test"), ("test", "full")]:
        for a, b in zip(split.history(stage_a), split.history(stage_b), strict=True):
            assert len(b) == len(a) + 1
            np.testing.assert_array_equal(b[:-1], a)
    np.testing.assert_array_equal([h[-1] for h in split.history("test")], split.targets("valid"))


def test_train_pairs_match_history(split: SplitData) -> None:
    users, items = split.train_pairs()
    assert len(users) == len(items) == sum(len(h) for h in split.history("valid"))
    assert set(items[users == 0]) == set(split.history("valid")[0])


def test_features(split: SplitData) -> None:
    genres, n_genres = split.item_genre_matrix()
    assert genres.shape[0] == split.n_items + 1
    assert (genres[0] == 0).all()  # padding item has no genres
    assert genres.max() == n_genres and (genres[1:, 0] > 0).all()
    features, cards = split.user_feature_matrix()
    assert features.shape == (split.n_users, 3)
    assert (features >= 1).all() and all(features[:, i].max() < c for i, c in enumerate(cards))


def test_processed_roundtrip(processed: ProcessedData, tmp_path: Path) -> None:
    processed.save(tmp_path)
    loaded = ProcessedData.load(tmp_path)
    pd.testing.assert_frame_equal(loaded.interactions, processed.interactions)
    pd.testing.assert_frame_equal(loaded.items, processed.items)


def test_read_raw_dat_files(raw_tables, tmp_path: Path) -> None:
    ratings, movies, users = raw_tables
    for name, frame in [("ratings", ratings), ("movies", movies), ("users", users)]:
        lines = frame.astype(str).agg("::".join, axis=1)
        (tmp_path / f"{name}.dat").write_text("\n".join(lines) + "\n", encoding="latin-1")
    r, m, u = read_raw_movielens(tmp_path)
    pd.testing.assert_frame_equal(r, ratings, check_dtype=False)
    assert m["title"].tolist() == movies["title"].tolist()
    assert len(u) == len(users)
    stats = dataset_stats(preprocess(r, m, u))
    assert stats["interactions"] == len(ratings)


def test_pad_left() -> None:
    np.testing.assert_array_equal(pad_left(np.array([1, 2, 3]), 5), [0, 0, 1, 2, 3])
    np.testing.assert_array_equal(pad_left(np.array([1, 2, 3]), 2), [2, 3])
    np.testing.assert_array_equal(pad_left(np.array([], dtype=np.int64), 3), [0, 0, 0])


def test_bpr_negatives_are_unseen(split: SplitData) -> None:
    users, items = split.train_pairs()
    dataset = BPRDataset(users, items, n_items=split.n_items, seed=0)
    loader = make_batch_loader(dataset, batch_size=64, shuffle=True, seed=0)
    train_sets = [set(h) for h in split.history("valid")]
    n_seen = n_total = 0
    for batch in loader:
        assert batch["users"].shape == batch["pos_items"].shape == batch["neg_items"].shape
        for u, neg in zip(batch["users"].tolist(), batch["neg_items"].tolist(), strict=True):
            n_seen += neg in train_sets[u]
            n_total += 1
            assert 1 <= neg <= split.n_items
    assert n_total == len(dataset)
    assert n_seen / n_total < 0.01


def test_interaction_loader_covers_all_pairs(split: SplitData) -> None:
    users, items = split.train_pairs()
    loader = make_batch_loader(InteractionDataset(users, items), 100, shuffle=True, seed=1)
    seen = sorted(
        (u, i)
        for batch in loader
        for u, i in zip(batch["users"].tolist(), batch["items"].tolist(), strict=True)
    )
    assert seen == sorted(zip(users.tolist(), items.tolist(), strict=True))


def test_next_item_dataset_shifts_targets() -> None:
    dataset = NextItemDataset([np.array([4, 5, 6, 7]), np.array([9])], max_len=5)
    assert len(dataset) == 1  # sequences shorter than 2 cannot provide a target
    example = dataset[0]
    np.testing.assert_array_equal(example["inputs"], [0, 0, 4, 5, 6])
    np.testing.assert_array_equal(example["targets"], [0, 0, 5, 6, 7])


def test_build_split_sizes(processed: ProcessedData) -> None:
    split = build_split(processed)
    assert split.n_users == processed.users.shape[0]
    assert split.n_items == processed.items.shape[0]
