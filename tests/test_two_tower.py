"""Two-tower internals: tower shapes, normalisation and in-batch loss masking."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from recsys.config import ExperimentConfig, TrainConfig
from recsys.data import SplitData
from recsys.models import build_model, load_model
from recsys.models.two_tower import TwoTowerParams, TwoTowerRecommender, in_batch_logits


def make_model(split: SplitData, **params: object) -> TwoTowerRecommender:
    config = ExperimentConfig(
        model="two_tower",
        seed=0,
        params={"id_dim": 8, "feature_dim": 4, "hidden_dims": [16], "output_dim": 8, **params},
        train=TrainConfig(epochs=1, batch_size=16, device="cpu"),
    )
    model = build_model(config, split.n_users, split.n_items)
    assert isinstance(model, TwoTowerRecommender)
    model.net_kwargs = model.get_net_kwargs(split)
    model.net = model.build_net(**model.net_kwargs)
    model.on_fit_start(split)
    return model


def test_towers_output_unit_vectors(split: SplitData) -> None:
    model = make_model(split)
    users = torch.arange(5)
    items = torch.arange(1, 6)
    u, v = model.net.user_tower(users), model.net.item_tower(items)
    assert u.shape == v.shape == (5, 8)
    torch.testing.assert_close(u.norm(dim=-1), torch.ones(5))
    torch.testing.assert_close(v.norm(dim=-1), torch.ones(5))


def test_features_are_loaded_into_buffers(split: SplitData) -> None:
    model = make_model(split)
    genres, _ = split.item_genre_matrix()
    assert torch.equal(model.net.item_tower.genres, torch.from_numpy(genres))
    assert (model.net.user_tower.features > 0).all()


def test_id_only_variant(split: SplitData) -> None:
    model = make_model(split, use_features=False)
    assert model.net.score_all(torch.arange(3)).shape == (3, split.n_items + 1)


def test_in_batch_logits_mask_false_negatives() -> None:
    users = torch.tensor([0, 1, 2, 0])
    items = torch.tensor([5, 6, 5, 7])
    vecs = torch.nn.functional.normalize(torch.randn(4, 3), dim=-1)
    logits = in_batch_logits(vecs, vecs, users, items, temperature=0.1)
    masked = torch.isinf(logits)
    expected = torch.zeros(4, 4, dtype=torch.bool)
    expected[0, 2] = expected[2, 0] = True  # same item (5) for users 0 and 2
    expected[0, 3] = expected[3, 0] = True  # rows 0 and 3 are both user 0's positives
    assert torch.equal(masked, expected)
    assert torch.isfinite(logits.diagonal()).all()


def test_logq_correction_shifts_columns() -> None:
    users, items = torch.tensor([0, 1]), torch.tensor([3, 4])
    vecs = torch.eye(2)
    log_q = torch.tensor([-1.0, -5.0])
    plain = in_batch_logits(vecs, vecs, users, items, temperature=1.0)
    corrected = in_batch_logits(vecs, vecs, users, items, temperature=1.0, log_q=log_q)
    torch.testing.assert_close(corrected - plain, -log_q.expand(2, 2))


def test_training_step_is_finite(split: SplitData) -> None:
    model = make_model(split)
    batch = {"users": torch.tensor([0, 0, 1, 2]), "items": torch.tensor([5, 6, 5, 7])}
    loss = model.training_step(batch)
    assert torch.isfinite(loss) and loss.item() > 0


def test_loaded_model_caches_item_vectors(split: SplitData, tmp_path: Path) -> None:
    model = make_model(split)
    model.net.eval()
    users = np.arange(split.n_users)
    assert model._item_vectors is None  # weights may still change: nothing is cached
    path = tmp_path / "two_tower.pt"
    model.save(path)

    restored = load_model(path, device="cpu")
    assert isinstance(restored, TwoTowerRecommender)
    assert restored._item_vectors.shape == (split.n_items + 1, 8)
    with torch.no_grad():
        torch.testing.assert_close(restored.score(users), model.score(users))

    restored.train_config.epochs = 1
    restored.fit(split)  # training invalidates the cache
    assert restored._item_vectors is None
    with torch.no_grad():
        expected = restored.net.score_all(torch.as_tensor(users))
        torch.testing.assert_close(restored.score(users), expected)


def test_params_defaults_are_valid() -> None:
    params = TwoTowerParams()
    assert params.temperature > 0 and params.output_dim > 0
