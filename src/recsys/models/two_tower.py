"""Two-tower retrieval model trained with in-batch negatives and a sampled softmax.

* **User tower:** user-ID embedding + demographic features (gender, age, occupation) → MLP.
* **Item tower:** item-ID embedding + mean of genre embeddings → MLP.

Both towers output L2-normalised vectors; the score is their cosine similarity divided by a
temperature. For a batch of ``B`` positive pairs, every other item in the batch serves as a
negative for each user (``B × B`` logits, softmax cross-entropy on the diagonal).

Because popular items appear in batches more often, plain in-batch negatives over-penalise
them. The *logQ correction* (Yi et al., 2019) subtracts ``log P(item in batch)`` from the
training logits so that the model estimates the full-softmax scores instead.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader

from recsys.data import PAD_ITEM, InteractionDataset, SplitData, make_batch_loader
from recsys.models.base import TorchRecommender


@dataclass
class TwoTowerParams:
    """Attributes:
    id_dim: Size of user/item ID embeddings.
    feature_dim: Size of each side-feature embedding (demographics, genres).
    hidden_dims: Hidden layer sizes of each tower's MLP.
    output_dim: Size of the final (normalised) user and item vectors.
    dropout: Dropout inside the MLPs.
    temperature: Softmax temperature applied to cosine similarities.
    logq_correction: Subtract log item sampling probability from training logits.
    use_features: Use side features in addition to IDs.
    """

    id_dim: int = 64
    feature_dim: int = 16
    hidden_dims: list[int] = field(default_factory=lambda: [128])
    output_dim: int = 64
    dropout: float = 0.1
    temperature: float = 0.05
    logq_correction: bool = True
    use_features: bool = True


def in_batch_logits(
    user_vecs: torch.Tensor,
    item_vecs: torch.Tensor,
    users: torch.Tensor,
    items: torch.Tensor,
    temperature: float,
    log_q: torch.Tensor | None = None,
) -> torch.Tensor:
    """Logits of every user against every item in the batch, ``(batch, batch)``.

    Row ``i`` scores user ``users[i]`` against all batch items; the positive is on the
    diagonal. Off-diagonal entries that are not real negatives — the same item as the
    positive, or another positive of the same user — are set to ``-inf``.

    Args:
        user_vecs: ``(batch, dim)`` user tower outputs.
        item_vecs: ``(batch, dim)`` item tower outputs.
        users: ``(batch,)`` user ids.
        items: ``(batch,)`` item ids.
        temperature: Softmax temperature.
        log_q: Optional ``(batch,)`` log sampling probability of each item (logQ correction).
    """
    logits = user_vecs @ item_vecs.T / temperature  # (batch, batch)
    if log_q is not None:
        logits = logits - log_q[None, :]
    eye = torch.eye(len(users), dtype=torch.bool, device=users.device)
    false_negatives = (items[:, None] == items[None, :]) | (users[:, None] == users[None, :])
    return logits.masked_fill(false_negatives & ~eye, float("-inf"))


def mlp(in_dim: int, hidden_dims: list[int], out_dim: int, dropout: float) -> nn.Sequential:
    """``Linear → ReLU → Dropout`` stack followed by a linear projection to ``out_dim``."""
    layers: list[nn.Module] = []
    for hidden in hidden_dims:
        layers += [nn.Linear(in_dim, hidden), nn.ReLU(), nn.Dropout(dropout)]
        in_dim = hidden
    layers.append(nn.Linear(in_dim, out_dim))
    return nn.Sequential(*layers)


class UserTower(nn.Module):
    """User ID + categorical demographics → normalised user vector."""

    def __init__(
        self,
        n_users: int,
        feature_cardinalities: list[int],
        params: TwoTowerParams,
    ) -> None:
        super().__init__()
        self.id_emb = nn.Embedding(n_users, params.id_dim)
        self.feature_embs = nn.ModuleList(
            nn.Embedding(cardinality, params.feature_dim) for cardinality in feature_cardinalities
        )
        # Per-user feature ids, filled from data at fit time and stored in checkpoints.
        self.register_buffer("features", torch.zeros(n_users, len(feature_cardinalities)).long())
        in_dim = params.id_dim + len(feature_cardinalities) * params.feature_dim
        self.mlp = mlp(in_dim, params.hidden_dims, params.output_dim, params.dropout)

    def forward(self, users: torch.Tensor) -> torch.Tensor:
        """``(batch,)`` → ``(batch, output_dim)``."""
        features = self.features[users]  # (batch, n_features)
        parts = [self.id_emb(users)] + [
            emb(features[:, i]) for i, emb in enumerate(self.feature_embs)
        ]
        x = torch.cat(parts, dim=-1)  # (batch, id_dim + n_features * feature_dim)
        return F.normalize(self.mlp(x), dim=-1)


class ItemTower(nn.Module):
    """Item ID + bag of genres → normalised item vector."""

    def __init__(self, n_items: int, n_genres: int, max_genres: int, params: TwoTowerParams):
        super().__init__()
        self.use_genres = n_genres > 0
        self.id_emb = nn.Embedding(n_items + 1, params.id_dim, padding_idx=PAD_ITEM)
        in_dim = params.id_dim
        if self.use_genres:
            self.genre_emb = nn.EmbeddingBag(
                n_genres + 1, params.feature_dim, mode="mean", padding_idx=0
            )
            self.register_buffer("genres", torch.zeros(n_items + 1, max_genres).long())
            in_dim += params.feature_dim
        self.mlp = mlp(in_dim, params.hidden_dims, params.output_dim, params.dropout)

    def forward(self, items: torch.Tensor) -> torch.Tensor:
        """``(batch,)`` → ``(batch, output_dim)``."""
        x = self.id_emb(items)  # (batch, id_dim)
        if self.use_genres:
            genres = self.genre_emb(self.genres[items])  # (batch, feature_dim)
            x = torch.cat([x, genres], dim=-1)  # (batch, id_dim + feature_dim)
        return F.normalize(self.mlp(x), dim=-1)


class TwoTowerNet(nn.Module):
    def __init__(
        self,
        n_users: int,
        n_items: int,
        params: TwoTowerParams,
        feature_cardinalities: list[int],
        n_genres: int,
        max_genres: int,
    ) -> None:
        super().__init__()
        self.user_tower = UserTower(n_users, feature_cardinalities, params)
        self.item_tower = ItemTower(n_items, n_genres, max_genres, params)
        self.temperature = params.temperature

    def set_features(self, user_features: np.ndarray, item_genres: np.ndarray) -> None:
        """Copy side features into the towers' buffers."""
        self.user_tower.features.copy_(torch.from_numpy(user_features))
        if self.item_tower.use_genres:
            self.item_tower.genres.copy_(torch.from_numpy(item_genres))

    def score_all(self, users: torch.Tensor) -> torch.Tensor:
        """``(batch,)`` → ``(batch, n_items + 1)`` scores against the whole catalogue."""
        all_items = torch.arange(self.item_tower.id_emb.num_embeddings, device=users.device)
        item_vectors = self.item_tower(all_items)  # (n_items + 1, output_dim)
        return self.user_tower(users) @ item_vectors.T / self.temperature


class TwoTowerRecommender(TorchRecommender):
    """User/item towers trained with in-batch negatives (softmax loss + logQ correction)."""

    name = "two_tower"
    display_name = "Two-tower"
    Params = TwoTowerParams
    params: TwoTowerParams

    def get_net_kwargs(self, data: SplitData) -> dict[str, Any]:
        if not self.params.use_features:
            return {"feature_cardinalities": [], "n_genres": 0, "max_genres": 0}
        _, cardinalities = data.user_feature_matrix()
        genres, n_genres = data.item_genre_matrix()
        return {
            "feature_cardinalities": cardinalities,
            "n_genres": n_genres,
            "max_genres": genres.shape[1],
        }

    def build_net(self, **net_kwargs: Any) -> TwoTowerNet:
        return TwoTowerNet(self.n_users, self.n_items, self.params, **net_kwargs)

    def on_fit_start(self, data: SplitData) -> None:
        if self.params.use_features:
            user_features, _ = data.user_feature_matrix()
            genres, _ = data.item_genre_matrix()
            self.net.set_features(user_features, genres)
        # log P(item appears as a positive) for the logQ correction.
        _, items = data.train_pairs()
        counts = np.bincount(items, minlength=self.n_items + 1).astype(np.float64)
        probs = np.clip(counts / counts.sum(), 1e-12, None)
        self.log_q = torch.as_tensor(np.log(probs), dtype=torch.float32, device=self.device)

    def make_loader(self, data: SplitData) -> DataLoader:
        users, items = data.train_pairs()
        return make_batch_loader(
            InteractionDataset(users, items),
            batch_size=self.train_config.batch_size,
            shuffle=True,
            seed=self.seed,
        )

    def training_step(self, batch: dict[str, torch.Tensor]) -> torch.Tensor:
        users, items = batch["users"], batch["items"]  # (batch,), (batch,)
        logits = in_batch_logits(
            user_vecs=self.net.user_tower(users),  # (batch, dim)
            item_vecs=self.net.item_tower(items),  # (batch, dim)
            users=users,
            items=items,
            temperature=self.params.temperature,
            log_q=self.log_q[items] if self.params.logq_correction else None,
        )  # (batch, batch)
        labels = torch.arange(len(users), device=users.device)  # positives on the diagonal
        return F.cross_entropy(logits, labels)

    def score(self, user_ids: np.ndarray) -> torch.Tensor:
        users = torch.as_tensor(user_ids, dtype=torch.long, device=self.device)
        return self.net.score_all(users)  # (batch, n_items + 1)
