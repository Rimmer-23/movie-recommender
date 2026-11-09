"""Matrix Factorization trained with Bayesian Personalized Ranking (BPR).

Score: ``s(u, i) = <p_u, q_i>``. For every observed ``(u, i)`` a negative ``j`` is sampled
uniformly from items ``u`` has not seen, and the model maximises
``log σ(s(u, i) − s(u, j))`` (Rendle et al., 2009).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader

from recsys.data import PAD_ITEM, BPRDataset, SplitData, make_batch_loader
from recsys.models.base import TorchRecommender


@dataclass
class MFParams:
    """Attributes:
    embedding_dim: Size of user and item vectors.
    l2_reg: L2 penalty on the embeddings used in each batch (per example).
    init_std: Std of the normal initialisation.
    """

    embedding_dim: int = 64
    l2_reg: float = 1e-5
    init_std: float = 0.01


class MFNet(nn.Module):
    """User and item embedding tables with dot-product scoring."""

    def __init__(self, n_users: int, n_items: int, embedding_dim: int, init_std: float) -> None:
        super().__init__()
        self.user_emb = nn.Embedding(n_users, embedding_dim)
        self.item_emb = nn.Embedding(n_items + 1, embedding_dim, padding_idx=PAD_ITEM)
        nn.init.normal_(self.user_emb.weight, std=init_std)
        nn.init.normal_(self.item_emb.weight, std=init_std)
        with torch.no_grad():
            self.item_emb.weight[PAD_ITEM].zero_()

    def forward(self, users: torch.Tensor, items: torch.Tensor) -> torch.Tensor:
        """Scores for aligned ``(users, items)`` pairs: ``(batch,)`` → ``(batch,)``."""
        return (self.user_emb(users) * self.item_emb(items)).sum(dim=-1)

    def score_all(self, users: torch.Tensor) -> torch.Tensor:
        """Scores against the whole catalogue: ``(batch,)`` → ``(batch, n_items + 1)``."""
        return self.user_emb(users) @ self.item_emb.weight.T


class MFRecommender(TorchRecommender):
    """BPR-MF with uniform negative sampling."""

    name = "mf"
    display_name = "MF (BPR)"
    Params = MFParams
    params: MFParams

    def build_net(self) -> MFNet:
        p = self.params
        return MFNet(self.n_users, self.n_items, p.embedding_dim, p.init_std)

    def make_loader(self, data: SplitData) -> DataLoader:
        users, items = data.train_pairs()
        dataset = BPRDataset(users, items, n_items=self.n_items, seed=self.seed)
        return make_batch_loader(
            dataset, batch_size=self.train_config.batch_size, shuffle=True, seed=self.seed
        )

    def training_step(self, batch: dict[str, torch.Tensor]) -> torch.Tensor:
        user = self.net.user_emb(batch["users"])  # (batch, dim)
        pos = self.net.item_emb(batch["pos_items"])  # (batch, dim)
        neg = self.net.item_emb(batch["neg_items"])  # (batch, dim)
        pos_scores = (user * pos).sum(dim=-1)  # (batch,)
        neg_scores = (user * neg).sum(dim=-1)  # (batch,)
        bpr = -F.logsigmoid(pos_scores - neg_scores).mean()
        l2 = (user.pow(2).sum() + pos.pow(2).sum() + neg.pow(2).sum()) / len(user)
        return bpr + self.params.l2_reg * l2

    def score(self, user_ids: np.ndarray) -> torch.Tensor:
        users = torch.as_tensor(user_ids, dtype=torch.long, device=self.device)
        return self.net.score_all(users)  # (batch, n_items + 1)
