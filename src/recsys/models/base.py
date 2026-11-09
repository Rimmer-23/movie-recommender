"""Common model interface shared by evaluation, training scripts and serving.

Every recommender implements ``fit``, ``recommend(user_ids, k, exclude_seen=True)``,
``save`` and ``load``. Scores are always produced over the full catalogue
(``n_items + 1`` columns, column 0 = padding), so evaluation is a full ranking.
"""

from __future__ import annotations

import copy
import logging
import math
import time
from abc import ABC, abstractmethod
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, ClassVar

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader

from recsys.config import TrainConfig, from_dict
from recsys.data import PAD_ITEM, SplitData
from recsys.utils import get_device, set_seed

logger = logging.getLogger(__name__)


@dataclass
class NoParams:
    """Placeholder for models without hyperparameters."""


class Recommender(ABC):
    """Base class for all recommenders.

    A model ranks items for users given a *history* (each user's previously seen items).
    The history is used to exclude already-seen items and, for sequential models, as the
    model input. It is set explicitly with :meth:`set_history` so the same fitted model can
    be evaluated on validation (train history), on test (train + validation history) or
    served on the full log.
    """

    name: ClassVar[str]  # registry key, also used for file names
    display_name: ClassVar[str]  # human-readable name for reports
    Params: ClassVar[type] = NoParams

    def __init__(
        self,
        n_users: int,
        n_items: int,
        params: Any | None = None,
        train_config: TrainConfig | None = None,
        seed: int = 42,
    ) -> None:
        self.n_users = n_users
        self.n_items = n_items
        self.params = params if params is not None else self.Params()
        self.train_config = train_config or TrainConfig()
        self.seed = seed
        self.device = get_device(self.train_config.device)
        self.training_log: list[dict[str, float]] = []
        self._history: list[np.ndarray] | None = None

    # ---------------------------------------------------------------- interface

    @abstractmethod
    def fit(self, data: SplitData, log_dir: Path | None = None) -> Recommender:
        """Train on ``data``'s train part (validation may be used for early stopping)."""

    @abstractmethod
    def score(self, user_ids: np.ndarray) -> torch.Tensor:
        """Unmasked scores for every item, shape ``(len(user_ids), n_items + 1)``."""

    @abstractmethod
    def get_state(self) -> dict[str, Any]:
        """Learned state to persist (tensors and plain Python values only)."""

    @abstractmethod
    def set_state(self, state: dict[str, Any]) -> None:
        """Restore state produced by :meth:`get_state`."""

    # ---------------------------------------------------------------- history

    def set_history(self, histories: Sequence[np.ndarray]) -> None:
        """Set each user's seen items (index = user id), oldest first."""
        if len(histories) != self.n_users:
            raise ValueError(f"Expected {self.n_users} histories, got {len(histories)}")
        self._history = list(histories)

    @property
    def history(self) -> list[np.ndarray]:
        if self._history is None:
            raise RuntimeError("History is not set; call fit() or set_history() first.")
        return self._history

    # ---------------------------------------------------------------- inference

    @torch.no_grad()
    def recommend(
        self, user_ids: Sequence[int] | np.ndarray, k: int = 10, exclude_seen: bool = True
    ) -> tuple[np.ndarray, np.ndarray]:
        """Top-``k`` items for each user.

        Args:
            user_ids: Internal user indices.
            k: Number of items to return (capped at ``n_items``).
            exclude_seen: Mask items in the user's history.

        Returns:
            ``(items, scores)``, both of shape ``(len(user_ids), k)``, best first.
            Items that could not be recommended (e.g. everything seen) get score ``-inf``.
        """
        user_ids = np.asarray(user_ids, dtype=np.int64)
        scores = self.score(user_ids).float().cpu()  # (batch, n_items + 1)
        scores[:, PAD_ITEM] = -math.inf
        if exclude_seen:
            rows, cols = self._seen_coordinates(user_ids)
            scores[rows, cols] = -math.inf
        top_scores, top_items = torch.topk(scores, min(k, self.n_items), dim=1)
        return top_items.numpy(), top_scores.numpy()

    def _seen_coordinates(self, user_ids: np.ndarray) -> tuple[torch.Tensor, torch.Tensor]:
        seen = [self.history[u] for u in user_ids]
        rows = np.repeat(np.arange(len(user_ids)), [len(s) for s in seen])
        cols = np.concatenate(seen) if seen else np.empty(0, dtype=np.int64)
        return torch.from_numpy(rows), torch.from_numpy(cols.astype(np.int64))

    # ---------------------------------------------------------------- persistence

    def save(self, path: Path) -> None:
        """Serialise model metadata, hyperparameters and learned state to ``path``."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        checkpoint = {
            "name": self.name,
            "n_users": self.n_users,
            "n_items": self.n_items,
            "seed": self.seed,
            "params": asdict(self.params),
            "train_config": asdict(self.train_config),
            "training_log": self.training_log,
            "state": self.get_state(),
        }
        torch.save(checkpoint, path)

    @classmethod
    def load(cls, path: Path, device: str | None = None) -> Recommender:
        """Restore a model saved with :meth:`save`."""
        checkpoint = torch.load(Path(path), map_location="cpu", weights_only=True)
        return cls.from_checkpoint(checkpoint, device=device)

    @classmethod
    def from_checkpoint(cls, checkpoint: dict[str, Any], device: str | None = None) -> Recommender:
        if checkpoint["name"] != cls.name:
            raise ValueError(f"Checkpoint is for '{checkpoint['name']}', not '{cls.name}'")
        train_config = from_dict(TrainConfig, checkpoint["train_config"])
        train_config.device = device
        model = cls(
            n_users=checkpoint["n_users"],
            n_items=checkpoint["n_items"],
            params=from_dict(cls.Params, checkpoint["params"]),
            train_config=train_config,
            seed=checkpoint["seed"],
        )
        model.training_log = checkpoint.get("training_log", [])
        model.set_state(checkpoint["state"])
        return model


class TorchRecommender(Recommender):
    """Recommender backed by an ``nn.Module`` and trained with a shared loop.

    Subclasses implement :meth:`build_net`, :meth:`make_loader`, :meth:`training_step` and
    :meth:`score`. The loop handles optimisation, periodic full-ranking validation,
    early stopping on ``train_config.monitor``, best-checkpoint restoring and TensorBoard
    logging.
    """

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.net: nn.Module | None = None
        self.net_kwargs: dict[str, Any] = {}
        self.best_epoch: int | None = None

    # ---------------------------------------------------------------- hooks

    def get_net_kwargs(self, data: SplitData) -> dict[str, Any]:
        """Data-dependent constructor arguments for :meth:`build_net` (saved in checkpoints)."""
        return {}

    @abstractmethod
    def build_net(self, **net_kwargs: Any) -> nn.Module:
        """Create the network from hyperparameters and ``net_kwargs``."""

    def on_fit_start(self, data: SplitData) -> None:
        """Called after the network is built, before training (e.g. to load features)."""

    @abstractmethod
    def make_loader(self, data: SplitData) -> DataLoader:
        """Training ``DataLoader`` yielding dicts of tensors."""

    @abstractmethod
    def training_step(self, batch: dict[str, torch.Tensor]) -> torch.Tensor:
        """Scalar loss for one batch (already on ``self.device``)."""

    # ---------------------------------------------------------------- training

    def fit(self, data: SplitData, log_dir: Path | None = None) -> TorchRecommender:
        from recsys.evaluate import evaluate  # local import: evaluate depends on models

        cfg = self.train_config
        set_seed(self.seed)
        self.net_kwargs = self.get_net_kwargs(data)
        self.net = self.build_net(**self.net_kwargs).to(self.device)
        self.on_fit_start(data)
        loader = self.make_loader(data)
        optimizer = torch.optim.Adam(
            self.net.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay
        )
        writer = _summary_writer(log_dir)

        best_score, best_state, rounds_without_improvement = -math.inf, None, 0
        self.training_log = []
        for epoch in range(1, cfg.epochs + 1):
            start = time.perf_counter()
            loss = self._train_epoch(loader, optimizer)
            record: dict[str, float] = {"epoch": epoch, "loss": loss}
            if epoch % cfg.eval_every == 0 or epoch == cfg.epochs:
                metrics = evaluate(self, data, "valid", batch_size=cfg.eval_batch_size)
                record.update({f"valid_{name}": value for name, value in metrics.items()})
                if metrics[cfg.monitor] > best_score:
                    best_score, self.best_epoch = metrics[cfg.monitor], epoch
                    best_state = copy.deepcopy(self.net.state_dict())
                    rounds_without_improvement = 0
                else:
                    rounds_without_improvement += 1
            record["seconds"] = time.perf_counter() - start
            self.training_log.append(record)
            _log_epoch(writer, record)
            if rounds_without_improvement >= cfg.patience:
                logger.info("Early stopping at epoch %d (best epoch %d)", epoch, self.best_epoch)
                break

        if best_state is not None:
            self.net.load_state_dict(best_state)
        logger.info("Best valid %s = %.4f at epoch %s", cfg.monitor, best_score, self.best_epoch)
        if writer is not None:
            writer.close()
        self.set_history(data.history("valid"))
        return self

    def _train_epoch(self, loader: DataLoader, optimizer: torch.optim.Optimizer) -> float:
        assert self.net is not None
        self.net.train()
        total, n_batches = 0.0, 0
        for batch in loader:
            batch = {key: value.to(self.device) for key, value in batch.items()}
            loss = self.training_step(batch)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            if self.train_config.grad_clip is not None:
                nn.utils.clip_grad_norm_(self.net.parameters(), self.train_config.grad_clip)
            optimizer.step()
            total += loss.item()
            n_batches += 1
        return total / max(n_batches, 1)

    # ---------------------------------------------------------------- inference

    def recommend(
        self, user_ids: Sequence[int] | np.ndarray, k: int = 10, exclude_seen: bool = True
    ) -> tuple[np.ndarray, np.ndarray]:
        if self.net is None:
            raise RuntimeError("Model is not trained; call fit() or load() first.")
        self.net.eval()
        return super().recommend(user_ids, k=k, exclude_seen=exclude_seen)

    # ---------------------------------------------------------------- persistence

    def get_state(self) -> dict[str, Any]:
        if self.net is None:
            raise RuntimeError("Nothing to save: model is not trained.")
        return {
            "net_kwargs": self.net_kwargs,
            "best_epoch": self.best_epoch,
            "state_dict": {k: v.detach().cpu() for k, v in self.net.state_dict().items()},
        }

    def set_state(self, state: dict[str, Any]) -> None:
        self.net_kwargs = state["net_kwargs"]
        self.best_epoch = state.get("best_epoch")
        self.net = self.build_net(**self.net_kwargs)
        self.net.load_state_dict(state["state_dict"])
        self.net.to(self.device).eval()


def _summary_writer(log_dir: Path | None) -> Any:
    if log_dir is None:
        return None
    from torch.utils.tensorboard import SummaryWriter  # optional dependency (train extra)

    return SummaryWriter(log_dir=str(log_dir))


def _log_epoch(writer: Any, record: dict[str, float]) -> None:
    epoch = int(record["epoch"])
    metrics = "  ".join(
        f"{key.removeprefix('valid_')}={value:.4f}"
        for key, value in record.items()
        if key.startswith("valid_")
    )
    logger.info(
        "epoch %3d  loss=%.4f  %s  (%.1fs)", epoch, record["loss"], metrics, record["seconds"]
    )
    if writer is None:
        return
    writer.add_scalar("train/loss", record["loss"], epoch)
    for key, value in record.items():
        if key.startswith("valid_"):
            writer.add_scalar(f"valid/{key.removeprefix('valid_')}", value, epoch)
    writer.flush()
