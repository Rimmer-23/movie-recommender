"""SASRec: self-attentive sequential recommendation (Kang & McAuley, 2018).

A causal Transformer encoder reads a user's chronological item sequence; the hidden state
at each position is scored against every item embedding (shared input/output embeddings)
to predict the next item. Training uses full softmax cross-entropy over the whole
catalogue at every non-padding position, which is both simple and markedly stronger than
the original paper's single sampled negative.

Sequences are **left-padded** to ``max_len`` so that the most recent item is always at the
last position; learned positional embeddings therefore encode recency.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader

from recsys.data import PAD_ITEM, NextItemDataset, SplitData, pad_left
from recsys.models.base import TorchRecommender


@dataclass
class SASRecParams:
    """Attributes:
    hidden_dim: Embedding / model width.
    max_len: Number of most recent items the model attends to.
    n_blocks: Number of Transformer blocks.
    n_heads: Attention heads per block.
    dropout: Dropout on embeddings, attention weights and residual branches.
    init_std: Std of the normal initialisation of embeddings.
    """

    hidden_dim: int = 64
    max_len: int = 200
    n_blocks: int = 2
    n_heads: int = 1
    dropout: float = 0.2
    init_std: float = 0.02


class CausalSelfAttention(nn.Module):
    """Multi-head self-attention driven by an explicit boolean mask."""

    def __init__(self, hidden_dim: int, n_heads: int, dropout: float) -> None:
        super().__init__()
        if hidden_dim % n_heads:
            raise ValueError("hidden_dim must be divisible by n_heads")
        self.n_heads = n_heads
        self.dropout = dropout
        self.qkv = nn.Linear(hidden_dim, 3 * hidden_dim)
        self.proj = nn.Linear(hidden_dim, hidden_dim)

    def forward(
        self, x: torch.Tensor, allowed: torch.Tensor, last_only: bool = False
    ) -> torch.Tensor:
        """Args:
        x: ``(batch, seq_len, hidden)`` input.
        allowed: ``(batch, 1, n_queries, seq_len)`` bool, ``True`` where query i may attend
            to key j.
        last_only: Use only the final position as a query (``n_queries = 1``); otherwise
            every position is a query (``n_queries = seq_len``).

        Returns:
            ``(batch, n_queries, hidden)``.
        """
        batch, _, hidden = x.shape
        head_dim = hidden // self.n_heads
        q, k, v = self.qkv(x).split(hidden, dim=-1)  # each (batch, seq_len, hidden)
        if last_only:
            q = q[:, -1:]  # (batch, 1, hidden)
        q, k, v = (
            t.view(batch, -1, self.n_heads, head_dim).transpose(1, 2) for t in (q, k, v)
        )  # q: (batch, heads, n_queries, head_dim); k, v: (batch, heads, seq_len, head_dim)
        out = F.scaled_dot_product_attention(
            q, k, v, attn_mask=allowed, dropout_p=self.dropout if self.training else 0.0
        )  # (batch, heads, n_queries, head_dim)
        out = out.transpose(1, 2).reshape(batch, -1, hidden)  # (batch, n_queries, hidden)
        return self.proj(out)


class TransformerBlock(nn.Module):
    """Pre-LayerNorm block: ``x + Attn(LN(x))`` then ``x + FFN(LN(x))``."""

    def __init__(self, hidden_dim: int, n_heads: int, dropout: float) -> None:
        super().__init__()
        self.attn_norm = nn.LayerNorm(hidden_dim)
        self.attn = CausalSelfAttention(hidden_dim, n_heads, dropout)
        self.ffn_norm = nn.LayerNorm(hidden_dim)
        self.ffn = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.dropout = nn.Dropout(dropout)

    def forward(
        self, x: torch.Tensor, allowed: torch.Tensor, last_only: bool = False
    ) -> torch.Tensor:
        """``(batch, seq_len, hidden)`` → ``(batch, seq_len, hidden)``.

        With ``last_only`` only the final position is computed — it still attends to the
        whole sequence — and the output is ``(batch, 1, hidden)``.
        """
        attended = self.attn(self.attn_norm(x), allowed, last_only)  # (batch, n_queries, hidden)
        if last_only:
            x = x[:, -1:]  # (batch, 1, hidden)
        x = x + self.dropout(attended)
        return x + self.dropout(self.ffn(self.ffn_norm(x)))


class SASRecNet(nn.Module):
    def __init__(self, n_items: int, params: SASRecParams) -> None:
        super().__init__()
        self.hidden_dim = params.hidden_dim
        self.item_emb = nn.Embedding(n_items + 1, params.hidden_dim, padding_idx=PAD_ITEM)
        self.pos_emb = nn.Embedding(params.max_len, params.hidden_dim)
        self.emb_dropout = nn.Dropout(params.dropout)
        self.blocks = nn.ModuleList(
            TransformerBlock(params.hidden_dim, params.n_heads, params.dropout)
            for _ in range(params.n_blocks)
        )
        self.final_norm = nn.LayerNorm(params.hidden_dim)
        nn.init.normal_(self.item_emb.weight, std=params.init_std)
        nn.init.normal_(self.pos_emb.weight, std=params.init_std)
        with torch.no_grad():
            self.item_emb.weight[PAD_ITEM].zero_()

    @staticmethod
    def attention_mask(is_pad: torch.Tensor) -> torch.Tensor:
        """Causal + key-padding mask, ``(batch, seq_len)`` → ``(batch, 1, seq_len, seq_len)``.

        Query ``i`` may attend to keys ``j <= i`` that are not padding. Every position may
        also attend to itself, so padding queries never have an empty row (which would
        produce NaNs); their outputs are discarded anyway.
        """
        seq_len = is_pad.shape[1]
        causal = torch.ones(seq_len, seq_len, dtype=torch.bool, device=is_pad.device).tril()
        allowed = causal[None] & ~is_pad[:, None, :]  # (batch, seq_len, seq_len)
        diagonal = torch.eye(seq_len, dtype=torch.bool, device=is_pad.device)
        return (allowed | diagonal[None])[:, None]  # (batch, 1, seq_len, seq_len)

    def forward(self, seqs: torch.Tensor, last_only: bool = False) -> torch.Tensor:
        """Encode left-padded sequences: ``(batch, seq_len)`` → ``(batch, seq_len, hidden)``.

        Args:
            seqs: Item ids, left-padded with 0. ``seq_len`` may be shorter than ``max_len``;
                positions are aligned to the right so the last item always gets the same
                positional embedding.
            last_only: Return only the final position, ``(batch, 1, hidden)``. That is all
                next-item inference needs, and it lets the last block skip the other
                ``seq_len - 1`` queries.
        """
        seq_len, max_len = seqs.shape[1], self.pos_emb.num_embeddings
        is_pad = seqs == PAD_ITEM  # (batch, seq_len)
        keep = (~is_pad).unsqueeze(-1)  # (batch, seq_len, 1)
        positions = torch.arange(max_len - seq_len, max_len, device=seqs.device)  # (seq_len,)
        x = self.item_emb(seqs) * math.sqrt(self.hidden_dim) + self.pos_emb(positions)
        x = self.emb_dropout(x) * keep  # (batch, seq_len, hidden)
        allowed = self.attention_mask(is_pad)  # (batch, 1, seq_len, seq_len)
        final = len(self.blocks) - 1
        for i, block in enumerate(self.blocks):
            if last_only and i == final:
                x = block(x, allowed[:, :, -1:], last_only=True) * keep[:, -1:]
            else:
                x = block(x, allowed) * keep
        return self.final_norm(x[:, -1:] if last_only else x)

    def logits(self, hidden: torch.Tensor) -> torch.Tensor:
        """Scores against all items: ``(..., hidden)`` → ``(..., n_items + 1)``."""
        return hidden @ self.item_emb.weight.T


class SASRecRecommender(TorchRecommender):
    """Causal Transformer over item sequences, trained with full-softmax next-item CE."""

    name = "sasrec"
    display_name = "SASRec"
    Params = SASRecParams
    params: SASRecParams

    def build_net(self) -> SASRecNet:
        return SASRecNet(self.n_items, self.params)

    def make_loader(self, data: SplitData) -> DataLoader:
        dataset = NextItemDataset(data.history("valid"), max_len=self.params.max_len)
        generator = torch.Generator().manual_seed(self.seed)
        return DataLoader(
            dataset, batch_size=self.train_config.batch_size, shuffle=True, generator=generator
        )

    def training_step(self, batch: dict[str, torch.Tensor]) -> torch.Tensor:
        hidden = self.net(batch["inputs"])  # (batch, max_len, hidden)
        is_target = batch["targets"] != PAD_ITEM  # (batch, max_len)
        logits = self.net.logits(hidden[is_target])  # (n_targets, n_items + 1)
        logits[:, PAD_ITEM] = float("-inf")
        return F.cross_entropy(logits, batch["targets"][is_target])

    def score(self, user_ids: np.ndarray) -> torch.Tensor:
        histories = [self.history[u] for u in user_ids]
        # Pad only to the longest history in the batch: attention is O(seq_len^2), and a
        # single user (the API case) is usually far shorter than max_len.
        longest = max((len(history) for history in histories), default=0)
        seq_len = min(max(longest, 1), self.params.max_len)
        seqs = np.stack([pad_left(history, seq_len) for history in histories])
        hidden = self.net(torch.from_numpy(seqs).to(self.device), last_only=True)
        return self.net.logits(hidden[:, 0])  # (batch, 1, hidden) → (batch, n_items + 1)
