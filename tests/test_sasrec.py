"""SASRec internals: attention mask, causality and output shapes."""

from __future__ import annotations

import torch

from recsys.models.sasrec import SASRecNet, SASRecParams

N_ITEMS = 50


def make_net(max_len: int = 6) -> SASRecNet:
    torch.manual_seed(0)
    net = SASRecNet(N_ITEMS, SASRecParams(hidden_dim=16, max_len=max_len, n_heads=2))
    return net.eval()


def test_attention_mask_is_causal_and_ignores_padding() -> None:
    is_pad = torch.tensor([[True, True, False, False]])
    allowed = SASRecNet.attention_mask(is_pad)[0, 0]  # (seq_len, seq_len)
    expected = torch.tensor(
        [
            [True, False, False, False],  # padding query: only itself
            [False, True, False, False],  # padding query: only itself
            [False, False, True, False],  # first real item sees itself
            [False, False, True, True],  # second real item sees both real items
        ]
    )
    assert torch.equal(allowed, expected)


def test_output_shapes() -> None:
    net = make_net()
    seqs = torch.tensor([[0, 0, 3, 4, 5, 6], [1, 2, 3, 4, 5, 6]])
    hidden = net(seqs)
    assert hidden.shape == (2, 6, 16)
    assert net.logits(hidden).shape == (2, 6, N_ITEMS + 1)
    assert torch.isfinite(hidden).all()


def test_future_items_do_not_affect_past_positions() -> None:
    net = make_net()
    a = torch.tensor([[0, 0, 3, 4, 5, 6]])
    b = torch.tensor([[0, 0, 3, 4, 9, 8]])  # differs only after position 3
    with torch.no_grad():
        ha, hb = net(a), net(b)
    torch.testing.assert_close(ha[:, :4], hb[:, :4])
    assert not torch.allclose(ha[:, 4:], hb[:, 4:])


def test_padding_items_are_ignored() -> None:
    net = make_net()
    # Same real items; the padding id is 0 so no other item can sneak into padded slots.
    seqs = torch.tensor([[0, 0, 0, 7, 8, 9]])
    with torch.no_grad():
        hidden = net(seqs)
        # Changing the padding embedding row must not change real positions.
        net.item_emb.weight[0].fill_(3.0)
        hidden_after = net(seqs)
    torch.testing.assert_close(hidden[:, 3:], hidden_after[:, 3:])
