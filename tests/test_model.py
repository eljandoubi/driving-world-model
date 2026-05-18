"""Tests for model components."""

import pytest
import torch

from model import ActionEmbedder


def test_action_embedder_shape():
    embed = ActionEmbedder(action_dim=3, embed_dim=64, num_tokens=4, hidden_dim=128)
    a = torch.randn(2, 3)
    out = embed(a)
    assert out.shape == (2, 4, 64)


def test_action_embedder_different_tokens():
    embed = ActionEmbedder(action_dim=3, embed_dim=32, num_tokens=8, hidden_dim=64)
    a = torch.randn(1, 3)
    out = embed(a)
    assert out.shape == (1, 8, 32)


@pytest.mark.parametrize("num_tokens", [1, 4, 16])
def test_action_embedder_various_tokens(num_tokens):
    embed = ActionEmbedder(
        action_dim=3, embed_dim=64, num_tokens=num_tokens, hidden_dim=128
    )
    out = embed(torch.randn(4, 3))
    assert out.shape == (4, num_tokens, 64)


@pytest.mark.parametrize("batch_size", [1, 2, 8])
def test_action_embedder_batch_sizes(batch_size):
    embed = ActionEmbedder(action_dim=3, embed_dim=32, num_tokens=4, hidden_dim=64)
    out = embed(torch.randn(batch_size, 3))
    assert out.shape[0] == batch_size


def test_action_embedder_gradient_flow():
    embed = ActionEmbedder(action_dim=3, embed_dim=64, num_tokens=4, hidden_dim=128)
    a = torch.randn(2, 3, requires_grad=True)
    out = embed(a)
    out.sum().backward()
    assert a.grad is not None
    assert a.grad.shape == (2, 3)


def test_action_embedder_deterministic():
    embed = ActionEmbedder(action_dim=3, embed_dim=64, num_tokens=4, hidden_dim=128)
    embed.eval()
    a = torch.randn(2, 3)
    out1 = embed(a)
    out2 = embed(a)
    assert torch.allclose(out1, out2)
