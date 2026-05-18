"""Tests for model components."""

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
