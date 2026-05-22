"""Tests for validate_model."""

import math

import torch
import torch.nn as nn
from torch.nn.functional import mse_loss

from dataset import TensorDict
from validate import validate_model


class _DummyModel(nn.Module):
    def forward(self, a_t, x_t, t=None):
        return torch.zeros_like(x_t)


def _make_batches(n: int, channels: int = 4, size: int = 4):
    batches = []
    for _ in range(n):
        td = TensorDict(
            {
                "a_t": torch.randn(2, 3),
                "x_t": torch.randn(2, channels, size, size),
                "x_tp1": torch.randn(2, channels, size, size),
            }
        )
        batches.append(td)
    return batches


class _FakeDataloader:
    def __init__(self, batches):
        self._batches = batches

    def __len__(self):
        return len(self._batches)

    def __iter__(self):
        yield from self._batches


def test_validate_single_gpu():
    model = _DummyModel()
    batches = _make_batches(5)
    dl = _FakeDataloader(batches)
    loss = validate_model(model, dl, torch.device("cpu"), mse_loss)
    assert isinstance(loss, float)
    assert loss >= 0


def test_validate_empty_dataloader():
    model = _DummyModel()
    dl = _FakeDataloader([])
    loss = validate_model(model, dl, torch.device("cpu"), mse_loss)
    assert math.isinf(loss)


def test_validate_deterministic():
    model = _DummyModel()
    batches = _make_batches(3)
    dl1 = _FakeDataloader(batches)
    dl2 = _FakeDataloader(batches)
    loss1 = validate_model(model, dl1, torch.device("cpu"), mse_loss)
    loss2 = validate_model(model, dl2, torch.device("cpu"), mse_loss)
    assert loss1 == loss2


def test_validate_model_set_to_eval():
    model = _DummyModel()
    model.train()
    batches = _make_batches(2)
    dl = _FakeDataloader(batches)
    validate_model(model, dl, torch.device("cpu"), mse_loss)
    assert not model.training
