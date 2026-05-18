"""Tests for dataset utilities."""

import torch

from dataset import TensorDict


def test_tensor_dict_to_device():
    td = TensorDict({"a": torch.zeros(2), "b": torch.ones(3)})
    td.to(torch.device("cpu"))
    assert td["a"].device == torch.device("cpu")


def test_tensor_dict_non_tensor():
    td = TensorDict({"a": torch.zeros(2), "label": "hello"})
    td.to(torch.device("cpu"))  # should not crash on non-tensor values
    assert td["label"] == "hello"


def test_tensor_dict_preserves_values():
    td = TensorDict({"x": torch.tensor([1.0, 2.0, 3.0])})
    td.to(torch.device("cpu"))
    assert torch.allclose(td["x"], torch.tensor([1.0, 2.0, 3.0]))


def test_tensor_dict_is_dict():
    td = TensorDict({"a": 1, "b": 2})
    assert isinstance(td, dict)
    assert len(td) == 2
    assert "a" in td


def test_tensor_dict_returns_self():
    td = TensorDict({"a": torch.zeros(2)})
    result = td.to(torch.device("cpu"))
    assert result is td
