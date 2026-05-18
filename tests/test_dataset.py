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
