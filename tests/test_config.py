"""Tests for TrainingConfig validation."""

import pytest

from config import TrainingConfig


def test_n_gpus_zero_raises():
    with pytest.raises(AssertionError, match="n_gpus must be > 0"):
        TrainingConfig(n_gpus=0)


def test_n_nodes_zero_raises():
    with pytest.raises(AssertionError, match="n_nodes must be > 0"):
        TrainingConfig(n_nodes=0)


def test_checkpoint_every_not_multiple_of_log_every_raises():
    with pytest.raises(AssertionError, match="checkpoint_every must be a multiple"):
        TrainingConfig(log_every=300, checkpoint_every=1000)


def test_invalid_dropout_raises():
    with pytest.raises(AssertionError, match="dropout must be in"):
        TrainingConfig(dropout=1.0)
