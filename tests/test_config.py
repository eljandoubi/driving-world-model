"""Tests for TrainingConfig."""

import pytest

from config import TrainingConfig


def test_default_config():
    cfg = TrainingConfig()
    assert cfg.num_tokens == 8
    assert cfg.scheduler_t0 == 10
    assert cfg.scheduler_t_mult == 2


def test_invalid_learning_rate():
    with pytest.raises(AssertionError):
        TrainingConfig(learning_rate=-1)


def test_invalid_batch_size():
    with pytest.raises(AssertionError):
        TrainingConfig(batch_size=0)


def test_invalid_dropout():
    with pytest.raises(AssertionError):
        TrainingConfig(dropout=1.0)


def test_checkpoint_every_not_multiple_of_log_every():
    with pytest.raises(AssertionError):
        TrainingConfig(log_every=1000, checkpoint_every=1500)
