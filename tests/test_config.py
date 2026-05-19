"""Tests for TrainingConfig validation."""

import pytest

from config import TrainingConfig


def test_default_config():
    config = TrainingConfig()
    assert config.n_gpus == 1
    assert config.n_nodes == 1
    assert config.node_rank == 0


def test_multi_gpu_single_node():
    config = TrainingConfig(n_gpus=4)
    assert config.n_gpus == 4
    assert config.n_nodes == 1
    assert config.node_rank == 0


def test_multi_node():
    config = TrainingConfig(n_gpus=4, n_nodes=2, node_rank=1)
    assert config.n_gpus == 4
    assert config.n_nodes == 2
    assert config.node_rank == 1


def test_n_gpus_zero_raises():
    with pytest.raises(AssertionError, match="n_gpus must be > 0"):
        TrainingConfig(n_gpus=0)


def test_n_nodes_zero_raises():
    with pytest.raises(AssertionError, match="n_nodes must be > 0"):
        TrainingConfig(n_nodes=0)


def test_node_rank_negative_raises():
    with pytest.raises(AssertionError, match="node_rank must be in"):
        TrainingConfig(node_rank=-1)


def test_node_rank_exceeds_n_nodes_raises():
    with pytest.raises(AssertionError, match="node_rank must be in"):
        TrainingConfig(n_nodes=2, node_rank=2)


def test_node_rank_boundary():
    config = TrainingConfig(n_nodes=3, node_rank=2)
    assert config.node_rank == 2


def test_checkpoint_every_not_multiple_of_log_every_raises():
    with pytest.raises(AssertionError, match="checkpoint_every must be a multiple"):
        TrainingConfig(log_every=300, checkpoint_every=1000)


def test_invalid_dropout_raises():
    with pytest.raises(AssertionError, match="dropout must be in"):
        TrainingConfig(dropout=1.0)
