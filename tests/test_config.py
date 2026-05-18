"""Tests for TrainingConfig."""

import pytest

from config import TrainingConfig


def test_default_config():
    cfg = TrainingConfig()
    assert cfg.num_tokens == 8
    assert cfg.scheduler_t0 == 10
    assert cfg.scheduler_t_mult == 2
    assert cfg.learning_rate == 1e-4
    assert cfg.max_grad_norm == 1.0
    assert cfg.activation == "swiglu"
    assert cfg.image_size == 256


def test_custom_config():
    cfg = TrainingConfig(
        num_tokens=16,
        hidden_dim=512,
        learning_rate=3e-4,
        scheduler_t0=5,
        scheduler_t_mult=1,
        batch_size=4,
        num_workers=4,
    )
    assert cfg.num_tokens == 16
    assert cfg.hidden_dim == 512
    assert cfg.learning_rate == 3e-4
    assert cfg.scheduler_t0 == 5


def test_invalid_learning_rate():
    with pytest.raises(AssertionError):
        TrainingConfig(learning_rate=-1)


def test_invalid_batch_size():
    with pytest.raises(AssertionError):
        TrainingConfig(batch_size=0)


def test_invalid_dropout():
    with pytest.raises(AssertionError):
        TrainingConfig(dropout=1.0)


def test_negative_dropout():
    with pytest.raises(AssertionError):
        TrainingConfig(dropout=-0.1)


def test_invalid_num_tokens():
    with pytest.raises(AssertionError):
        TrainingConfig(num_tokens=0)


def test_invalid_hidden_dim():
    with pytest.raises(AssertionError):
        TrainingConfig(hidden_dim=-1)


def test_invalid_epochs():
    with pytest.raises(AssertionError):
        TrainingConfig(epochs=0)


def test_invalid_max_grad_norm():
    with pytest.raises(AssertionError):
        TrainingConfig(max_grad_norm=0)


def test_invalid_patience():
    with pytest.raises(AssertionError):
        TrainingConfig(patience=0)


def test_negative_min_delta():
    with pytest.raises(AssertionError):
        TrainingConfig(min_delta=-1.0)


def test_checkpoint_every_not_multiple_of_log_every():
    with pytest.raises(AssertionError):
        TrainingConfig(log_every=1000, checkpoint_every=1500)


def test_num_workers_less_than_batch_size():
    with pytest.raises(AssertionError):
        TrainingConfig(batch_size=16, num_workers=4)


def test_invalid_resume_path():
    with pytest.raises(AssertionError):
        TrainingConfig(resume="/nonexistent/path.pt")


def test_set_id_and_update_paths(tmp_path):
    cfg = TrainingConfig(runs_dir=str(tmp_path))
    cfg.set_id("test_run_123")
    cfg.update_paths()
    assert cfg.run_id == "test_run_123"
    assert cfg.checkpoint_dir.exists()
    assert cfg.plot_dir.exists()
    assert "test_run_123" in str(cfg.checkpoint_dir)
