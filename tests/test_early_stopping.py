"""Tests for early stopping."""

import logging

import pytest

from early_stopping import EarlyStopping


def test_no_stop_when_improving():
    es = EarlyStopping(patience=3, min_delta=0.0)
    assert not es.step(1.0)
    assert not es.step(0.9)
    assert not es.step(0.8)


def test_stops_after_patience():
    es = EarlyStopping(patience=3, min_delta=0.0)
    es.step(1.0)  # sets best
    es.step(1.1)  # counter=1
    es.step(1.2)  # counter=2
    assert es.step(1.3)  # counter=3 -> stop


def test_min_delta():
    es = EarlyStopping(patience=2, min_delta=0.1)
    es.step(1.0)
    # improvement of 0.05 < min_delta, doesn't count
    assert not es.step(0.95)
    assert es.step(0.96)


def test_reset():
    es = EarlyStopping(patience=2)
    es.step(1.0)
    es.step(1.5)
    es.reset()
    assert es.counter == 0
    assert es.best_loss == 1.0


def test_reset_full():
    es = EarlyStopping(patience=2)
    es.step(1.0)
    es.step(1.5)
    es.reset(only_counter=False)
    assert es.counter == 0
    assert es.best_loss is None


def test_state_dict_roundtrip():
    es = EarlyStopping(patience=5, min_delta=0.01)
    es.step(0.5)
    es.step(0.6)
    state = es.state_dict()

    es2 = EarlyStopping(patience=5, min_delta=0.01)
    es2.load_state_dict(state)
    assert es2.counter == es.counter
    assert es2.best_loss == es.best_loss


def test_load_state_dict_patience_mismatch(caplog):
    es = EarlyStopping(patience=5, min_delta=0.01)
    es.step(0.5)
    state = es.state_dict()

    es2 = EarlyStopping(patience=3, min_delta=0.01)
    with caplog.at_level(logging.WARNING):
        es2.load_state_dict(state)
    assert "patience changed" in caplog.text
    assert es2.best_loss == 0.5


def test_first_step_never_stops():
    es = EarlyStopping(patience=1, min_delta=0.0)
    assert not es.step(999.0)


@pytest.mark.parametrize("patience", [1, 5, 10])
def test_exact_patience_boundary(patience):
    es = EarlyStopping(patience=patience, min_delta=0.0)
    es.step(0.5)  # set best
    for _ in range(patience - 1):
        assert not es.step(1.0)
    assert es.step(1.0)  # exactly at patience
