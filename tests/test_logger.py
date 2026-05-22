"""Tests for logging setup."""

import logging

from logger import TqdmLoggingHandler, setup_logging


def test_setup_logging_returns_logger():
    logger = setup_logging(rank=0)
    assert isinstance(logger, logging.Logger)


def test_rank_zero_has_handler():
    logger = setup_logging(rank=0)
    root = logging.getLogger()
    assert any(isinstance(h, TqdmLoggingHandler) for h in root.handlers)


def test_non_rank_zero_no_handler():
    setup_logging(rank=1)
    root = logging.getLogger()
    assert not any(isinstance(h, TqdmLoggingHandler) for h in root.handlers)


def test_rank_injected_into_records():
    setup_logging(rank=7)
    factory = logging.getLogRecordFactory()
    record = factory("test", logging.INFO, "", 0, "msg", (), None)
    assert hasattr(record, "rank")
    assert record.rank == 7


def test_noisy_libs_silenced():
    setup_logging(rank=0)
    for name in ["httpx", "urllib3", "huggingface_hub"]:
        assert logging.getLogger(name).level == logging.ERROR
