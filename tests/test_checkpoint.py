"""Tests for checkpoint save/load."""

import tempfile
from pathlib import Path

import torch
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts

from checkpoint import load_checkpoint, save_checkpoint
from early_stopping import EarlyStopping


def _make_model():
    return torch.nn.Linear(4, 4)


def test_save_load_checkpoint():
    model = _make_model()
    optimizer = AdamW(model.parameters(), lr=1e-3)
    scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=10)
    es = EarlyStopping(patience=5, min_delta=0.0)
    es.step(0.5)

    with tempfile.TemporaryDirectory() as tmpdir:
        path = Path(tmpdir) / "ckpt.pt"
        save_checkpoint(
            path,
            model,
            optimizer,
            scheduler,
            es,
            epoch=3,
            iteration=100,
            best_val_loss=0.42,
        )

        model2 = _make_model()
        optimizer2 = AdamW(model2.parameters(), lr=1e-3)
        scheduler2 = CosineAnnealingWarmRestarts(optimizer2, T_0=10)
        es2 = EarlyStopping(patience=5, min_delta=0.0)

        epoch, iteration, best_val = load_checkpoint(
            path, model2, optimizer2, scheduler2, torch.device("cpu"), es2
        )

        assert epoch == 3
        assert iteration == 100
        assert best_val == 0.42
        assert es2.best_loss == 0.5


def test_save_load_no_scheduler():
    model = _make_model()
    optimizer = AdamW(model.parameters(), lr=1e-3)
    es = EarlyStopping(patience=5, min_delta=0.0)

    with tempfile.TemporaryDirectory() as tmpdir:
        path = Path(tmpdir) / "ckpt.pt"
        save_checkpoint(
            path, model, optimizer, None, es, epoch=1, iteration=50, best_val_loss=0.3
        )

        model2 = _make_model()
        optimizer2 = AdamW(model2.parameters(), lr=1e-3)
        es2 = EarlyStopping(patience=5, min_delta=0.0)

        epoch, iteration, best_val = load_checkpoint(
            path, model2, optimizer2, None, torch.device("cpu"), es2
        )
        assert epoch == 1
        assert iteration == 50


def test_checkpoint_preserves_model_weights():
    model = _make_model()
    # Set specific weights
    with torch.no_grad():
        model.weight.fill_(42.0)
    optimizer = AdamW(model.parameters(), lr=1e-3)
    es = EarlyStopping(patience=5, min_delta=0.0)

    with tempfile.TemporaryDirectory() as tmpdir:
        path = Path(tmpdir) / "ckpt.pt"
        save_checkpoint(
            path, model, optimizer, None, es, epoch=0, iteration=0, best_val_loss=1.0
        )

        model2 = _make_model()
        optimizer2 = AdamW(model2.parameters(), lr=1e-3)
        es2 = EarlyStopping(patience=5, min_delta=0.0)
        load_checkpoint(path, model2, optimizer2, None, torch.device("cpu"), es2)

        assert torch.allclose(model2.weight, torch.full_like(model2.weight, 42.0))


def test_checkpoint_creates_parent_dirs():
    model = _make_model()
    optimizer = AdamW(model.parameters(), lr=1e-3)
    es = EarlyStopping(patience=5, min_delta=0.0)

    with tempfile.TemporaryDirectory() as tmpdir:
        path = Path(tmpdir) / "nested" / "deep" / "ckpt.pt"
        save_checkpoint(
            path, model, optimizer, None, es, epoch=0, iteration=0, best_val_loss=0.0
        )
        assert path.exists()
