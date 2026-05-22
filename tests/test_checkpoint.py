"""Tests for checkpoint save/load."""

import tempfile
from pathlib import Path
from unittest.mock import MagicMock

import torch
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts

from checkpoint import load_checkpoint, save_checkpoint
from early_stopping import EarlyStopping


def _make_model():
    return torch.nn.Linear(4, 4)


def _make_dataset_mock(rank=0, world_size=1, batch_size=2):
    ds = MagicMock()
    ds.state_dict.return_value = {
        "counter": 42,
        "len": 100,
        "rank": rank,
        "world_size": world_size,
        "batch_size": batch_size,
    }
    return ds


def test_save_load_checkpoint():
    model = _make_model()
    optimizer = AdamW(model.parameters(), lr=1e-3)
    scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=10)
    es = EarlyStopping(patience=5, min_delta=0.0)
    es.step(0.5)
    dataset = _make_dataset_mock()

    with tempfile.TemporaryDirectory() as tmpdir:
        path = Path(tmpdir) / "ckpt.pt"
        save_checkpoint(
            path,
            model,
            optimizer,
            scheduler,
            es,
            dataset=dataset,
            epoch=3,
            best_val_loss=0.42,
        )

        model2 = _make_model()
        optimizer2 = AdamW(model2.parameters(), lr=1e-3)
        scheduler2 = CosineAnnealingWarmRestarts(optimizer2, T_0=10)
        es2 = EarlyStopping(patience=5, min_delta=0.0)
        dataset2 = _make_dataset_mock()

        epoch, best_val = load_checkpoint(
            path, model2, optimizer2, scheduler2, torch.device("cpu"), es2, dataset2
        )

        assert epoch == 3
        assert best_val == 0.42
        assert es2.best_loss == 0.5
        dataset2.load_state_dict.assert_called_once()


def test_save_load_no_scheduler():
    model = _make_model()
    optimizer = AdamW(model.parameters(), lr=1e-3)
    es = EarlyStopping(patience=5, min_delta=0.0)
    dataset = _make_dataset_mock()

    with tempfile.TemporaryDirectory() as tmpdir:
        path = Path(tmpdir) / "ckpt.pt"
        save_checkpoint(
            path, model, optimizer, None, es, dataset=dataset, epoch=1, best_val_loss=0.3
        )

        model2 = _make_model()
        optimizer2 = AdamW(model2.parameters(), lr=1e-3)
        es2 = EarlyStopping(patience=5, min_delta=0.0)
        dataset2 = _make_dataset_mock()

        epoch, best_val = load_checkpoint(
            path, model2, optimizer2, None, torch.device("cpu"), es2, dataset2
        )
        assert epoch == 1
        assert best_val == 0.3


def test_checkpoint_preserves_model_weights():
    model = _make_model()
    with torch.no_grad():
        model.weight.fill_(42.0)
    optimizer = AdamW(model.parameters(), lr=1e-3)
    es = EarlyStopping(patience=5, min_delta=0.0)
    dataset = _make_dataset_mock()

    with tempfile.TemporaryDirectory() as tmpdir:
        path = Path(tmpdir) / "ckpt.pt"
        save_checkpoint(
            path, model, optimizer, None, es, dataset=dataset, epoch=0, best_val_loss=1.0
        )

        model2 = _make_model()
        optimizer2 = AdamW(model2.parameters(), lr=1e-3)
        es2 = EarlyStopping(patience=5, min_delta=0.0)
        dataset2 = _make_dataset_mock()
        load_checkpoint(path, model2, optimizer2, None, torch.device("cpu"), es2, dataset2)

        assert torch.allclose(model2.weight, torch.full_like(model2.weight, 42.0))


def test_checkpoint_creates_parent_dirs():
    model = _make_model()
    optimizer = AdamW(model.parameters(), lr=1e-3)
    es = EarlyStopping(patience=5, min_delta=0.0)
    dataset = _make_dataset_mock()

    with tempfile.TemporaryDirectory() as tmpdir:
        path = Path(tmpdir) / "nested" / "deep" / "ckpt.pt"
        save_checkpoint(
            path, model, optimizer, None, es, dataset=dataset, epoch=0, best_val_loss=0.0
        )
        assert path.exists()
