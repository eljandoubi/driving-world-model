"""Checkpoint save/load utilities."""

from pathlib import Path

import torch

from early_stopping import EarlyStopping


def save_checkpoint(
    path: str | Path,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler | None,
    early_stopping: EarlyStopping,
    epoch: int,
    iteration: int,
    best_val_loss: float,
) -> None:
    """Save training checkpoint."""
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    data = {
        "epoch": epoch,
        "iteration": iteration,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict()
        if scheduler is not None
        else None,
        "early_stopping_state_dict": early_stopping.state_dict(),
        "best_val_loss": best_val_loss,
    }
    torch.save(data, path)


def load_checkpoint(
    path: str | Path,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    device: torch.device,
    early_stopping: EarlyStopping,
) -> tuple[int, int, float]:
    """Load training checkpoint. Returns start_epoch, start_iteration, and best_val_loss."""
    ckpt = torch.load(path, map_location=device, weights_only=True)
    model.load_state_dict(ckpt["model_state_dict"])
    optimizer.load_state_dict(ckpt["optimizer_state_dict"])
    scheduler.load_state_dict(ckpt["scheduler_state_dict"])
    early_stopping.load_state_dict(ckpt["early_stopping_state_dict"])
    return (
        ckpt["epoch"],
        ckpt["iteration"],
        ckpt["best_val_loss"],
    )
