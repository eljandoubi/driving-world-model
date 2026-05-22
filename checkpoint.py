"""Checkpoint save/load utilities."""

from pathlib import Path

import torch
import torch.distributed as dist

from dataset import StreamDataset
from early_stopping import EarlyStopping


def save_checkpoint(
    path: str | Path,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler | None,
    early_stopping: EarlyStopping,
    dataset: StreamDataset,
    epoch: int,
    best_val_loss: float,
    rank: int = 0,
    world_size: int = 1,
) -> None:
    """Save training checkpoint."""
    if world_size > 1:
        dist.barrier()  # Ensure all processes have finished the epoch before saving
        local_dataset_state = dataset.state_dict()
        gathered_dataset_states = [{} for _ in range(world_size)]
        dist.all_gather_object(gathered_dataset_states, local_dataset_state)
    else:
        gathered_dataset_states = [dataset.state_dict()]

    if rank == 0:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        data = {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict()
            if scheduler is not None
            else None,
            "early_stopping_state_dict": early_stopping.state_dict(),
            "best_val_loss": best_val_loss,
            "dataset_state_dict": gathered_dataset_states,
        }
        torch.save(data, path)


def load_checkpoint(
    path: str | Path,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler | None,
    device: torch.device,
    early_stopping: EarlyStopping,
    dataset: StreamDataset,
    rank: int = 0,
) -> tuple[int, float]:
    """Load training checkpoint. Returns start_epoch and best_val_loss."""
    ckpt = torch.load(path, map_location=device, weights_only=True)
    model.load_state_dict(ckpt["model_state_dict"])
    optimizer.load_state_dict(ckpt["optimizer_state_dict"])
    if scheduler is not None and ckpt["scheduler_state_dict"] is not None:
        scheduler.load_state_dict(ckpt["scheduler_state_dict"])
    early_stopping.load_state_dict(ckpt["early_stopping_state_dict"])
    dataset.load_state_dict(ckpt["dataset_state_dict"][rank])
    return (
        ckpt["epoch"],
        ckpt["best_val_loss"],
    )
