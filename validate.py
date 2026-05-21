from typing import Callable

import torch
import torch.distributed as dist
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm


@torch.inference_mode()
def validate_model(
    model: nn.Module,
    dataloader: DataLoader,
    device: torch.device,
    loss_fn: Callable,
    world_size: int = 1,
    disable_tqdm: bool = False,
):
    model.eval()
    cum_loss = 0.0
    pbar = tqdm(
        enumerate(dataloader, start=1),
        desc="validate",
        total=len(dataloader),
        disable=disable_tqdm,
    )

    for i, batch in pbar:
        batch = batch.to(device, non_blocking=True)
        pred_delta = model(batch["a_t"], batch["x_t"])
        target_delta = batch["x_tp1"] - batch["x_t"]
        loss = loss_fn(pred_delta, target_delta)
        cum_loss += loss.item()

        pbar.set_postfix(loss=cum_loss / i)

    if world_size > 1:
        loss_tensor = torch.tensor([cum_loss, float(i)], device=device)
        dist.all_reduce(loss_tensor, op=dist.ReduceOp.SUM)
        return (loss_tensor[0] / loss_tensor[1]).item()

    return cum_loss / i
