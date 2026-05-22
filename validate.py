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
    loss_fn: Callable[[torch.Tensor,torch.Tensor],torch.Tensor],
    world_size: int = 1,
    is_main: bool = True,
):
    
    if world_size > 1:
        dist.barrier()

    model.eval()
    cum_loss = torch.zeros(2,device=device,dtype=torch.float32)

    pbar = tqdm(
        enumerate(dataloader, start=1),
        desc="validate",
        total=len(dataloader),
        disable= not is_main
    )

    for i, batch in pbar:
        batch = batch.to(device, non_blocking=True)
        pred_delta = model(batch["a_t"], batch["x_t"])
        target_delta = batch["x_tp1"] - batch["x_t"]
        loss = loss_fn(pred_delta, target_delta)
        cum_loss[0] += loss.detach()


    if world_size > 1:
        cum_loss[1]=i
        dist.all_reduce(cum_loss, op=dist.ReduceOp.SUM)


    return (cum_loss[0] / cum_loss[1]).item()
