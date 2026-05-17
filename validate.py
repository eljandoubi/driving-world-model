from math import ceil

import torch
import torch.nn as nn
from torch.nn.functional import mse_loss
from torch.utils.data import DataLoader
from tqdm import tqdm


@torch.inference_mode()
def validate_model(
    model: nn.Module, dataloader: DataLoader, device: torch.device, batch_size: int
):
    model.eval()
    cum_loss = 0.0
    total = int(ceil(len(dataloader)) / float(batch_size))
    pbar = tqdm(enumerate(dataloader, start=1), desc="validate", total=total)

    for i, batch in pbar:
        batch = batch.to(device)
        x_tp1_pred = model(batch["a_t"], batch["x_t"])
        loss = mse_loss(x_tp1_pred, batch["x_tp1"])
        cum_loss += loss.item()

        pbar.set_postfix(loss=cum_loss / i)

    return cum_loss / i
