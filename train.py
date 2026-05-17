import wandb
from torch.nn.functional import mse_loss
from torch.nn.utils import clip_grad_norm_
from torch.optim import AdamW
from torch.utils.data import DataLoader

from config import TrainingConfig
from dataset import StreamDataset
from model import WorldModel


def main(config: TrainingConfig) -> None:
    # --- WANDB INIT ---
    run = wandb.init(
        project="driving-world-model",
        config=vars(config),
        id=config.run_id,
        resume="allow" if config.run_id else None,
        mode="disabled",  # for debug
    )
    config.set_id(run.id)
    config.update_paths()

    # Update wandb config with run paths
    wandb.config.update(
        {
            "run_dir": str(config.run_dir),
            "run_id": run.id,
            "checkpoint_dir": str(config.checkpoint_dir),
            "plot_dir": str(config.plot_dir),
        },
        allow_val_change=True,
    )

    dataloaders = {}
    for k in ["train", "validation", "test"]:
        dataloaders[k] = DataLoader(
            StreamDataset(split=k, im_s=config.image_size),
            batch_size=config.batch_size,
            num_workers=config.num_workers,
        )

    model = WorldModel(
        base_name=config.base_name,
        num_tokens=config.num_tokens,
        hidden_dim=config.hidden_dim,
    )

    optimizer = AdamW(model.parameters(), lr=config.learning_rate)

    for batch in dataloaders["train"]:
        optimizer.zero_grad(set_to_none=True)
        x_tp1_pred = model(batch["a_t"], batch["x_t"])
        loss = mse_loss(x_tp1_pred, batch["x_tp1"])
        loss.backward()
        clip_grad_norm_(model.parameters(), max_norm=config.max_grad_norm)
        optimizer.step()
