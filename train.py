import torch
import wandb
from dotenv import load_dotenv
from torch.nn.functional import mse_loss
from torch.nn.utils import clip_grad_norm_
from torch.optim import AdamW
from torch.utils.data import DataLoader
from transformers import HfArgumentParser

from config import TrainingConfig
from dataset import StreamDataset, ceil
from model import WorldModel
from validate import tqdm, validate_model

print("Loading environment variables...", load_dotenv())


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
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("current device", device)
    dataloaders = {}
    for k in ["train", "validation", "test"]:
        dataloaders[k] = DataLoader(
            StreamDataset(split=k, im_s=config.image_size),
            batch_size=config.batch_size,
            num_workers=config.num_workers,
            persistent_workers=config.persistent_workers,
            pin_memory=config.pin_memory,
        )

    model = WorldModel(
        base_name=config.base_name,
        num_tokens=config.num_tokens,
        hidden_dim=config.hidden_dim,
    ).to(device)

    optimizer = AdamW(model.parameters(), lr=config.learning_rate)

    total = int(ceil(len(dataloaders["train"])) / float(config.batch_size))
    pbar = tqdm(
        enumerate(dataloaders["train"], start=1), desc="training epoch", total=total
    )
    cum_loss = 0.0
    for i, batch in pbar:
        model.train()
        optimizer.zero_grad(set_to_none=True)
        batch = batch.to(device)
        x_tp1_pred = model(batch["a_t"], batch["x_t"])
        loss = mse_loss(x_tp1_pred, batch["x_tp1"])
        loss.backward()
        clip_grad_norm_(model.parameters(), max_norm=config.max_grad_norm)
        optimizer.step()
        cum_loss += loss.item()
        pbar.set_postfix(loss=cum_loss / i)

    test_loss = validate_model(model, dataloaders["test"], device, config.batch_size)

    print("final test loss", test_loss)
    wandb.log({"test/loss": test_loss})
    wandb.finish()
    print("Training complete.")


if __name__ == "__main__":
    parser = HfArgumentParser(TrainingConfig)  # pyright: ignore[reportArgumentType]
    args = parser.parse_args_into_dataclasses()
    main(args[0])
