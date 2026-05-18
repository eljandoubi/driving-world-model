import torch
import wandb
from dotenv import load_dotenv
from torch.nn.functional import mse_loss
from torch.nn.utils import clip_grad_norm_
from torch.optim import AdamW
from torch.utils.data import DataLoader
from transformers import HfArgumentParser

from checkpoint import load_checkpoint, save_checkpoint
from config import TrainingConfig
from dataset import StreamDataset, ceil
from early_stopping import EarlyStopping
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
    early_stopping = EarlyStopping(patience=config.patience, min_delta=config.min_delta)

    if config.resume:
        print(f"Resuming from checkpoint {config.resume}...")
        start_epoch, start_iteration = load_checkpoint(
            config.resume, model, optimizer, None, device, early_stopping
        )
        print(f"Resumed from epoch {start_epoch}, iteration {start_iteration}")
    else:
        start_epoch = 0
        start_iteration = 0

    total = int(ceil(len(dataloaders["train"])) / float(config.batch_size))
    pbar = tqdm(
        enumerate(dataloaders["train"], start=1), desc="training one epoch", total=total
    )

    cum_loss = 0.0
    for epoch in tqdm(range(start_epoch, config.epochs), desc="training epochs"):
        for i, batch in pbar:
            if i < start_iteration:
                continue  # skip already trained iterations when resuming
            model.train()
            optimizer.zero_grad(set_to_none=True)
            batch = batch.to(device, non_blocking=True)
            x_tp1_pred = model(batch["a_t"], batch["x_t"])
            loss = mse_loss(x_tp1_pred, batch["x_tp1"])
            loss.backward()
            clip_grad_norm_(model.parameters(), max_norm=config.max_grad_norm)
            optimizer.step()
            cum_loss += loss.item()
            pbar.set_postfix(loss=cum_loss / i)
            if i % config.log_every == 0:
                avg_loss = cum_loss / i
                cum_loss = 0.0  # reset cumulative loss after logging
                pbar.set_postfix(avg_loss=avg_loss)
                wandb.log({"train/loss": avg_loss}, step=i)
            if i % config.checkpoint_every == 0:
                val_loss = validate_model(
                    model, dataloaders["validation"], device, config.batch_size * 4
                )
                pbar.set_postfix(val_loss=val_loss)
                wandb.log({"validation/loss": val_loss}, step=i)
                save_checkpoint(
                    config.checkpoint_dir / f"checkpoint_step_{i}.pt",
                    model,
                    optimizer,
                    None,
                    early_stopping,
                    epoch=i,
                )
                if early_stopping.step(val_loss):
                    print(
                        f"Early stopping at step {i} with validation loss {val_loss:.4f}"
                    )
                    break

        test_loss = validate_model(
            model, dataloaders["test"], device, config.batch_size * 4
        )

    print("final test loss", test_loss)
    wandb.log({"test/loss": test_loss})
    wandb.finish()
    print("Training complete.")


if __name__ == "__main__":
    parser = HfArgumentParser(TrainingConfig)  # pyright: ignore[reportArgumentType]
    args = parser.parse_args_into_dataclasses()
    main(args[0])
