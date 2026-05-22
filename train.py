import gc
import os

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import wandb
from dotenv import load_dotenv
from torch.nn.functional import l1_loss, mse_loss
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.nn.utils import clip_grad_norm_
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
from transformers import HfArgumentParser

from checkpoint import load_checkpoint, save_checkpoint
from config import TrainingConfig
from dataset import StreamDataset
from early_stopping import EarlyStopping
from logger import setup_logging
from model import WorldModel
from plot import plot_video
from validate import tqdm, validate_model

os.environ["WANDB_DISABLE_SYMLINKS"] = "true"
load_dotenv()

LOSS_FN_MAP = {"l2": mse_loss, "l1": l1_loss}


def setup_ddp(local_rank: int, global_rank: int, world_size: int) -> None:
    os.environ.setdefault("MASTER_ADDR", "localhost")
    os.environ.setdefault("MASTER_PORT", "12355")
    dist.init_process_group("nccl", rank=global_rank, world_size=world_size)
    torch.cuda.set_device(local_rank)


def cleanup_ddp() -> None:
    dist.destroy_process_group()


def main(local_rank: int, config: TrainingConfig) -> None:
    world_size = config.n_gpus * config.n_nodes
    global_rank = config.node_rank * config.n_gpus + local_rank
    is_main = global_rank == 0

    if world_size > 1:
        setup_ddp(local_rank, global_rank, world_size)
        device = torch.device(f"cuda:{local_rank}")
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    logger = setup_logging(rank=global_rank)

    logger.info(f"[Rank {global_rank}] using device {device}")

    # --- WANDB INIT (rank 0 only) ---
    if is_main:
        run = wandb.init(
            project="driving-world-model",
            config=vars(config),
            id=config.run_id,
            resume="allow" if config.run_id else None,
            # mode="disabled",  # for debug
        )
        config.set_id(run.id)
        config.update_paths()

        wandb.config.update(
            {
                "run_dir": str(config.run_dir),
                "run_id": run.id,
                "checkpoint_dir": str(config.checkpoint_dir),
                "plot_dir": str(config.plot_dir),
            },
            allow_val_change=True,
        )

    loss_fn = LOSS_FN_MAP[config.loss_type]

    dataloaders = {}
    for k in ["train", "validation", "test"]:
        for prefix in ["video", ""]:
            if prefix == "video" and k == "train":
                continue  # only create video dataloaders for validation and test sets
            if prefix == "video":
                batch_size = 1
                buffer_size = (
                    config.buffer_size // 10
                )  # use smaller buffer size for video dataloaders to reduce memory usage
            else:
                batch_size = config.batch_size
                if k != "train":
                    batch_size *= 4  # use larger batch size for validation and test to speed up evaluation
                buffer_size = config.buffer_size
            key = f"{prefix}_{k}" if prefix else k

            dataloaders[key] = StreamDataset(
                split=k,
                im_s=config.image_size,
                rank=global_rank,
                world_size=world_size,
                batch_size=batch_size,
                pin_memory=config.pin_memory,
                buffer_size=buffer_size,
            )

    raw_model = WorldModel(
        base_name=config.base_name,
        num_tokens=config.num_tokens,
        hidden_dim=config.hidden_dim,
        activation=config.activation,
        dropout=config.dropout,
    ).to(device, non_blocking=True)

    if world_size > 1:
        model = DDP(raw_model, device_ids=[local_rank])
    else:
        model = raw_model

    optimizer = AdamW(model.parameters(), lr=config.learning_rate)
    scheduler = CosineAnnealingWarmRestarts(
        optimizer, T_0=config.scheduler_t0, T_mult=config.scheduler_t_mult
    )

    early_stopping = EarlyStopping(
        patience=config.patience, min_delta=config.min_delta
    )

    if config.resume:
        if is_main:
            logger.info(f"Resuming from checkpoint {config.resume}...")
        start_epoch, best_val_loss = load_checkpoint(
            config.resume, raw_model, optimizer, scheduler, device, early_stopping, dataloaders["train"], global_rank
        )
        if is_main:
            logger.info(
                f"Resumed from epoch {start_epoch}"
            )
    else:
        start_epoch = 0
        best_val_loss = float("inf")

    

    model.train()
    avg_loss = 0.0
    stopped = False
    if world_size > 1:
        stopped_tensor = torch.tensor([0], device=device)
    
    for epoch in tqdm(
        range(start_epoch, config.epochs), desc="training epochs", disable=not is_main
    ):
        cum_loss = 0.0
        total = len(dataloaders["train"])
        pbar = tqdm(
            enumerate(dataloaders["train"].reset_stream(), start=1),
            desc=f"epoch {epoch} training rank {global_rank}",
            total=total,
            disable=not is_main,
            initial=dataloaders["train"].initial_step,
        )

        for i, batch in pbar:
            step = epoch * total + i
            optimizer.zero_grad(set_to_none=True)
            batch = batch.to(device, non_blocking=True)
            pred_delta = model(batch["a_t"], batch["x_t"])
            target_delta = batch["x_tp1"] - batch["x_t"]
            loss = loss_fn(pred_delta, target_delta)
            loss.backward()
            clip_grad_norm_(model.parameters(), max_norm=config.max_grad_norm)
            optimizer.step()

            if is_main:
                loss_value = loss.item()
                cum_loss += loss_value
                avg_loss += loss_value
                pbar.set_postfix(loss=cum_loss / i)
                if step % config.log_every == 0:
                    avg_loss /= config.log_every
                    pbar.set_postfix(avg_loss=avg_loss)
                    wandb.log(
                        {
                            "train/avg_loss": avg_loss,
                            "train/cum_loss": cum_loss / i,
                            "train/lr": scheduler.get_last_lr()[0],
                        },
                        step=step,
                    )
                    avg_loss = 0.0

            if step % config.checkpoint_every == 0:
                gc.collect()  # collect garbage before validation to free up memory
                torch.cuda.empty_cache()  # clear CUDA cache before validation
                

                val_loss = validate_model(
                    model,
                    dataloaders["validation"].reset_stream(),
                    device,
                    loss_fn,
                    world_size,
                    is_main,
                )

                if is_main:
                    pbar.set_postfix(val_loss=val_loss)
                    wandb.log({"validation/loss": val_loss}, step=step)

                ckpt_path = config.checkpoint_dir / f"checkpoint_step_{step}.pt"
                save_checkpoint(
                    ckpt_path,
                    raw_model,
                    optimizer,
                    scheduler,
                    early_stopping,
                    dataset=dataloaders["train"],
                    epoch=epoch,
                    best_val_loss=best_val_loss,
                    rank=global_rank,
                    world_size=world_size,
                )

                if is_main:
                    wandb.save(str(ckpt_path), base_path=str(config.run_dir))
                    video_path = plot_video(
                        raw_model,
                        dataloaders["video_validation"].reset_stream(),
                        save_path=config.plot_dir / f"driving_video_step_{step}.mp4",
                        device=device,
                    )
                    wandb.log(
                        {
                            "video/prediction": wandb.Video(
                                str(video_path), format="mp4"
                            )
                        },
                        step=step,
                    )

                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    best_ckpt_path = config.checkpoint_dir / "best_checkpoint.pt"
                    save_checkpoint(
                        best_ckpt_path,
                        raw_model,
                        optimizer,
                        scheduler,
                        early_stopping,
                        dataset=dataloaders["train"],
                        epoch=epoch,
                        best_val_loss=best_val_loss,
                        rank=global_rank,
                        world_size=world_size,
                    )

                    if is_main:
                        wandb.save(str(best_ckpt_path), base_path=str(config.run_dir))
                        best_video_path = plot_video(
                            raw_model,
                            dataloaders["video_test"].reset_stream(),
                            save_path=config.plot_dir / "best_driving_video.mp4",
                            device=device,
                        )
                        wandb.log(
                            {
                                "video/best_prediction": wandb.Video(
                                    str(best_video_path), format="mp4"
                                )
                            },
                            step=step,
                        )

                if early_stopping.step(val_loss):
                    if is_main:
                        logger.info(
                            f"Early stopping at step {step} with validation loss {val_loss:.4f}"
                        )
                    stopped = True

                # Propagate stopped flag to all workers
                if world_size > 1:
                    if is_main:
                        stopped_tensor[0] = int(stopped)
                    dist.broadcast(stopped_tensor, src=0)
                    stopped = bool(stopped_tensor.item())

                if stopped:
                    break

                if world_size > 1:
                    dist.barrier()

                model.train()  # ensure model is in train mode after validation

        if stopped:
            break

        scheduler.step()

    gc.collect()  # collect garbage before validation to free up memory
    torch.cuda.empty_cache()  # clear CUDA cache before validation
    test_loss = validate_model(
        model,
        dataloaders["test"].reset_stream(),
        device,
        loss_fn,
        world_size,
        is_main=is_main,
    )
    logger.info("final test loss %f", test_loss)
    final_ckpt_path = config.checkpoint_dir / "final_checkpoint.pt"

    if is_main:
        wandb.log({"test/loss": test_loss})
        wandb.save(str(final_ckpt_path), base_path=str(config.run_dir))

    save_checkpoint(
        final_ckpt_path,
        raw_model,
        optimizer,
        scheduler,
        early_stopping,
        dataset=dataloaders["train"],
        epoch=epoch,
        best_val_loss=best_val_loss,
        rank=global_rank,
        world_size=world_size,
    )
    if is_main:
        final_video_path = plot_video(
            raw_model,
            dataloaders["video_test"].reset_stream(),
            save_path=config.plot_dir / "final_driving_video.mp4",
            device=device,
        )
        wandb.log(
            {
                "video/final_prediction": wandb.Video(
                    str(final_video_path), format="mp4"
                )
            },
        )
        
        wandb.finish()
    
    logger.info("Training complete.")

    if world_size > 1:
        cleanup_ddp()


if __name__ == "__main__":
    parser = HfArgumentParser(TrainingConfig)  # pyright: ignore[reportArgumentType]
    config = parser.parse_args_into_dataclasses()[0]

    if "LOCAL_RANK" in os.environ:
        # Launched via torchrun — env vars override config
        local_rank = int(os.environ["LOCAL_RANK"])
        config.n_gpus = int(os.environ.get("LOCAL_WORLD_SIZE", config.n_gpus))  # pyright: ignore[reportArgumentType]
        total_world_size = int(os.environ["WORLD_SIZE"])
        config.n_nodes = total_world_size // config.n_gpus
        main(local_rank, config)
    elif config.n_gpus * config.n_nodes > 1:
        mp.spawn(main, args=(config,), nprocs=config.n_gpus, join=True)  # pyright: ignore[reportPrivateImportUsage, reportAttributeAccessIssue]
    else:
        main(0, config)
