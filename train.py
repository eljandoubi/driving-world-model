import wandb

from config import TrainingConfig


def main(config: TrainingConfig) -> None:
    # --- WANDB INIT ---
    run = wandb.init(
        project="PINN_CLT",
        config=vars(config),
        id=config.run_id,
        resume="allow" if config.run_id else None,
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
