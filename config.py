from dataclasses import dataclass
from pathlib import Path
from typing import Literal


@dataclass
class TrainingConfig:
    """world model training configuration."""

    num_tokens: int = 4
    base_name: str = "runwayml/stable-diffusion-v1-5"
    hidden_dim: int = 1024
    activation: Literal[
        "gelu",
        "gelu-approximate",
        "geglu",
        "geglu-approximate",
        "swiglu",
        "linear-silu",
    ] = "swiglu"
    persistent_workers: bool = True
    pin_memory: bool = True
    learning_rate: float = 1e-4
    image_size: int = 64
    num_workers: int = 4
    epochs: int = 100000
    max_grad_norm: float = 1.0
    batch_size: int = 4
    dropout: float = 0.1
    log_every: int = 1000
    checkpoint_every: int = 1000
    patience: int = 5
    min_delta: float = 1e-8
    runs_dir: str | Path = "runs"
    checkpoint_dir: Path = Path("checkpoints")
    plot_dir: Path = Path("plots")
    resume: str = ""  # Path to checkpoint to resume from
    run_id: str | None = None  # Optional run ID for logging (overrides auto-generated)

    def __post_init__(self) -> None:
        assert self.num_tokens > 0, "num_tokens must be > 0"
        assert self.hidden_dim > 0, "hidden_dim must be > 0"
        assert self.num_workers >= self.batch_size, (
            "num_worker must be greater or equal to batch_size"
        )
        assert self.activation in (
            "gelu",
            "gelu-approximate",
            "geglu",
            "geglu-approximate",
            "swiglu",
            "linear-silu",
        ), f"""activation must be one of "gelu",
            "gelu-approximate",
            "geglu",
            "geglu-approximate",
            "swiglu",
            "linear-silu",; got {self.activation}"""

        assert self.learning_rate > 0, "learning_rate must be > 0"
        assert self.epochs > 0, "epochs must be > 0"
        assert self.max_grad_norm > 0, "max_grad_norm must be > 0"
        assert self.batch_size > 0, "batch_size must be > 0"
        assert self.log_every > 0, "log_every must be > 0"
        assert self.image_size > 0, "image_size must be > 0"
        assert self.patience > 0, "patience must be > 0"
        assert self.min_delta >= 0, "min_delta must be >= 0"
        assert self.checkpoint_every > 0, "checkpoint_every must be > 0"
        assert self.dropout >= 0 and self.dropout < 1, "dropout must be in [0,1)"
        if self.resume:
            assert Path(self.resume).is_file(), (
                f"Checkpoint file {self.resume} does not exist"
            )
        assert self.checkpoint_every % self.log_every == 0, (
            "checkpoint_every must be a multiple of log_every"
        )

    def set_id(self, run_id: str) -> None:
        """Set run ID (for logging) after initialization."""
        self.run_id = run_id

    def update_paths(self) -> None:
        """Update checkpoint and plot directories based on base run directory."""
        assert self.run_id is not None, "run_id must be set before updating paths"

        self.run_dir = Path(self.runs_dir) / self.run_id
        self.checkpoint_dir = self.run_dir / Path(self.checkpoint_dir)
        self.plot_dir = self.run_dir / Path(self.plot_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.plot_dir.mkdir(parents=True, exist_ok=True)
        assert isinstance(self.checkpoint_dir, Path) and isinstance(
            self.plot_dir, Path
        ), "checkpoint_dir and plot_dir must be Path objects after update_paths()"
