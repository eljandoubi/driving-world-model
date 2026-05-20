import logging
import math

logger = logging.getLogger(__name__)


class EarlyStopping:
    """Stop training when loss stops improving."""

    def __init__(self, patience: int = 5, min_delta: float = 1e-8) -> None:
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss: float | None = None

    def step(self, loss: float) -> bool:
        """Returns True if training should stop."""
        # Handle NaN/Inf values immediately
        if math.isnan(loss) or math.isinf(loss):
            logger.warning(
                "EarlyStopping: Loss is NaN or Inf! Triggering immediate stop."
            )
            return True

        if self.best_loss is None or loss < self.best_loss - self.min_delta:
            self.best_loss = loss
            self.counter = 0
            return False

        self.counter += 1
        return self.counter >= self.patience

    def reset(self) -> None:
        """Reset the early stopping state."""
        self.counter = 0
        self.best_loss = None

    def load_state_dict(self, state_dict: dict) -> None:
        """Load the early stopping state."""
        # Log warnings instead of crashing the script if config changed
        if state_dict.get("patience", self.patience) != self.patience:
            logger.warning(
                f"EarlyStopping: patience changed from {state_dict['patience']} "
                f"(checkpoint) to {self.patience} (current config)."
            )
        if state_dict.get("min_delta", self.min_delta) != self.min_delta:
            logger.warning(
                f"EarlyStopping: min_delta changed from {state_dict['min_delta']} "
                f"(checkpoint) to {self.min_delta} (current config)."
            )

        self.counter = state_dict["counter"]
        self.best_loss = state_dict["best_loss"]

    def state_dict(self) -> dict:
        """Return the early stopping state."""
        return {
            "counter": self.counter,
            "best_loss": self.best_loss,
            "patience": self.patience,
            "min_delta": self.min_delta,
        }
