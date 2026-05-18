"""Early stopping utility."""


class EarlyStopping:
    """Stop training when loss stops improving."""

    def __init__(self, patience: int = 5, min_delta: float = 1e-8) -> None:
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss: float | None = None

    def step(self, loss: float) -> bool:
        """Returns True if training should stop."""
        if self.best_loss is None or loss < self.best_loss - self.min_delta:
            self.best_loss = loss
            self.counter = 0
            return False
        self.counter += 1
        return self.counter >= self.patience

    def reset(self, only_counter: bool = True) -> None:
        """Reset the early stopping state."""
        self.counter = 0
        if not only_counter:
            self.best_loss = None

    def load_state_dict(self, state_dict) -> None:
        """Load the early stopping state."""
        assert state_dict["patience"] == self.patience, (
            "patience in state_dict does not match"
        )
        assert state_dict["min_delta"] == self.min_delta, (
            "min_delta in state_dict does not match"
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
