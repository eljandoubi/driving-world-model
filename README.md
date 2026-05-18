# Driving World Model

A world model for autonomous driving that learns to predict the next visual observation given the current frame and a driving action. The model is designed for hyperparameter tuning — all training parameters are exposed as CLI flags, making it straightforward to run sweeps with tools like W&B Sweeps, Optuna, or grid search scripts.

## Motivation

World models enable planning and simulation without interacting with the real environment. This project frames next-frame prediction as a conditional generation task: given the current camera frame $x_t$ and a 3-dimensional action $a_t = (\text{throttle}, \text{steer}, \text{brake})$, the model predicts $x_{t+1}$.

## Architecture

| Component | Description |
|-----------|-------------|
| **WorldModel** | Pretrained Stable Diffusion UNet conditioned on actions via cross-attention |
| **ActionEmbedder** | Feed-forward network (SwiGLU) projecting $\mathbb{R}^3 \to \mathbb{R}^{T \times D}$ multi-token embeddings |
| **Loss** | MSE between predicted and ground-truth next frame |

The UNet backbone is loaded from `runwayml/stable-diffusion-v1-5` (configurable). Actions are embedded into `num_tokens` vectors of dimension `cross_attention_dim` and injected as the encoder hidden states.

## Dataset

Uses the [CARLA Autopilot Multimodal Dataset](https://huggingface.co/datasets/immanuelpeter/carla-autopilot-multimodal-dataset) streamed via HuggingFace `datasets`. Each sample pairs consecutive frames within the same driving run: $(x_t, a_t) \to x_{t+1}$.

## Installation

```bash
uv sync            # production deps
uv sync --group dev  # + pytest for testing
```

## Training

```bash
uv run python train.py [OPTIONS]
```

All fields in `TrainingConfig` are valid CLI flags (via HuggingFace `HfArgumentParser`). This makes it trivial to run hyperparameter sweeps:

```bash
# Example: sweep over learning rate and scheduler
uv run python train.py --learning_rate 3e-4 --scheduler_t0 5 --scheduler_t_mult 1
uv run python train.py --learning_rate 1e-4 --scheduler_t0 20 --scheduler_t_mult 2
```

### Resume from checkpoint

```bash
uv run python train.py --resume runs/<run_id>/checkpoints/best_checkpoint.pt --run_id <run_id>
```

## Default Configuration

### Model

| Parameter | Default | Description |
|-----------|---------|-------------|
| `base_name` | `runwayml/stable-diffusion-v1-5` | Pretrained UNet backbone |
| `num_tokens` | `8` | Number of action embedding tokens |
| `hidden_dim` | `1024` | Action embedder hidden dimension |
| `activation` | `swiglu` | Activation in action embedder (`gelu`, `geglu`, `swiglu`, etc.) |
| `dropout` | `0.1` | Dropout in action embedder |

### Optimization

| Parameter | Default | Description |
|-----------|---------|-------------|
| `learning_rate` | `1e-4` | AdamW learning rate |
| `max_grad_norm` | `1.0` | Gradient clipping max norm |
| `scheduler_t0` | `10` | CosineAnnealingWarmRestarts $T_0$ (epochs per first restart) |
| `scheduler_t_mult` | `2` | CosineAnnealingWarmRestarts $T_{mult}$ (period multiplier) |
| `epochs` | `100000` | Maximum training epochs |

### Early Stopping & Checkpointing

| Parameter | Default | Description |
|-----------|---------|-------------|
| `patience` | `10` | Early stopping patience (number of validations without improvement) |
| `min_delta` | `1e-8` | Minimum improvement to reset patience counter |
| `log_every` | `1000` | Log training loss every N steps |
| `checkpoint_every` | `10000` | Validate & checkpoint every N steps |

### Data

| Parameter | Default | Description |
|-----------|---------|-------------|
| `image_size` | `256` | Input image resize resolution |
| `batch_size` | `8` | Training batch size |
| `num_workers` | `8` | DataLoader workers |
| `persistent_workers` | `True` | Keep workers alive between epochs |
| `pin_memory` | `True` | Pin memory for GPU transfer |

### Paths

| Parameter | Default | Description |
|-----------|---------|-------------|
| `runs_dir` | `runs` | Base directory for all run outputs |
| `checkpoint_dir` | `checkpoints` | Subdirectory for checkpoints (under `runs/<run_id>/`) |
| `plot_dir` | `plots` | Subdirectory for plots (under `runs/<run_id>/`) |
| `resume` | `""` | Path to checkpoint to resume from |
| `run_id` | `None` | W&B run ID (auto-generated if not set) |

## Hyperparameter Tuning

Key parameters to sweep:

- **`learning_rate`**: Try log-uniform in `[1e-5, 1e-3]`
- **`num_tokens`**: `{4, 8, 16}` — controls action conditioning capacity
- **`hidden_dim`**: `{512, 1024, 2048}` — action embedder width
- **`scheduler_t0`** / **`scheduler_t_mult`**: Controls warm restart frequency
- **`dropout`**: `[0.0, 0.3]`
- **`image_size`**: `{128, 256, 512}` — resolution/compute trade-off
- **`batch_size`**: `{4, 8, 16, 32}` — adjust with available GPU memory

All parameters are logged to W&B automatically, enabling comparison across runs.

## Testing

```bash
uv run pytest tests/ -v
```

## Project Structure

```
├── model.py            # WorldModel + ActionEmbedder
├── train.py            # Training loop with scheduler, early stopping, checkpointing
├── validate.py         # Validation/test evaluation
├── dataset.py          # Streaming dataset from HuggingFace
├── config.py           # TrainingConfig dataclass (all hyperparameters)
├── checkpoint.py       # Save/load checkpoints
├── early_stopping.py   # EarlyStopping utility
├── tests/              # Unit tests
└── pyproject.toml      # Dependencies & config
```

## License

See [LICENSE](LICENSE).