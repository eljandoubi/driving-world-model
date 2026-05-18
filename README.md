# Driving World Model

A world model for autonomous driving that predicts next-frame observations conditioned on actions, built on a UNet2D diffusion backbone with learned action embeddings.

## Architecture

- **WorldModel**: Wraps a pretrained Stable Diffusion UNet (`runwayml/stable-diffusion-v1-5`) and conditions it on driving actions (throttle, steer, brake) via cross-attention.
- **ActionEmbedder**: SwiGLU-based feed-forward network projecting 3D actions into multi-token embeddings for cross-attention conditioning.

## Dataset

Uses the [CARLA Autopilot Multimodal Dataset](https://huggingface.co/datasets/immanuelpeter/carla-autopilot-multimodal-dataset) streamed via HuggingFace `datasets`. Each sample pairs consecutive frames `(x_t, a_t) → x_{t+1}`.

## Training

```bash
uv sync
uv run python train.py --learning_rate 1e-4 --batch_size 8 --epochs 100000
```

Key features:
- **Optimizer**: AdamW
- **Scheduler**: CosineAnnealingWarmRestarts (configurable `scheduler_t0`, `scheduler_t_mult`)
- **Early stopping**: Patience-based on validation loss
- **Checkpointing**: Periodic + best-model saving with full state (model, optimizer, scheduler, early stopping)
- **Logging**: Weights & Biases

### Resume from checkpoint

```bash
uv run python train.py --resume runs/<run_id>/checkpoints/best_checkpoint.pt --run_id <run_id>
```

## Configuration

All hyperparameters are defined in `config.py` (`TrainingConfig` dataclass) and parsed via HuggingFace `HfArgumentParser`. Pass any field as a CLI flag (e.g. `--scheduler_t0 20`).

## Testing

```bash
uv sync --group dev
uv run pytest tests/ -v
```

## Project Structure

```
├── model.py            # WorldModel + ActionEmbedder
├── train.py            # Training loop
├── validate.py         # Validation/test evaluation
├── dataset.py          # Streaming dataset from HuggingFace
├── config.py           # TrainingConfig dataclass
├── checkpoint.py       # Save/load checkpoints
├── early_stopping.py   # EarlyStopping utility
├── tests/              # Unit tests
└── pyproject.toml      # Dependencies & config
```

## License

See [LICENSE](LICENSE).