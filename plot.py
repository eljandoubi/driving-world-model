from pathlib import Path

import matplotlib.animation as animation
import matplotlib.pyplot as plt
import torch
from torch.nn import Module
from tqdm import tqdm

from dataset import StreamDataset

MEAN = torch.tensor([0.4738, 0.4824, 0.4592, 1.0000])
STD = torch.tensor([0.2823, 0.2809, 0.2801, 1e-8])


def _denormalize(
    tensor: torch.Tensor, mean: torch.Tensor, std: torch.Tensor
) -> torch.Tensor:
    """(C, H, W) normalized tensor -> (H, W, 3) RGB clipped to [0, 1]."""
    tensor = tensor.squeeze(0)
    img = tensor * std[:, None, None] + mean[:, None, None]
    img = img[:3].clamp(0, 1).permute(1, 2, 0)
    return img.to("cpu", non_blocking=True)


@torch.inference_mode()
def plot_video(
    model: Module,
    dataloader: StreamDataset,
    save_path: str | Path = "driving_video.mp4",
    num_frames: int = 600,
    fps: int = 10,
    device: torch.device = torch.device("cpu"),
    time_step: int = 25,
    num_timesteps: int = 1000,
) -> Path:
    model.eval()
    assert dataloader.batch_size == 1, "plot_video only supports batch_size=1 for now"
    device_type = device.type  # e.g., "cuda" or "cpu"

    if device_type == "cuda" and torch.cuda.is_bf16_supported():
        amp_dtype = torch.bfloat16
    elif device_type == "cuda":
        amp_dtype = torch.float16  # Fallback to FP16 if GPU doesn't support BF16
    else:
        amp_dtype = torch.bfloat16  # CPU supports BF16 in PyTorch 2.x

    mean = MEAN.to(device, non_blocking=True)
    std = STD.to(device, non_blocking=True)

    fig, (ax_gt, ax_pred) = plt.subplots(1, 2, figsize=(8, 4))
    ax_gt.set_axis_off()
    ax_pred.set_axis_off()
    ax_gt.set_title("Ground Truth")
    ax_pred.set_title("Predicted")
    data_iter = iter(dataloader)
    data = next(data_iter)

    x = data["x_t"].to(device, torch.float32, non_blocking=True)

    all_T = torch.arange(
        num_timesteps, -time_step, -time_step, device=device, dtype=torch.long
    )

    frames: list[tuple[torch.Tensor, torch.Tensor]] = []
    for _ in tqdm(range(num_frames), desc="Generating video frames"):
        a_t = data["a_t"].to(device, non_blocking=True)
        for i in range(len(all_T)):
            t = all_T[i]
            with torch.amp.autocast(device_type=device_type, dtype=amp_dtype):  # pyright: ignore[reportPrivateImportUsage]
                pred_delta = model(a_t, x, t)

            x = x + pred_delta.float()

        gt_img = _denormalize(data["x_tp1"], mean, std)
        pred_img = _denormalize(x, mean, std)
        frames.append((gt_img, pred_img))

        try:
            data = next(data_iter)
        except StopIteration:
            break

    im_gt = ax_gt.imshow(frames[0][0])
    im_pred = ax_pred.imshow(frames[0][1])

    def _update(idx: int):
        im_gt.set_data(frames[idx][0])
        im_pred.set_data(frames[idx][1])
        return im_gt, im_pred

    ani = animation.FuncAnimation(fig, _update, frames=len(frames), blit=True)

    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    ani.save(str(save_path), writer="ffmpeg", fps=fps)
    plt.close(fig)
    return save_path
