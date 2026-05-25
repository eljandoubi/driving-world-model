from pathlib import Path

import matplotlib.animation as animation
import matplotlib.pyplot as plt
import torch
from torch.nn import Module
from tqdm import tqdm

from dataset import StreamDataset

MEAN = torch.tensor([0.4738, 0.4824, 0.4592, 1.0000])
STD = torch.tensor([0.2823, 0.2809, 0.2801, 1e-8])
if torch.cuda.is_available():
    MEAN = MEAN.cuda(non_blocking=True)
    STD = STD.cuda(non_blocking=True)


def _denormalize(tensor: torch.Tensor) -> torch.Tensor:
    """(C, H, W) normalized tensor -> (H, W, 3) RGB clipped to [0, 1]."""
    tensor = tensor.squeeze(0)
    img = tensor * STD[:, None, None] + MEAN[:, None, None]
    img = img[:3].clamp(0, 1).permute(1, 2, 0)
    return img.to("cpu", non_blocking=True)


@torch.inference_mode()
def plot_video(
    model: Module,
    dataloader: StreamDataset,
    save_path: str | Path = "driving_video.mp4",
    num_frames: int = 600,
    fps: int = 10,
    device: torch.device | str = "cpu",
    time_step: int = 25,
    num_timesteps: int = 1000,
) -> Path:
    model.eval()
    assert dataloader.batch_size == 1, "plot_video only supports batch_size=1 for now"
    fig, (ax_gt, ax_pred) = plt.subplots(1, 2, figsize=(8, 4))
    ax_gt.set_axis_off()
    ax_pred.set_axis_off()
    ax_gt.set_title("Ground Truth")
    ax_pred.set_title("Predicted")
    data_iter = iter(dataloader)
    data = next(data_iter)

    x = data["x_t"].to(device, non_blocking=True)

    all_T = torch.arange(num_timesteps, -time_step, -time_step, device=device)

    frames: list[tuple[torch.Tensor, torch.Tensor]] = []
    for _ in tqdm(range(num_frames), desc="Generating video frames"):
        a_t = data["a_t"].to(device, non_blocking=True)
        for i in range(len(all_T)):
            t = all_T[i]
            pred_delta = model(a_t, x, t)
            x = x + pred_delta

        gt_img = _denormalize(data["x_tp1"])
        pred_img = _denormalize(x)
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
