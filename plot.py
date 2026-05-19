from pathlib import Path

import matplotlib.animation as animation
import matplotlib.pyplot as plt
import torch
from torch.nn import Module

from dataset import StreamDataset

MEAN = torch.tensor([0.4738, 0.4824, 0.4592, 1.0000])
STD = torch.tensor([0.2823, 0.2809, 0.2801, 1e-8])


def _denormalize(tensor: torch.Tensor) -> torch.Tensor:
    """(C, H, W) normalized tensor -> (H, W, 3) RGB clipped to [0, 1]."""
    img = tensor.cpu() * STD[:, None, None] + MEAN[:, None, None]
    return img[:3].clamp(0, 1).permute(1, 2, 0)


@torch.inference_mode()
def plot_video(
    model: Module,
    dataloader: StreamDataset,
    save_path: str | Path = "driving_video.mp4",
    num_frames: int = 600,
    fps: int = 10,
    device: torch.device | str = "cpu",
) -> Path:
    model.eval()

    fig, (ax_gt, ax_pred) = plt.subplots(1, 2, figsize=(8, 4))
    ax_gt.set_axis_off()
    ax_pred.set_axis_off()
    ax_gt.set_title("Ground Truth")
    ax_pred.set_title("Predicted")

    frames: list[tuple[torch.Tensor, torch.Tensor]] = []
    for i, data in enumerate(dataloader):
        if i >= num_frames:
            break
        x_t = data["x_t"].unsqueeze(0).to(device)
        a_t = data["a_t"].unsqueeze(0).to(device)
        x_tp1_pred = model(a_t, x_t).squeeze(0)

        gt_img = _denormalize(data["x_tp1"])
        pred_img = _denormalize(x_tp1_pred)
        frames.append((gt_img, pred_img))

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
