from math import ceil

import torch
from datasets import load_dataset
from torch.utils.data import IterableDataset, get_worker_info
from torchvision import transforms


class TensorDict(dict):
    def to(self, device: torch.device, non_blocking: bool = False):
        for k, v in self.items():
            if hasattr(v, "to"):
                self[k] = v.to(device, non_blocking=non_blocking)
        return self


class StreamDataset(IterableDataset):
    def __init__(
        self,
        name: str = "immanuelpeter/carla-autopilot-multimodal-dataset",
        split: str = "train",
        im_s=256,
        mean=(0.4738, 0.4824, 0.4592, 1.0000),
        std=(0.2823, 0.2809, 0.2801, 1e-8),
        rank: int = 0,
        world_size: int = 1,
    ) -> None:
        super().__init__()
        self.rank = rank
        self.world_size = world_size
        self.main_stream = load_dataset(name, split=split, streaming=True)
        self._len = self.main_stream.info.splits[split].num_examples  # pyright: ignore[reportOptionalSubscript]

        self.trans = transforms.Compose(
            [
                transforms.Resize((im_s, im_s)),
                transforms.ToTensor(),
                transforms.Normalize(mean=mean, std=std),
            ]
        )

    def __len__(self):
        return self._len

    def _pick(self, sample: dict):
        data = TensorDict()
        data["x_t"] = self.trans(sample["image_front"])
        data["a_t"] = torch.tensor(
            [sample["throttle"], sample["steer"], sample["brake"]],
            dtype=torch.float32,
        )
        meta = {}
        meta["run_id"] = sample["run_id"]
        meta["frame"] = sample["frame"]
        return data, meta

    def __iter__(self):
        total = len(self)

        if self.world_size > 1:
            per_rank = total // self.world_size
            rank_start = self.rank * per_rank
            rank_size = per_rank
        else:
            rank_start = 0
            rank_size = total

        worker_info = get_worker_info()
        if worker_info is not None:
            per_worker = int(ceil(rank_size / float(worker_info.num_workers)))
            worker_id = worker_info.id
            worker_start = worker_id * per_worker
            start = rank_start + worker_start
            chunk = min(per_worker, rank_size - worker_start)
        else:
            start = rank_start
            chunk = rank_size

        itb = self.main_stream
        if start > 0:
            itb = itb.skip(start)
        itb = itb.take(chunk)

        run_id = ""
        for it in itb:
            data, meta = self._pick(it)
            if run_id != meta["run_id"]:
                run_id = meta["run_id"]
                idx = meta["frame"]
                current_data = data
                continue
            assert idx < meta["frame"]
            idx = meta["frame"]
            current_data["x_tp1"] = data["x_t"]
            yield current_data
            current_data = data
