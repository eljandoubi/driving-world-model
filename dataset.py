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
    ) -> None:
        super().__init__()
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
            [sample["throttle"], sample["steer"], sample["brake"]]
        )
        meta = {}
        meta["run_id"] = sample["run_id"]
        meta["frame"] = sample["frame"]
        return data, meta

    def __iter__(self):
        worker_info = get_worker_info()
        if worker_info is None:  # single-process data loading, return the full iterator
            itb = self.main_stream
        else:  # in a worker process
            # split workload
            per_worker = int(ceil(len(self) / float(worker_info.num_workers)))
            worker_id = worker_info.id
            iter_start = worker_id * per_worker
            chunk = min(per_worker, len(self) - iter_start)
            itb = self.main_stream.skip(iter_start).take(chunk)

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
