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
        worker_info = get_worker_info()

        # Determine unique ID for each worker-process across all GPUs
        if worker_info is not None:
            num_workers = worker_info.num_workers
            worker_id = worker_info.id
            # Global shard index across all GPUs and workers
            global_worker_id = self.rank * num_workers + worker_id
            total_shards = self.world_size * num_workers
        else:
            global_worker_id = self.rank
            total_shards = self.world_size

        # Let Hugging Face handle the distributed partitioning efficiently!
        if total_shards > 1:
            itb = self.main_stream.shard(
                num_shards=total_shards, index=global_worker_id
            )
        else:
            itb = self.main_stream

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
