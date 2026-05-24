import queue
import threading
from copy import deepcopy
from math import ceil

import torch
from datasets import IterableDataset, load_dataset
from torchvision import transforms
from tqdm import tqdm

from logger import setup_logging


class TensorDict(dict):
    def to(self, device: torch.device, non_blocking: bool = False):
        for k, v in self.items():
            if hasattr(v, "to"):
                self[k] = v.to(device, non_blocking=non_blocking)
        return self

    def pin_memory(self, pin_memory: bool = True):
        if not pin_memory:
            return self
        for k, v in self.items():
            if hasattr(v, "pin_memory"):
                self[k] = v.pin_memory()
        return self


class StreamDataset:
    def __init__(
        self,
        name: str = "immanuelpeter/carla-autopilot-multimodal-dataset",
        split: str = "train",
        batch_size: int = 2,
        im_s=256,
        mean=(0.4738, 0.4824, 0.4592, 1.0000),
        std=(0.2823, 0.2809, 0.2801, 1e-8),
        rank: int = 0,
        world_size: int = 1,
        buffer_size: int = 100,
        pin_memory: bool = True,
    ) -> None:
        super().__init__()
        self.rank = rank
        self.world_size = world_size
        self.main_stream = load_dataset(name, split=split, streaming=True).reshard()
        num_examples = self.main_stream.info.splits[split].num_examples  # pyright: ignore[reportOptionalSubscript]
        self._len_estimated = int(ceil(num_examples / float(world_size * batch_size)))
        self._len = None
        self._counter = 0
        self.batch_size = batch_size
        self.buffer_size = buffer_size
        self.pin_memory = pin_memory
        self.logger = setup_logging(rank=rank)
        self.split = split

        self.trans = transforms.Compose(
            [
                transforms.Resize((im_s, im_s)),
                transforms.ToTensor(),
                transforms.Normalize(mean=mean, std=std),
            ]
        )

    def reset_stream(self):
        self.logger.info(f"Resetting {self.split} stream for rank {self.rank}...")
        self.stream = deepcopy(self.main_stream)
        if self.world_size > 1:
            self.logger.info(
                f"Sharding {self.split} dataset across {self.world_size} processes..., rank {self.rank}"
            )
            self.stream = self.stream.shard(num_shards=self.world_size, index=self.rank)

        return self

    @property
    def initial_step(self):
        return self._counter // self.batch_size

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

    def _generate_samples(self):
        run_id = ""
        if self._counter > 0:
            self.logger.warning(
                f"Resuming {self.split} stream from step {self.initial_step} (counter={self._counter}) for rank {self.rank}..."
            )
        for i, it in enumerate(self.stream):
            if i < self._counter:
                continue
            data, meta = self._pick(it)
            if run_id != meta["run_id"]:
                run_id = meta["run_id"]
                idx = meta["frame"]
                current_data = data
                continue
            assert idx < meta["frame"]
            idx = meta["frame"]
            current_data["x_tp1"] = data["x_t"]
            self._counter += 1
            yield current_data
            current_data = data

        if self._len is None:
            self._len = self._counter // self.batch_size
            self.logger.info(
                f"Computed {self.split} dataset length for rank {self.rank}: {self._len} "
            )

        self._counter = 0

    def __len__(self):
        if self._len is None:
            self.logger.warning(
                f"Length of {self.split} dataset not known yet for rank {self.rank}; returning estimated length {self._len_estimated}"
            )
            return self._len_estimated
        return self._len

    def _collate_fn(self, batch: dict[str, list[torch.Tensor]]) -> TensorDict:
        collated = TensorDict()
        for k in batch.keys():
            collated[k] = torch.stack(batch[k], dim=0)
        return collated

    def __iter__(self):
        q = queue.Queue(maxsize=self.buffer_size)
        stop_token = object()

        iterds = IterableDataset.from_generator(self._generate_samples).batch(
            batch_size=self.batch_size, drop_last_batch=(self.world_size > 1)
        )

        def worker():
            pbar = tqdm(
                iterable=iterds,
                total=self.buffer_size,
                desc="Prefetching",
                unit="samples",
                disable=self.rank > 0,
            )
            for sample in pbar:
                sample = self._collate_fn(sample)
                q.put(sample.pin_memory(self.pin_memory))
                pbar.n = len(q.queue)
                pbar.refresh()

            q.put(stop_token)

        thread = threading.Thread(target=worker, daemon=True)
        thread.start()

        while True:
            item = q.get()
            if item is stop_token:
                break
            yield item

    def state_dict(self) -> dict:
        return {
            "counter": self._counter,
            "len": self._len,
            "rank": self.rank,
            "world_size": self.world_size,
            "batch_size": self.batch_size,
        }

    def load_state_dict(self, state_dict: dict) -> None:
        assert state_dict["rank"] == self.rank, (
            f"Rank mismatch: checkpoint rank {state_dict['rank']} vs current rank {self.rank}"
        )
        assert state_dict["world_size"] == self.world_size, (
            f"World size mismatch: checkpoint world size {state_dict['world_size']} vs current world size {self.world_size}"
        )
        assert state_dict["batch_size"] == self.batch_size, (
            f"Batch size mismatch: checkpoint batch size {state_dict['batch_size']} vs current batch size {self.batch_size}"
        )
        self._counter = state_dict["counter"]
        self._len = state_dict["len"]
