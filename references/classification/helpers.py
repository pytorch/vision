import itertools
import os
import random
from functools import partial
from pathlib import Path

import torch
import torch.distributed as dist
import torchvision
from PIL import Image
from torchdata.datapipes.iter import FileLister, IterDataPipe
from torchvision.prototype.datasets.utils._internal import INFINITE_BUFFER_SIZE
from torchvision.prototype.features import Label

IMAGENET_TRAIN_LEN = 1_281_167
IMAGENET_TEST_LEN = 50_000


class _LenSetter(IterDataPipe):
    def __init__(self, dp, root):
        self.dp = dp

        if "train" in str(root):
            self.size = IMAGENET_TRAIN_LEN
        elif "val" in str(root):
            self.size = IMAGENET_TEST_LEN
        else:
            raise ValueError("oops?")

    def __iter__(self):
        yield from self.dp

    def __len__(self):
        # The // world_size part shouldn't be needed. See https://github.com/pytorch/data/issues/533
        return self.size // dist.get_world_size()


def _decode(path, root, categories):
    category = Path(path).relative_to(root).parts[0]

    image = Image.open(path).convert("RGB")
    label = Label.from_category(category, categories=categories)

    return image, label


def _apply_tranforms(img_and_label, transforms):
    img, label = img_and_label
    return transforms(img), label


def make_dp(root, transforms):

    root = Path(root).expanduser().resolve()
    categories = sorted(entry.name for entry in os.scandir(root) if entry.is_dir())
    dp = FileLister(str(root), recursive=True, masks=["*.JPEG"])

    dp = dp.shuffle(buffer_size=INFINITE_BUFFER_SIZE).set_shuffle(False).sharding_filter()
    dp = dp.map(partial(_decode, root=root, categories=categories))
    dp = dp.map(partial(_apply_tranforms, transforms=transforms))

    dp = _LenSetter(dp, root=root)
    return dp


class PreLoadedMapStyle:
    # All the data is pre-loaded and transformed in __init__, so the DataLoader should be crazy fast.
    # This is just to assess how fast a model could theoretically be trained if there was no data bottleneck at all.
    def __init__(self, dir, transform, buffer_size=100):
        dataset = torchvision.datasets.ImageFolder(dir, transform=transform)
        self.size = len(dataset)
        self.samples = [dataset[torch.randint(0, len(dataset), size=(1,)).item()] for i in range(buffer_size)]

    def __len__(self):
        return self.size

    def __getitem__(self, idx):
        return self.samples[idx % len(self.samples)]


class _PreLoadedDP(IterDataPipe):
    # Same as above, but this is a DataPipe
    def __init__(self, root, transforms, buffer_size=100):
        dataset = torchvision.datasets.ImageFolder(root, transform=transforms)
        self.size = len(dataset)
        self.samples = [dataset[torch.randint(0, len(dataset), size=(1,)).item()] for i in range(buffer_size)]
        # Note: the rng might be different across DDP workers so they'll all have different samples.
        # But we don't care about accuracy here so whatever.

    def __iter__(self):
        for idx in range(self.size):
            yield self.samples[idx % len(self.samples)]


def make_pre_loaded_dp(root, transforms):
    dp = _PreLoadedDP(root=root, transforms=transforms)
    dp = dp.shuffle(buffer_size=INFINITE_BUFFER_SIZE).set_shuffle(False).sharding_filter()
    dp = _LenSetter(dp, root=root)
    return dp


class MapStyleToIterable(torch.utils.data.IterableDataset):
    # This converts a MapStyle dataset into an iterable one.
    # Not sure this kind of Iterable dataset is actually useful to benchmark. It
    # was necessary when benchmarking async-io stuff, but not anymore.
    # If anything, it shows how tricky Iterable datasets are to implement.
    def __init__(self, dataset, shuffle):
        self.dataset = dataset
        self.shuffle = shuffle

        self.size = len(self.dataset)
        self.seed = 0  # has to be hard-coded for all DDP workers to have the same shuffling

    def __len__(self):
        return self.size // dist.get_world_size()

    def __iter__(self):

        worker_info = torch.utils.data.get_worker_info()
        num_dl_workers = worker_info.num_workers
        dl_worker_id = worker_info.id

        num_ddp_workers = dist.get_world_size()
        ddp_worker_id = dist.get_rank()

        num_total_workers = num_ddp_workers * num_dl_workers
        current_worker_id = ddp_worker_id + (num_ddp_workers * dl_worker_id)

        indices = range(self.size)
        if self.shuffle:
            rng = random.Random(self.seed)
            indices = rng.sample(indices, k=self.size)
        indices = itertools.islice(indices, current_worker_id, None, num_total_workers)

        samples = (self.dataset[i] for i in indices)
        yield from samples
