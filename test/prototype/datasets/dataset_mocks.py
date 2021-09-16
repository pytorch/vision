import atexit
import functools
import gzip
import itertools
import lzma
import os
import pathlib
import pickle
import shutil
import tarfile
import tempfile
import warnings
import zipfile
from typing import Any, Dict, List, Tuple

import pytest
import torch
import torchvision.datasets
from torch.testing import make_tensor
from torch.utils.data import IterDataPipe
from torchvision.prototype.datasets.utils import Dataset, DatasetConfig, LocalResource

__all__ = ["load"]


class DatasetMocks:
    def __init__(self):
        self._fakedata_fns = {}
        self._tmp_dir = pathlib.Path(tempfile.mkdtemp())
        self._cache: Dict[
            Tuple[str, DatasetConfig], Tuple[List[LocalResource], Dict[str, Any]]
        ] = {}

    def register_fakedata(self, fakedata_fn):
        @functools.wraps(fakedata_fn)
        def wrapper(root, config):
            fakedata_info = fakedata_fn(root, config)
            # FIXME
            if isinstance(fakedata_info, int):
                return dict(num_samples=fakedata_info)
            else:
                return fakedata_info

        self._fakedata_fns[fakedata_fn.__name__] = wrapper
        return wrapper

    def _get(
        self, dataset: Dataset, config: DatasetConfig
    ) -> Tuple[List[LocalResource], Dict[str, Any]]:
        name = dataset.info.name
        resources_and_fakedata_info = self._cache.get((name, config))
        if resources_and_fakedata_info:
            return resources_and_fakedata_info

        try:
            fakedata_fn = self._fakedata_fns[name]
        except KeyError:
            raise pytest.UsageError(f"Unknown fakedata generator {name}.")

        root = pathlib.Path(tempfile.mkdtemp(dir=self._tmp_dir))
        fakedata_info = fakedata_fn(root, config)

        resources_map = {}
        for child in root.iterdir():
            if not child.is_file():
                raise pytest.UsageError("FIXME")

            resource = LocalResource(child)
            resources_map[resource.file_name] = resource

        resources = []
        for resource in dataset.resources(config):
            try:
                resources.append(resources_map.pop(resource.file_name))
            except KeyError as error:
                raise pytest.UsageError("FIXME") from error

        if resources_map:
            warnings.warn("FIXME")

        self._cache[(name, config)] = resources, fakedata_info
        return resources, fakedata_info

    def load(
        self, name: str, decoder=None, **options: Any
    ) -> Tuple[IterDataPipe, Dict[str, Any]]:
        dataset = torchvision.prototype.datasets._api.find(name)
        config = dataset.info.make_config(**options)
        resources, fakedata_info = self._get(dataset, config)
        datapipe = dataset._make_datapipe(
            [resource.to_datapipe() for resource in resources],
            config=config,
            decoder=decoder,
        )
        return datapipe, fakedata_info

    def cleanup(self):
        shutil.rmtree(self._tmp_dir)


dataset_mocks = DatasetMocks()
load = dataset_mocks.load
atexit.register(lambda: dataset_mocks.cleanup())


def _split_files_or_dirs(root, *files_or_dirs):
    files = set()
    dirs = set()
    for file_or_dir in files_or_dirs:
        path = pathlib.Path(file_or_dir)
        if not path.is_absolute():
            path = root / path
        if path.is_file():
            files.add(path)
        else:
            dirs.add(path)
            for sub_file_or_dir in path.glob("**/*"):
                if sub_file_or_dir.is_file():
                    files.add(sub_file_or_dir)
                else:
                    dirs.add(sub_file_or_dir)

    if root in dirs:
        dirs.remove(root)

    return files, dirs


def _make_archive(root, name, *files_or_dirs, opener, adder, remove=True):
    archive = pathlib.Path(root) / name
    files, dirs = _split_files_or_dirs(root, *files_or_dirs)

    with opener(archive) as fh:
        for file in files:
            adder(fh, file, file.relative_to(root))

    if remove:
        for file in files:
            os.remove(file)
        for folder in dirs:
            os.rmdir(folder)

    return archive


def make_tar(root, name, *files_or_dirs, remove=True, compression=None):
    # TODO: detect compression from name
    return _make_archive(
        root,
        name,
        *files_or_dirs,
        opener=lambda archive: tarfile.open(
            archive, f"w:{compression}" if compression else "w"
        ),
        adder=lambda fh, file, relative_file: fh.add(file, arcname=relative_file),
        remove=remove,
    )


def make_zip(root, name, *files_or_dirs, remove=True):
    return _make_archive(
        root,
        name,
        *files_or_dirs,
        opener=lambda archive: zipfile.ZipFile(archive, "w"),
        adder=lambda fh, file, relative_file: fh.write(file, arcname=relative_file),
        remove=remove,
    )


class MNISTFakedata:
    _MAGIC_DTYPES = {
        torch.uint8: 8,
        torch.int8: 9,
        torch.int16: 11,
        torch.int32: 12,
        torch.float32: 13,
        torch.float64: 14,
    }

    @classmethod
    def _magic(cls, dtype, ndim):
        return cls._MAGIC_DTYPES[dtype] * 256 + ndim

    @staticmethod
    def _encode(t):
        return torch.tensor(t, dtype=torch.int32).numpy().tobytes()[::-1]

    @classmethod
    def _create_binary_file(cls, root, filename, *, shape, dtype, compressor):
        with compressor(root / filename, "wb") as fh:
            for meta in (cls._magic(dtype, len(shape)), *shape):
                fh.write(cls._encode(meta))

            data = make_tensor(shape, device="cpu", dtype=dtype)
            fh.write(data.numpy().tobytes())

    @classmethod
    def generate(
        cls,
        root,
        *,
        num_samples,
        images_file,
        labels_file,
        image_size=(28, 28),
        image_dtype=torch.uint8,
        label_size=(),
        label_dtype=torch.uint8,
        compressor=None,
    ):
        if compressor is None:
            compressor = gzip.open

        cls._create_binary_file(
            root,
            images_file,
            shape=(num_samples, *image_size),
            dtype=image_dtype,
            compressor=compressor,
        )
        cls._create_binary_file(
            root,
            labels_file,
            shape=(num_samples, *label_size),
            dtype=label_dtype,
            compressor=compressor,
        )


@dataset_mocks.register_fakedata
def mnist(root, config):
    train = config.split == "train"
    num_samples = 2 if train else 1
    images_file = f"{'train' if train else 't10k'}-images-idx3-ubyte.gz"
    labels_file = f"{'train' if train else 't10k'}-labels-idx1-ubyte.gz"
    MNISTFakedata.generate(
        root, num_samples=num_samples, images_file=images_file, labels_file=labels_file
    )
    return num_samples


@dataset_mocks.register_fakedata
def fashionmnist(root, config):
    train = config.split == "train"
    num_samples = 2 if train else 1
    images_file = f"{'train' if train else 't10k'}-images-idx3-ubyte.gz"
    labels_file = f"{'train' if train else 't10k'}-labels-idx1-ubyte.gz"
    MNISTFakedata.generate(
        root, num_samples=num_samples, images_file=images_file, labels_file=labels_file
    )
    return num_samples


@dataset_mocks.register_fakedata
def kmnist(root, config):
    train = config.split == "train"
    num_samples = 2 if train else 1
    images_file = f"{'train' if train else 't10k'}-images-idx3-ubyte.gz"
    labels_file = f"{'train' if train else 't10k'}-labels-idx1-ubyte.gz"
    MNISTFakedata.generate(
        root, num_samples=num_samples, images_file=images_file, labels_file=labels_file
    )
    return num_samples


@dataset_mocks.register_fakedata
def emnist(root, config):
    num_samples = 2 if config.split == "train" else 1

    splits = ("train", "test")
    image_sets = ("mnist", "byclass", "bymerge", "balanced", "digits", "letters")
    file_names = set()
    for split, image_set in itertools.product(splits, image_sets):
        images_file = f"emnist-{image_set}-{split}-images-idx3-ubyte.gz"
        labels_file = f"emnist-{image_set}-{split}-labels-idx1-ubyte.gz"
        file_names.update({images_file, labels_file})
        MNISTFakedata.generate(
            root,
            num_samples=num_samples,
            images_file=images_file,
            labels_file=labels_file,
        )

    make_zip(root, "emnist-gzip.zip", *file_names)

    return num_samples


@dataset_mocks.register_fakedata
def qmnist(root, config):
    if config.split == "train":
        num_samples = num_samples_gen = 2
        prefix = "qmnist-train"
        suffix = ".gz"
        compressor = gzip.open
    elif config.split.startswith("test"):
        num_samples = 1
        # The split 'test50k' is defined as the last 50k images beginning at index 10000. Thus, we need to create more
        # than 10000 images for the dataset to not be empty.
        num_samples_gen = num_samples + (10000 if config.split == "test50k" else 0)
        prefix = "qmnist-test"
        suffix = ".gz"
        compressor = gzip.open
    else:  # config.split == "nist"
        num_samples = num_samples_gen = 3
        prefix = "xnist"
        suffix = ".xz"
        compressor = lzma.open

    MNISTFakedata.generate(
        root,
        num_samples=num_samples_gen,
        images_file=f"{prefix}-images-idx3-ubyte{suffix}",
        labels_file=f"{prefix}-labels-idx2-int{suffix}",
        label_size=(8,),
        label_dtype=torch.int32,
        compressor=compressor,
    )

    return num_samples


class CIFARFakedata:
    NUM_PIXELS = 32 * 32 * 3

    @classmethod
    def _create_batch_file(cls, root, name, *, num_samples, num_categories, labels_key):
        content = {
            "data": make_tensor(
                (num_samples, cls.NUM_PIXELS), device="cpu", dtype=torch.uint8
            ).numpy(),
            labels_key: torch.randint(0, num_categories, size=(num_samples,)).tolist(),
        }
        with open(pathlib.Path(root) / name, "wb") as fh:
            pickle.dump(content, fh)

    @classmethod
    def generate(
        cls,
        root,
        name,
        *,
        folder,
        train_files,
        test_files,
        num_samples_per_file,
        num_categories,
        labels_key,
    ):
        folder = root / folder
        folder.mkdir()
        files = (*train_files, *test_files)
        for file in files:
            cls._create_batch_file(
                folder,
                file,
                num_samples=num_samples_per_file,
                num_categories=num_categories,
                labels_key=labels_key,
            )

        make_tar(root, name, folder, compression="gz")


@dataset_mocks.register_fakedata
def cifar10(root, config):
    train_files = [f"data_batch_{idx}" for idx in range(1, 6)]
    test_files = ["test_batch"]
    if config.split == "train":
        num_samples_per_file = 2
        num_samples = len(train_files) * num_samples_per_file
    else:
        num_samples_per_file = 1
        num_samples = len(test_files) * num_samples_per_file

    CIFARFakedata.generate(
        root=root,
        name="cifar-10-python.tar.gz",
        folder=pathlib.Path("cifar-10-batches-py"),
        train_files=train_files,
        test_files=test_files,
        num_samples_per_file=2 if config.split == "train" else 1,
        num_categories=10,
        labels_key="labels",
    )
    return num_samples


@dataset_mocks.register_fakedata
def cifar100(root, config):
    train_files = ["train"]
    test_files = ["test"]
    if config.split == "train":
        num_samples_per_file = 2
        num_samples = len(train_files) * num_samples_per_file
    else:
        num_samples_per_file = 1
        num_samples = len(test_files) * num_samples_per_file

    CIFARFakedata.generate(
        root=root,
        name="cifar-100-python.tar.gz",
        folder=pathlib.Path("cifar-100-python"),
        train_files=train_files,
        test_files=test_files,
        num_samples_per_file=num_samples_per_file,
        num_categories=100,
        labels_key="fine_labels",
    )

    return num_samples
