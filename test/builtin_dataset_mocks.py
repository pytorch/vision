import collections.abc
import contextlib
import csv
import functools
import gzip
import itertools
import json
import lzma
import pathlib
import pickle
import random
import tempfile
import unittest.mock
import xml.etree.ElementTree as ET
from collections import defaultdict, Counter, UserDict

import numpy as np
import PIL.Image
import pytest
import torch
from datasets_utils import make_zip, make_tar, create_image_folder, create_image_file
from torch.nn.functional import one_hot
from torch.testing import make_tensor as _make_tensor
from torchvision.prototype import datasets
from torchvision.prototype.datasets._api import find
from torchvision.prototype.utils._internal import sequence_to_str

make_tensor = functools.partial(_make_tensor, device="cpu")
make_scalar = functools.partial(make_tensor, ())

TEST_HOME = pathlib.Path(tempfile.mkdtemp())


__all__ = ["DATASET_MOCKS", "parametrize_dataset_mocks"]


class ResourceMock(datasets.utils.OnlineResource):
    def __init__(self, *, dataset_name, dataset_config, **kwargs):
        super().__init__(**kwargs)
        self.dataset_name = dataset_name
        self.dataset_config = dataset_config

    def _download(self, _):
        raise pytest.UsageError(
            f"Dataset '{self.dataset_name}' requires the file '{self.file_name}' for {self.dataset_config}, "
            f"but this file does not exist."
        )


class DatasetMock:
    def __init__(self, name, mock_data_fn, *, configs=None):
        self.dataset = find(name)
        self.root = TEST_HOME / self.dataset.name
        self.mock_data_fn = mock_data_fn
        self.configs = configs or self.info._configs
        self._cache = {}

    @property
    def info(self):
        return self.dataset.info

    @property
    def name(self):
        return self.info.name

    def _parse_mock_data(self, config, mock_infos):
        if mock_infos is None:
            raise pytest.UsageError(
                f"The mock data function for dataset '{self.name}' returned nothing. It needs to at least return an "
                f"integer indicating the number of samples for the current `config`."
            )

        key_types = set(type(key) for key in mock_infos) if isinstance(mock_infos, dict) else {}
        if datasets.utils.DatasetConfig not in key_types:
            mock_infos = {config: mock_infos}
        elif len(key_types) > 1:
            raise pytest.UsageError(
                f"Unable to handle the returned dictionary of the mock data function for dataset {self.name}. If "
                f"returned dictionary uses `DatasetConfig` as key type, all keys should be of that type."
            )

        for config_, mock_info in list(mock_infos.items()):
            if config_ in self._cache:
                raise pytest.UsageError(
                    f"The mock info for config {config_} of dataset {self.name} generated for config {config} "
                    f"already exists in the cache."
                )
            if isinstance(mock_info, int):
                mock_infos[config_] = dict(num_samples=mock_info)
            elif not isinstance(mock_info, dict):
                raise pytest.UsageError(
                    f"The mock data function for dataset '{self.name}' returned a {type(mock_infos)} for `config` "
                    f"{config_}. The returned object should be a dictionary containing at least the number of "
                    f"samples for the key `'num_samples'`. If no additional information is required for specific "
                    f"tests, the number of samples can also be returned as an integer."
                )
            elif "num_samples" not in mock_info:
                raise pytest.UsageError(
                    f"The dictionary returned by the mock data function for dataset '{self.name}' and config "
                    f"{config_} has to contain a `'num_samples'` entry indicating the number of samples."
                )

        return mock_infos

    def _prepare_resources(self, config):
        with contextlib.suppress(KeyError):
            return self._cache[config]

        self.root.mkdir(exist_ok=True)
        mock_infos = self._parse_mock_data(config, self.mock_data_fn(self.info, self.root, config))

        available_file_names = {path.name for path in self.root.glob("*")}
        for config_, mock_info in mock_infos.items():
            required_file_names = {resource.file_name for resource in self.dataset.resources(config_)}
            missing_file_names = required_file_names - available_file_names
            if missing_file_names:
                raise pytest.UsageError(
                    f"Dataset '{self.name}' requires the files {sequence_to_str(sorted(missing_file_names))} "
                    f"for {config_}, but they were not created by the mock data function."
                )

            self._cache[config_] = mock_info

        return self._cache[config]

    @contextlib.contextmanager
    def prepare(self, config):
        mock_info = self._prepare_resources(config)
        with unittest.mock.patch("torchvision.prototype.datasets._api.home", return_value=str(TEST_HOME)):
            yield mock_info


def config_id(name, config):
    parts = [name]
    for name, value in config.items():
        if isinstance(value, bool):
            part = ("" if value else "no_") + name
        else:
            part = str(value)
        parts.append(part)
    return "-".join(parts)


def parametrize_dataset_mocks(*dataset_mocks, marks=None):
    mocks = {}
    for mock in dataset_mocks:
        if isinstance(mock, DatasetMock):
            mocks[mock.name] = mock
        elif isinstance(mock, collections.abc.Sequence):
            mocks.update({mock_.name: mock_ for mock_ in mock})
        elif isinstance(mock, collections.abc.Mapping):
            mocks.update(mock)
        else:
            raise pytest.UsageError(
                f"The positional arguments passed to `parametrize_dataset_mocks` can either be a `DatasetMock`, "
                f"a sequence of `DatasetMock`'s, or a mapping of names to `DatasetMock`'s, "
                f"but got {mock} instead."
            )
    dataset_mocks = mocks

    if marks is None:
        marks = {}
    elif not isinstance(marks, collections.abc.Mapping):
        raise pytest.UsageError()

    return pytest.mark.parametrize(
        ("dataset_mock", "config"),
        [
            pytest.param(dataset_mock, config, id=config_id(name, config), marks=marks.get(name, ()))
            for name, dataset_mock in dataset_mocks.items()
            for config in dataset_mock.configs
        ],
    )


class DatasetMocks(UserDict):
    def set_from_named_callable(self, fn):
        name = fn.__name__.replace("_", "-")
        self.data[name] = DatasetMock(name, fn)
        return fn


DATASET_MOCKS = DatasetMocks()


class MNISTMockData:
    _DTYPES_ID = {
        torch.uint8: 8,
        torch.int8: 9,
        torch.int16: 11,
        torch.int32: 12,
        torch.float32: 13,
        torch.float64: 14,
    }

    @classmethod
    def _magic(cls, dtype, ndim):
        return cls._DTYPES_ID[dtype] * 256 + ndim + 1

    @staticmethod
    def _encode(t):
        return torch.tensor(t, dtype=torch.int32).numpy().tobytes()[::-1]

    @staticmethod
    def _big_endian_dtype(dtype):
        np_dtype = getattr(np, str(dtype).replace("torch.", ""))().dtype
        return np.dtype(f">{np_dtype.kind}{np_dtype.itemsize}")

    @classmethod
    def _create_binary_file(cls, root, filename, *, num_samples, shape, dtype, compressor, low=0, high):
        with compressor(root / filename, "wb") as fh:
            for meta in (cls._magic(dtype, len(shape)), num_samples, *shape):
                fh.write(cls._encode(meta))

            data = make_tensor((num_samples, *shape), dtype=dtype, low=low, high=high)

            fh.write(data.numpy().astype(cls._big_endian_dtype(dtype)).tobytes())

    @classmethod
    def generate(
        cls,
        root,
        *,
        num_categories,
        num_samples=None,
        images_file,
        labels_file,
        image_size=(28, 28),
        image_dtype=torch.uint8,
        label_size=(),
        label_dtype=torch.uint8,
        compressor=None,
    ):
        if num_samples is None:
            num_samples = num_categories
        if compressor is None:
            compressor = gzip.open

        cls._create_binary_file(
            root,
            images_file,
            num_samples=num_samples,
            shape=image_size,
            dtype=image_dtype,
            compressor=compressor,
            high=float("inf"),
        )
        cls._create_binary_file(
            root,
            labels_file,
            num_samples=num_samples,
            shape=label_size,
            dtype=label_dtype,
            compressor=compressor,
            high=num_categories,
        )

        return num_samples


@DATASET_MOCKS.set_from_named_callable
def mnist(info, root, config):
    train = config.split == "train"
    images_file = f"{'train' if train else 't10k'}-images-idx3-ubyte.gz"
    labels_file = f"{'train' if train else 't10k'}-labels-idx1-ubyte.gz"
    return MNISTMockData.generate(
        root,
        num_categories=len(info.categories),
        images_file=images_file,
        labels_file=labels_file,
    )


DATASET_MOCKS.update({name: DatasetMock(name, mnist) for name in ["fashionmnist", "kmnist"]})


@DATASET_MOCKS.set_from_named_callable
def emnist(info, root, _):
    # The image sets that merge some lower case letters in their respective upper case variant, still use dense
    # labels in the data files. Thus, num_categories != len(categories) there.
    num_categories = defaultdict(
        lambda: len(info.categories), {image_set: 47 for image_set in ("Balanced", "By_Merge")}
    )

    mock_infos = {}
    file_names = set()
    for config in info._configs:
        prefix = f"emnist-{config.image_set.replace('_', '').lower()}-{config.split}"
        images_file = f"{prefix}-images-idx3-ubyte.gz"
        labels_file = f"{prefix}-labels-idx1-ubyte.gz"
        file_names.update({images_file, labels_file})
        mock_infos[config] = dict(
            num_samples=MNISTMockData.generate(
                root,
                num_categories=num_categories[config.image_set],
                images_file=images_file,
                labels_file=labels_file,
            )
        )

    make_zip(root, "emnist-gzip.zip", *file_names)

    return mock_infos


@DATASET_MOCKS.set_from_named_callable
def qmnist(info, root, config):
    num_categories = len(info.categories)
    if config.split == "train":
        num_samples = num_samples_gen = num_categories + 2
        prefix = "qmnist-train"
        suffix = ".gz"
        compressor = gzip.open
        mock_infos = num_samples
    elif config.split.startswith("test"):
        # The split 'test50k' is defined as the last 50k images beginning at index 10000. Thus, we need to create
        # more than 10000 images for the dataset to not be empty.
        num_samples_gen = 10001
        prefix = "qmnist-test"
        suffix = ".gz"
        compressor = gzip.open
        mock_infos = {
            info.make_config(split="test"): num_samples_gen,
            info.make_config(split="test10k"): min(num_samples_gen, 10_000),
            info.make_config(split="test50k"): num_samples_gen - 10_000,
        }
    else:  # config.split == "nist"
        num_samples = num_samples_gen = num_categories + 3
        prefix = "xnist"
        suffix = ".xz"
        compressor = lzma.open
        mock_infos = num_samples

    MNISTMockData.generate(
        root,
        num_categories=num_categories,
        num_samples=num_samples_gen,
        images_file=f"{prefix}-images-idx3-ubyte{suffix}",
        labels_file=f"{prefix}-labels-idx2-int{suffix}",
        label_size=(8,),
        label_dtype=torch.int32,
        compressor=compressor,
    )
    return mock_infos


class CIFARMockData:
    NUM_PIXELS = 32 * 32 * 3

    @classmethod
    def _create_batch_file(cls, root, name, *, num_categories, labels_key, num_samples=1):
        content = {
            "data": make_tensor((num_samples, cls.NUM_PIXELS), dtype=torch.uint8).numpy(),
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
                num_categories=num_categories,
                labels_key=labels_key,
            )

        make_tar(root, name, folder, compression="gz")


@DATASET_MOCKS.set_from_named_callable
def cifar10(info, root, config):
    train_files = [f"data_batch_{idx}" for idx in range(1, 6)]
    test_files = ["test_batch"]

    CIFARMockData.generate(
        root=root,
        name="cifar-10-python.tar.gz",
        folder=pathlib.Path("cifar-10-batches-py"),
        train_files=train_files,
        test_files=test_files,
        num_categories=10,
        labels_key="labels",
    )

    return len(train_files if config.split == "train" else test_files)


@DATASET_MOCKS.set_from_named_callable
def cifar100(info, root, config):
    train_files = ["train"]
    test_files = ["test"]

    CIFARMockData.generate(
        root=root,
        name="cifar-100-python.tar.gz",
        folder=pathlib.Path("cifar-100-python"),
        train_files=train_files,
        test_files=test_files,
        num_categories=100,
        labels_key="fine_labels",
    )

    return len(train_files if config.split == "train" else test_files)


@DATASET_MOCKS.set_from_named_callable
def caltech101(info, root, config):
    def create_ann_file(root, name):
        import scipy.io

        box_coord = make_tensor((1, 4), dtype=torch.int32, low=0).numpy().astype(np.uint16)
        obj_contour = make_tensor((2, int(torch.randint(3, 6, size=()))), dtype=torch.float64, low=0).numpy()

        scipy.io.savemat(str(pathlib.Path(root) / name), dict(box_coord=box_coord, obj_contour=obj_contour))

    def create_ann_folder(root, name, file_name_fn, num_examples):
        root = pathlib.Path(root) / name
        root.mkdir(parents=True)

        for idx in range(num_examples):
            create_ann_file(root, file_name_fn(idx))

    images_root = root / "101_ObjectCategories"
    anns_root = root / "Annotations"

    ann_category_map = {
        "Faces_2": "Faces",
        "Faces_3": "Faces_easy",
        "Motorbikes_16": "Motorbikes",
        "Airplanes_Side_2": "airplanes",
    }

    num_images_per_category = 2
    for category in info.categories:
        create_image_folder(
            root=images_root,
            name=category,
            file_name_fn=lambda idx: f"image_{idx + 1:04d}.jpg",
            num_examples=num_images_per_category,
        )
        create_ann_folder(
            root=anns_root,
            name=ann_category_map.get(category, category),
            file_name_fn=lambda idx: f"annotation_{idx + 1:04d}.mat",
            num_examples=num_images_per_category,
        )

    (images_root / "BACKGROUND_Goodle").mkdir()
    make_tar(root, f"{images_root.name}.tar.gz", images_root, compression="gz")

    make_tar(root, f"{anns_root.name}.tar", anns_root)

    return num_images_per_category * len(info.categories)


@DATASET_MOCKS.set_from_named_callable
def caltech256(info, root, config):
    dir = root / "256_ObjectCategories"
    num_images_per_category = 2

    for idx, category in enumerate(info.categories, 1):
        files = create_image_folder(
            dir,
            name=f"{idx:03d}.{category}",
            file_name_fn=lambda image_idx: f"{idx:03d}_{image_idx + 1:04d}.jpg",
            num_examples=num_images_per_category,
        )
        if category == "spider":
            open(files[0].parent / "RENAME2", "w").close()

    make_tar(root, f"{dir.name}.tar", dir)

    return num_images_per_category * len(info.categories)


@DATASET_MOCKS.set_from_named_callable
def imagenet(info, root, config):
    wnids = tuple(info.extra.wnid_to_category.keys())
    if config.split == "train":
        images_root = root / "ILSVRC2012_img_train"

        num_samples = len(wnids)

        for wnid in wnids:
            files = create_image_folder(
                root=images_root,
                name=wnid,
                file_name_fn=lambda image_idx: f"{wnid}_{image_idx:04d}.JPEG",
                num_examples=1,
            )
            make_tar(images_root, f"{wnid}.tar", files[0].parent)
    elif config.split == "val":
        num_samples = 3
        files = create_image_folder(
            root=root,
            name="ILSVRC2012_img_val",
            file_name_fn=lambda image_idx: f"ILSVRC2012_val_{image_idx + 1:08d}.JPEG",
            num_examples=num_samples,
        )
        images_root = files[0].parent
    else:  # config.split == "test"
        images_root = root / "ILSVRC2012_img_test_v10102019"

        num_samples = 3

        create_image_folder(
            root=images_root,
            name="test",
            file_name_fn=lambda image_idx: f"ILSVRC2012_test_{image_idx + 1:08d}.JPEG",
            num_examples=num_samples,
        )
    make_tar(root, f"{images_root.name}.tar", images_root)

    devkit_root = root / "ILSVRC2012_devkit_t12"
    devkit_root.mkdir()
    data_root = devkit_root / "data"
    data_root.mkdir()
    with open(data_root / "ILSVRC2012_validation_ground_truth.txt", "w") as file:
        for label in torch.randint(0, len(wnids), (num_samples,)).tolist():
            file.write(f"{label}\n")
    make_tar(root, f"{devkit_root}.tar.gz", devkit_root, compression="gz")

    return num_samples


class CocoMockData:
    @classmethod
    def _make_images_archive(cls, root, name, *, num_samples):
        image_paths = create_image_folder(
            root, name, file_name_fn=lambda idx: f"{idx:012d}.jpg", num_examples=num_samples
        )

        images_meta = []
        for path in image_paths:
            with PIL.Image.open(path) as image:
                width, height = image.size
            images_meta.append(dict(file_name=path.name, id=int(path.stem), width=width, height=height))

        make_zip(root, f"{name}.zip")

        return images_meta

    @classmethod
    def _make_annotations_json(
        cls,
        root,
        name,
        *,
        images_meta,
        fn,
    ):
        num_anns_per_image = torch.randint(1, 5, (len(images_meta),))
        num_anns_total = int(num_anns_per_image.sum())
        ann_ids_iter = iter(torch.arange(num_anns_total)[torch.randperm(num_anns_total)])

        anns_meta = []
        for image_meta, num_anns in zip(images_meta, num_anns_per_image):
            for _ in range(num_anns):
                ann_id = int(next(ann_ids_iter))
                anns_meta.append(dict(fn(ann_id, image_meta), id=ann_id, image_id=image_meta["id"]))
        anns_meta.sort(key=lambda ann: ann["id"])

        with open(root / name, "w") as file:
            json.dump(dict(images=images_meta, annotations=anns_meta), file)

        return num_anns_per_image

    @staticmethod
    def _make_instances_data(ann_id, image_meta):
        def make_rle_segmentation():
            height, width = image_meta["height"], image_meta["width"]
            numel = height * width
            counts = []
            while sum(counts) <= numel:
                counts.append(int(torch.randint(5, 8, ())))
            if sum(counts) > numel:
                counts[-1] -= sum(counts) - numel
            return dict(counts=counts, size=[height, width])

        return dict(
            segmentation=make_rle_segmentation(),
            bbox=make_tensor((4,), dtype=torch.float32, low=0).tolist(),
            iscrowd=True,
            area=float(make_scalar(dtype=torch.float32)),
            category_id=int(make_scalar(dtype=torch.int64)),
        )

    @staticmethod
    def _make_captions_data(ann_id, image_meta):
        return dict(caption=f"Caption {ann_id} describing image {image_meta['id']}.")

    @classmethod
    def _make_annotations(cls, root, name, *, images_meta):
        num_anns_per_image = torch.zeros((len(images_meta),), dtype=torch.int64)
        for annotations, fn in (
            ("instances", cls._make_instances_data),
            ("captions", cls._make_captions_data),
        ):
            num_anns_per_image += cls._make_annotations_json(
                root, f"{annotations}_{name}.json", images_meta=images_meta, fn=fn
            )

        return int(num_anns_per_image.sum())

    @classmethod
    def generate(
        cls,
        root,
        *,
        year,
        num_samples,
    ):
        annotations_dir = root / "annotations"
        annotations_dir.mkdir()

        for split in ("train", "val"):
            config_name = f"{split}{year}"

            images_meta = cls._make_images_archive(root, config_name, num_samples=num_samples)
            cls._make_annotations(
                annotations_dir,
                config_name,
                images_meta=images_meta,
            )

        make_zip(root, f"annotations_trainval{year}.zip", annotations_dir)

        return num_samples


@DATASET_MOCKS.set_from_named_callable
def coco(info, root, config):
    return dict(
        zip(
            [config_ for config_ in info._configs if config_.year == config.year],
            itertools.repeat(CocoMockData.generate(root, year=config.year, num_samples=5)),
        )
    )


class SBDMockData:
    _NUM_CATEGORIES = 20

    @classmethod
    def _make_split_files(cls, root_map):
        ids_map = {
            split: [f"2008_{idx:06d}" for idx in idcs]
            for split, idcs in (
                ("train", [0, 1, 2]),
                ("train_noval", [0, 2]),
                ("val", [3]),
            )
        }

        for split, ids in ids_map.items():
            with open(root_map[split] / f"{split}.txt", "w") as fh:
                fh.writelines(f"{id}\n" for id in ids)

        return sorted(set(itertools.chain(*ids_map.values()))), {split: len(ids) for split, ids in ids_map.items()}

    @classmethod
    def _make_anns_folder(cls, root, name, ids):
        from scipy.io import savemat

        anns_folder = root / name
        anns_folder.mkdir()

        sizes = torch.randint(1, 9, size=(len(ids), 2)).tolist()
        for id, size in zip(ids, sizes):
            savemat(
                anns_folder / f"{id}.mat",
                {
                    "GTcls": {
                        "Boundaries": cls._make_boundaries(size),
                        "Segmentation": cls._make_segmentation(size),
                    }
                },
            )
        return sizes

    @classmethod
    def _make_boundaries(cls, size):
        from scipy.sparse import csc_matrix

        return [
            [csc_matrix(torch.randint(0, 2, size=size, dtype=torch.uint8).numpy())] for _ in range(cls._NUM_CATEGORIES)
        ]

    @classmethod
    def _make_segmentation(cls, size):
        return torch.randint(0, cls._NUM_CATEGORIES + 1, size=size, dtype=torch.uint8).numpy()

    @classmethod
    def generate(cls, root):
        archive_folder = root / "benchmark_RELEASE"
        dataset_folder = archive_folder / "dataset"
        dataset_folder.mkdir(parents=True, exist_ok=True)

        ids, num_samples_map = cls._make_split_files(defaultdict(lambda: dataset_folder, {"train_noval": root}))
        sizes = cls._make_anns_folder(dataset_folder, "cls", ids)
        create_image_folder(
            dataset_folder, "img", lambda idx: f"{ids[idx]}.jpg", num_examples=len(ids), size=lambda idx: sizes[idx]
        )

        make_tar(root, "benchmark.tgz", archive_folder, compression="gz")

        return num_samples_map


@DATASET_MOCKS.set_from_named_callable
def sbd(info, root, _):
    num_samples_map = SBDMockData.generate(root)
    return {config: num_samples_map[config.split] for config in info._configs}


@DATASET_MOCKS.set_from_named_callable
def semeion(info, root, config):
    num_samples = 3

    images = torch.rand(num_samples, 256)
    labels = one_hot(torch.randint(len(info.categories), size=(num_samples,)))
    with open(root / "semeion.data", "w") as fh:
        for image, one_hot_label in zip(images, labels):
            image_columns = " ".join([f"{pixel.item():.4f}" for pixel in image])
            labels_columns = " ".join([str(label.item()) for label in one_hot_label])
            fh.write(f"{image_columns} {labels_columns}\n")

    return num_samples


class VOCMockData:
    _TRAIN_VAL_FILE_NAMES = {
        "2007": "VOCtrainval_06-Nov-2007.tar",
        "2008": "VOCtrainval_14-Jul-2008.tar",
        "2009": "VOCtrainval_11-May-2009.tar",
        "2010": "VOCtrainval_03-May-2010.tar",
        "2011": "VOCtrainval_25-May-2011.tar",
        "2012": "VOCtrainval_11-May-2012.tar",
    }
    _TEST_FILE_NAMES = {
        "2007": "VOCtest_06-Nov-2007.tar",
    }

    @classmethod
    def _make_split_files(cls, root, *, year, trainval):
        split_folder = root / "ImageSets"

        if trainval:
            idcs_map = {
                "train": [0, 1, 2],
                "val": [3, 4],
            }
            idcs_map["trainval"] = [*idcs_map["train"], *idcs_map["val"]]
        else:
            idcs_map = {
                "test": [5],
            }
        ids_map = {split: [f"{year}_{idx:06d}" for idx in idcs] for split, idcs in idcs_map.items()}

        for task_sub_folder in ("Main", "Segmentation"):
            task_folder = split_folder / task_sub_folder
            task_folder.mkdir(parents=True, exist_ok=True)
            for split, ids in ids_map.items():
                with open(task_folder / f"{split}.txt", "w") as fh:
                    fh.writelines(f"{id}\n" for id in ids)

        return sorted(set(itertools.chain(*ids_map.values()))), {split: len(ids) for split, ids in ids_map.items()}

    @classmethod
    def _make_detection_anns_folder(cls, root, name, *, file_name_fn, num_examples):
        folder = root / name
        folder.mkdir(parents=True, exist_ok=True)

        for idx in range(num_examples):
            cls._make_detection_ann_file(folder, file_name_fn(idx))

    @classmethod
    def _make_detection_ann_file(cls, root, name):
        def add_child(parent, name, text=None):
            child = ET.SubElement(parent, name)
            child.text = text
            return child

        def add_name(obj, name="dog"):
            add_child(obj, "name", name)
            return name

        def add_bndbox(obj, bndbox=None):
            if bndbox is None:
                bndbox = {"xmin": "1", "xmax": "2", "ymin": "3", "ymax": "4"}

            obj = add_child(obj, "bndbox")
            for name, text in bndbox.items():
                add_child(obj, name, text)

            return bndbox

        annotation = ET.Element("annotation")
        obj = add_child(annotation, "object")
        data = dict(name=add_name(obj), bndbox=add_bndbox(obj))

        with open(root / name, "wb") as fh:
            fh.write(ET.tostring(annotation))

        return data

    @classmethod
    def generate(cls, root, *, year, trainval):
        archive_folder = root
        if year == "2011":
            archive_folder /= "TrainVal"
        data_folder = archive_folder / "VOCdevkit" / f"VOC{year}"
        data_folder.mkdir(parents=True, exist_ok=True)

        ids, num_samples_map = cls._make_split_files(data_folder, year=year, trainval=trainval)
        for make_folder_fn, name, suffix in [
            (create_image_folder, "JPEGImages", ".jpg"),
            (create_image_folder, "SegmentationClass", ".png"),
            (cls._make_detection_anns_folder, "Annotations", ".xml"),
        ]:
            make_folder_fn(data_folder, name, file_name_fn=lambda idx: ids[idx] + suffix, num_examples=len(ids))
        make_tar(root, (cls._TRAIN_VAL_FILE_NAMES if trainval else cls._TEST_FILE_NAMES)[year], data_folder)

        return num_samples_map


@DATASET_MOCKS.set_from_named_callable
def voc(info, root, config):
    trainval = config.split != "test"
    num_samples_map = VOCMockData.generate(root, year=config.year, trainval=trainval)
    return {
        config_: num_samples_map[config_.split]
        for config_ in info._configs
        if config_.year == config.year and ((config_.split == "test") ^ trainval)
    }


class CelebAMockData:
    @classmethod
    def _make_ann_file(cls, root, name, data, *, field_names=None):
        with open(root / name, "w") as file:
            if field_names:
                file.write(f"{len(data)}\r\n")
                file.write(" ".join(field_names) + "\r\n")
            file.writelines(" ".join(str(item) for item in row) + "\r\n" for row in data)

    _SPLIT_TO_IDX = {
        "train": 0,
        "val": 1,
        "test": 2,
    }

    @classmethod
    def _make_split_file(cls, root):
        num_samples_map = {"train": 4, "val": 3, "test": 2}

        data = [
            (f"{idx:06d}.jpg", cls._SPLIT_TO_IDX[split])
            for split, num_samples in num_samples_map.items()
            for idx in range(num_samples)
        ]
        cls._make_ann_file(root, "list_eval_partition.txt", data)

        image_file_names, _ = zip(*data)
        return image_file_names, num_samples_map

    @classmethod
    def _make_identity_file(cls, root, image_file_names):
        cls._make_ann_file(
            root, "identity_CelebA.txt", [(name, int(make_scalar(low=1, dtype=torch.int))) for name in image_file_names]
        )

    @classmethod
    def _make_attributes_file(cls, root, image_file_names):
        field_names = ("5_o_Clock_Shadow", "Young")
        data = [
            [name, *[" 1" if attr else "-1" for attr in make_tensor((len(field_names),), dtype=torch.bool)]]
            for name in image_file_names
        ]
        cls._make_ann_file(root, "list_attr_celeba.txt", data, field_names=(*field_names, ""))

    @classmethod
    def _make_bounding_boxes_file(cls, root, image_file_names):
        field_names = ("image_id", "x_1", "y_1", "width", "height")
        data = [
            [f"{name}  ", *[f"{coord:3d}" for coord in make_tensor((4,), low=0, dtype=torch.int).tolist()]]
            for name in image_file_names
        ]
        cls._make_ann_file(root, "list_bbox_celeba.txt", data, field_names=field_names)

    @classmethod
    def _make_landmarks_file(cls, root, image_file_names):
        field_names = ("lefteye_x", "lefteye_y", "rightmouth_x", "rightmouth_y")
        data = [
            [
                name,
                *[
                    f"{coord:4d}" if idx else coord
                    for idx, coord in enumerate(make_tensor((len(field_names),), low=0, dtype=torch.int).tolist())
                ],
            ]
            for name in image_file_names
        ]
        cls._make_ann_file(root, "list_landmarks_align_celeba.txt", data, field_names=field_names)

    @classmethod
    def generate(cls, root):
        image_file_names, num_samples_map = cls._make_split_file(root)

        image_files = create_image_folder(
            root, "img_align_celeba", file_name_fn=lambda idx: image_file_names[idx], num_examples=len(image_file_names)
        )
        make_zip(root, image_files[0].parent.with_suffix(".zip").name)

        for make_ann_file_fn in (
            cls._make_identity_file,
            cls._make_attributes_file,
            cls._make_bounding_boxes_file,
            cls._make_landmarks_file,
        ):
            make_ann_file_fn(root, image_file_names)

        return num_samples_map


@DATASET_MOCKS.set_from_named_callable
def celeba(info, root, _):
    num_samples_map = CelebAMockData.generate(root)
    return {config: num_samples_map[config.split] for config in info._configs}


@DATASET_MOCKS.set_from_named_callable
def dtd(info, root, _):
    data_folder = root / "dtd"

    num_images_per_class = 3
    image_folder = data_folder / "images"
    categories = {"banded", "marbled", "zigzagged"}
    image_ids_per_category = {
        category: [
            str(path.relative_to(path.parents[1]).as_posix())
            for path in create_image_folder(
                image_folder,
                category,
                file_name_fn=lambda idx: f"{category}_{idx:04d}.jpg",
                num_examples=num_images_per_class,
            )
        ]
        for category in categories
    }

    meta_folder = data_folder / "labels"
    meta_folder.mkdir()

    with open(meta_folder / "labels_joint_anno.txt", "w") as file:
        for cls, image_ids in image_ids_per_category.items():
            for image_id in image_ids:
                joint_categories = random.choices(
                    list(categories - {cls}), k=int(torch.randint(len(categories) - 1, ()))
                )
                file.write(" ".join([image_id, *sorted([cls, *joint_categories])]) + "\n")

    image_ids = list(itertools.chain(*image_ids_per_category.values()))
    splits = ("train", "val", "test")
    num_samples_map = {}
    for fold in range(1, 11):
        random.shuffle(image_ids)
        for offset, split in enumerate(splits):
            image_ids_in_config = image_ids[offset :: len(splits)]
            with open(meta_folder / f"{split}{fold}.txt", "w") as file:
                file.write("\n".join(image_ids_in_config) + "\n")

            num_samples_map[info.make_config(split=split, fold=str(fold))] = len(image_ids_in_config)

    make_tar(root, "dtd-r1.0.1.tar.gz", data_folder, compression="gz")

    return num_samples_map


@DATASET_MOCKS.set_from_named_callable
def fer2013(info, root, config):
    num_samples = 5 if config.split == "train" else 3

    path = root / f"{config.split}.csv"
    with open(path, "w", newline="") as file:
        field_names = ["emotion"] if config.split == "train" else []
        field_names.append("pixels")

        file.write(",".join(field_names) + "\n")

        writer = csv.DictWriter(file, fieldnames=field_names, quotechar='"', quoting=csv.QUOTE_NONNUMERIC)
        for _ in range(num_samples):
            rowdict = {
                "pixels": " ".join([str(int(pixel)) for pixel in torch.randint(256, (48 * 48,), dtype=torch.uint8)])
            }
            if config.split == "train":
                rowdict["emotion"] = int(torch.randint(7, ()))
            writer.writerow(rowdict)

    make_zip(root, f"{path.name}.zip", path)

    return num_samples


@DATASET_MOCKS.set_from_named_callable
def clevr(info, root, config):
    data_folder = root / "CLEVR_v1.0"

    num_samples_map = {
        "train": 3,
        "val": 2,
        "test": 1,
    }

    images_folder = data_folder / "images"
    image_files = {
        split: create_image_folder(
            images_folder,
            split,
            file_name_fn=lambda idx: f"CLEVR_{split}_{idx:06d}.jpg",
            num_examples=num_samples,
        )
        for split, num_samples in num_samples_map.items()
    }

    scenes_folder = data_folder / "scenes"
    scenes_folder.mkdir()
    for split in ["train", "val"]:
        with open(scenes_folder / f"CLEVR_{split}_scenes.json", "w") as file:
            json.dump(
                {
                    "scenes": [
                        {
                            "image_filename": image_file.name,
                            # We currently only return the number of objects in a scene.
                            # Thus, it is sufficient for now to only mock the number of elements.
                            "objects": [None] * int(torch.randint(1, 5, ())),
                        }
                        for image_file in image_files[split]
                    ]
                },
                file,
            )

    make_zip(root, f"{data_folder.name}.zip", data_folder)

    return {config_: num_samples_map[config_.split] for config_ in info._configs}


class OxfordIIITPetMockData:
    @classmethod
    def _meta_to_split_and_classification_ann(cls, meta, idx):
        image_id = "_".join(
            [
                *[(str.title if meta["species"] == "cat" else str.lower)(part) for part in meta["cls"].split()],
                str(idx),
            ]
        )
        class_id = str(meta["label"] + 1)
        species = "1" if meta["species"] == "cat" else "2"
        breed_id = "-1"
        return (image_id, class_id, species, breed_id)

    @classmethod
    def generate(self, root):
        classification_anns_meta = (
            dict(cls="Abyssinian", label=0, species="cat"),
            dict(cls="Keeshond", label=18, species="dog"),
            dict(cls="Yorkshire Terrier", label=36, species="dog"),
        )
        split_and_classification_anns = [
            self._meta_to_split_and_classification_ann(meta, idx)
            for meta, idx in itertools.product(classification_anns_meta, (1, 2, 10))
        ]
        image_ids, *_ = zip(*split_and_classification_anns)

        image_files = create_image_folder(
            root, "images", file_name_fn=lambda idx: f"{image_ids[idx]}.jpg", num_examples=len(image_ids)
        )

        anns_folder = root / "annotations"
        anns_folder.mkdir()
        random.shuffle(split_and_classification_anns)
        splits = ("trainval", "test")
        num_samples_map = {}
        for offset, split in enumerate(splits):
            split_and_classification_anns_in_split = split_and_classification_anns[offset :: len(splits)]
            with open(anns_folder / f"{split}.txt", "w") as file:
                writer = csv.writer(file, delimiter=" ")
                for split_and_classification_ann in split_and_classification_anns_in_split:
                    writer.writerow(split_and_classification_ann)

            num_samples_map[split] = len(split_and_classification_anns_in_split)

        segmentation_files = create_image_folder(
            anns_folder, "trimaps", file_name_fn=lambda idx: f"{image_ids[idx]}.png", num_examples=len(image_ids)
        )

        # The dataset has some rogue files
        for path in image_files[:3]:
            path.with_suffix(".mat").touch()
        for path in segmentation_files:
            path.with_name(f".{path.name}").touch()

        make_tar(root, "images.tar.gz", compression="gz")
        make_tar(root, anns_folder.with_suffix(".tar.gz").name, compression="gz")

        return num_samples_map


@DATASET_MOCKS.set_from_named_callable
def oxford_iiit_pet(info, root, config):
    num_samples_map = OxfordIIITPetMockData.generate(root)
    return {config_: num_samples_map[config_.split] for config_ in info._configs}


class _CUB200MockData:
    @classmethod
    def _category_folder(cls, category, idx):
        return f"{idx:03d}.{category}"

    @classmethod
    def _file_stem(cls, category, idx):
        return f"{category}_{idx:04d}"

    @classmethod
    def _make_images(cls, images_folder):
        image_files = []
        for category_idx, category in [
            (1, "Black_footed_Albatross"),
            (100, "Brown_Pelican"),
            (200, "Common_Yellowthroat"),
        ]:
            image_files.extend(
                create_image_folder(
                    images_folder,
                    cls._category_folder(category, category_idx),
                    lambda image_idx: f"{cls._file_stem(category, image_idx)}.jpg",
                    num_examples=5,
                )
            )

        return image_files


class CUB2002011MockData(_CUB200MockData):
    @classmethod
    def _make_archive(cls, root):
        archive_folder = root / "CUB_200_2011"

        images_folder = archive_folder / "images"
        image_files = cls._make_images(images_folder)
        image_ids = list(range(1, len(image_files) + 1))

        with open(archive_folder / "images.txt", "w") as file:
            file.write(
                "\n".join(
                    f"{id} {path.relative_to(images_folder).as_posix()}" for id, path in zip(image_ids, image_files)
                )
            )

        split_ids = torch.randint(2, (len(image_ids),)).tolist()
        counts = Counter(split_ids)
        num_samples_map = {"train": counts[1], "test": counts[0]}
        with open(archive_folder / "train_test_split.txt", "w") as file:
            file.write("\n".join(f"{image_id} {split_id}" for image_id, split_id in zip(image_ids, split_ids)))

        with open(archive_folder / "bounding_boxes.txt", "w") as file:
            file.write(
                "\n".join(
                    " ".join(
                        str(item)
                        for item in [image_id, *make_tensor((4,), dtype=torch.int, low=0).to(torch.float).tolist()]
                    )
                    for image_id in image_ids
                )
            )

        make_tar(root, archive_folder.with_suffix(".tgz").name, compression="gz")

        return image_files, num_samples_map

    @classmethod
    def _make_segmentations(cls, root, image_files):
        segmentations_folder = root / "segmentations"
        for image_file in image_files:
            folder = segmentations_folder.joinpath(image_file.relative_to(image_file.parents[1]))
            folder.mkdir(exist_ok=True, parents=True)
            create_image_file(
                folder,
                image_file.with_suffix(".png").name,
                size=[1, *make_tensor((2,), low=3, dtype=torch.int).tolist()],
            )

        make_tar(root, segmentations_folder.with_suffix(".tgz").name, compression="gz")

    @classmethod
    def generate(cls, root):
        image_files, num_samples_map = cls._make_archive(root)
        cls._make_segmentations(root, image_files)
        return num_samples_map


class CUB2002010MockData(_CUB200MockData):
    @classmethod
    def _make_hidden_rouge_file(cls, *files):
        for file in files:
            (file.parent / f"._{file.name}").touch()

    @classmethod
    def _make_splits(cls, root, image_files):
        split_folder = root / "lists"
        split_folder.mkdir()
        random.shuffle(image_files)
        splits = ("train", "test")
        num_samples_map = {}
        for offset, split in enumerate(splits):
            image_files_in_split = image_files[offset :: len(splits)]

            split_file = split_folder / f"{split}.txt"
            with open(split_file, "w") as file:
                file.write(
                    "\n".join(
                        sorted(
                            str(image_file.relative_to(image_file.parents[1]).as_posix())
                            for image_file in image_files_in_split
                        )
                    )
                )

            cls._make_hidden_rouge_file(split_file)
            num_samples_map[split] = len(image_files_in_split)

        make_tar(root, split_folder.with_suffix(".tgz").name, compression="gz")

        return num_samples_map

    @classmethod
    def _make_anns(cls, root, image_files):
        from scipy.io import savemat

        anns_folder = root / "annotations-mat"
        for image_file in image_files:
            ann_file = anns_folder / image_file.with_suffix(".mat").relative_to(image_file.parents[1])
            ann_file.parent.mkdir(parents=True, exist_ok=True)

            savemat(
                ann_file,
                {
                    "seg": torch.randint(
                        256, make_tensor((2,), low=3, dtype=torch.int).tolist(), dtype=torch.uint8
                    ).numpy(),
                    "bbox": dict(
                        zip(("left", "top", "right", "bottom"), make_tensor((4,), dtype=torch.uint8).tolist())
                    ),
                },
            )

        readme_file = anns_folder / "README.txt"
        readme_file.touch()
        cls._make_hidden_rouge_file(readme_file)

        make_tar(root, "annotations.tgz", anns_folder, compression="gz")

    @classmethod
    def generate(cls, root):
        images_folder = root / "images"
        image_files = cls._make_images(images_folder)
        cls._make_hidden_rouge_file(*image_files)
        make_tar(root, images_folder.with_suffix(".tgz").name, compression="gz")

        num_samples_map = cls._make_splits(root, image_files)
        cls._make_anns(root, image_files)

        return num_samples_map


@DATASET_MOCKS.set_from_named_callable
def cub200(info, root, config):
    num_samples_map = (CUB2002011MockData if config.year == "2011" else CUB2002010MockData).generate(root)
    return {config_: num_samples_map[config_.split] for config_ in info._configs if config_.year == config.year}
