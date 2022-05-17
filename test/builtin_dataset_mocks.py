import bz2
import collections.abc
import csv
import functools
import gzip
import io
import itertools
import json
import lzma
import pathlib
import pickle
import random
import shutil
import unittest.mock
import warnings
import xml.etree.ElementTree as ET
from collections import defaultdict, Counter

import numpy as np
import pytest
import torch
from datasets_utils import make_zip, make_tar, create_image_folder, create_image_file, combinations_grid
from torch.nn.functional import one_hot
from torch.testing import make_tensor as _make_tensor
from torchvision.prototype import datasets

make_tensor = functools.partial(_make_tensor, device="cpu")
make_scalar = functools.partial(make_tensor, ())


__all__ = ["DATASET_MOCKS", "parametrize_dataset_mocks"]


class DatasetMock:
    def __init__(self, name, *, mock_data_fn, configs):
        # FIXME: error handling for unknown names
        self.name = name
        self.mock_data_fn = mock_data_fn
        self.configs = configs

    def _parse_mock_info(self, mock_info):
        if mock_info is None:
            raise pytest.UsageError(
                f"The mock data function for dataset '{self.name}' returned nothing. It needs to at least return an "
                f"integer indicating the number of samples for the current `config`."
            )
        elif isinstance(mock_info, int):
            mock_info = dict(num_samples=mock_info)
        elif not isinstance(mock_info, dict):
            raise pytest.UsageError(
                f"The mock data function for dataset '{self.name}' returned a {type(mock_info)}. The returned object "
                f"should be a dictionary containing at least the number of samples for the key `'num_samples'`. If no "
                f"additional information is required for specific tests, the number of samples can also be returned as "
                f"an integer."
            )
        elif "num_samples" not in mock_info:
            raise pytest.UsageError(
                f"The dictionary returned by the mock data function for dataset '{self.name}' has to contain a "
                f"`'num_samples'` entry indicating the number of samples."
            )

        return mock_info

    def load(self, config):
        # `datasets.home()` is patched to a temporary directory through the autouse fixture `test_home` in
        # test/test_prototype_builtin_datasets.py
        root = pathlib.Path(datasets.home()) / self.name
        # We cannot place the mock data upfront in `root`. Loading a dataset calls `OnlineResource.load`. In turn,
        # this will only download **and** preprocess if the file is not present. In other words, if we already place
        # the file in `root` before the resource is loaded, we are effectively skipping the preprocessing.
        # To avoid that we first place the mock data in a temporary directory and patch the download logic to move it to
        # `root` only when it is requested.
        tmp_mock_data_folder = root / "__mock__"
        tmp_mock_data_folder.mkdir(parents=True)

        mock_info = self._parse_mock_info(self.mock_data_fn(tmp_mock_data_folder, config))

        def patched_download(resource, root, **kwargs):
            src = tmp_mock_data_folder / resource.file_name
            if not src.exists():
                raise pytest.UsageError(
                    f"Dataset '{self.name}' requires the file {resource.file_name} for {config}"
                    f"but it was not created by the mock data function."
                )

            dst = root / resource.file_name
            shutil.move(str(src), str(root))

            return dst

        with unittest.mock.patch(
            "torchvision.prototype.datasets.utils._resource.OnlineResource.download", new=patched_download
        ):
            dataset = datasets.load(self.name, **config)

        extra_files = list(tmp_mock_data_folder.glob("**/*"))
        if extra_files:
            raise pytest.UsageError(
                (
                    f"Dataset '{self.name}' created the following files for {config} in the mock data function, "
                    f"but they were not loaded:\n\n"
                )
                + "\n".join(str(file.relative_to(tmp_mock_data_folder)) for file in extra_files)
            )

        tmp_mock_data_folder.rmdir()

        return dataset, mock_info


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


DATASET_MOCKS = {}


def register_mock(name=None, *, configs):
    def wrapper(mock_data_fn):
        nonlocal name
        if name is None:
            name = mock_data_fn.__name__
        DATASET_MOCKS[name] = DatasetMock(name, mock_data_fn=mock_data_fn, configs=configs)

        return mock_data_fn

    return wrapper


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


def mnist(root, config):
    prefix = "train" if config["split"] == "train" else "t10k"
    return MNISTMockData.generate(
        root,
        num_categories=10,
        images_file=f"{prefix}-images-idx3-ubyte.gz",
        labels_file=f"{prefix}-labels-idx1-ubyte.gz",
    )


DATASET_MOCKS.update(
    {
        name: DatasetMock(name, mock_data_fn=mnist, configs=combinations_grid(split=("train", "test")))
        for name in ["mnist", "fashionmnist", "kmnist"]
    }
)


@register_mock(
    configs=combinations_grid(
        split=("train", "test"),
        image_set=("Balanced", "By_Merge", "By_Class", "Letters", "Digits", "MNIST"),
    )
)
def emnist(root, config):
    num_samples_map = {}
    file_names = set()
    for split, image_set in itertools.product(
        ("train", "test"),
        ("Balanced", "By_Merge", "By_Class", "Letters", "Digits", "MNIST"),
    ):
        prefix = f"emnist-{image_set.replace('_', '').lower()}-{split}"
        images_file = f"{prefix}-images-idx3-ubyte.gz"
        labels_file = f"{prefix}-labels-idx1-ubyte.gz"
        file_names.update({images_file, labels_file})
        num_samples_map[(split, image_set)] = MNISTMockData.generate(
            root,
            # The image sets that merge some lower case letters in their respective upper case variant, still use dense
            # labels in the data files. Thus, num_categories != len(categories) there.
            num_categories=47 if config["image_set"] in ("Balanced", "By_Merge") else 62,
            images_file=images_file,
            labels_file=labels_file,
        )

    make_zip(root, "emnist-gzip.zip", *file_names)

    return num_samples_map[(config["split"], config["image_set"])]


@register_mock(configs=combinations_grid(split=("train", "test", "test10k", "test50k", "nist")))
def qmnist(root, config):
    num_categories = 10
    if config["split"] == "train":
        num_samples = num_samples_gen = num_categories + 2
        prefix = "qmnist-train"
        suffix = ".gz"
        compressor = gzip.open
    elif config["split"].startswith("test"):
        # The split 'test50k' is defined as the last 50k images beginning at index 10000. Thus, we need to create
        # more than 10000 images for the dataset to not be empty.
        num_samples_gen = 10001
        num_samples = {
            "test": num_samples_gen,
            "test10k": min(num_samples_gen, 10_000),
            "test50k": num_samples_gen - 10_000,
        }[config["split"]]
        prefix = "qmnist-test"
        suffix = ".gz"
        compressor = gzip.open
    else:  # config["split"] == "nist"
        num_samples = num_samples_gen = num_categories + 3
        prefix = "xnist"
        suffix = ".xz"
        compressor = lzma.open

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
    return num_samples


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


@register_mock(configs=combinations_grid(split=("train", "test")))
def cifar10(root, config):
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

    return len(train_files if config["split"] == "train" else test_files)


@register_mock(configs=combinations_grid(split=("train", "test")))
def cifar100(root, config):
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

    return len(train_files if config["split"] == "train" else test_files)


@register_mock(configs=[dict()])
def caltech101(root, config):
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

    image_category_map = {
        "Faces": "Faces_2",
        "Faces_easy": "Faces_3",
        "Motorbikes": "Motorbikes_16",
        "airplanes": "Airplanes_Side_2",
    }

    categories = ["Faces", "Faces_easy", "Motorbikes", "airplanes", "yin_yang"]

    num_images_per_category = 2
    for category in categories:
        create_image_folder(
            root=images_root,
            name=category,
            file_name_fn=lambda idx: f"image_{idx + 1:04d}.jpg",
            num_examples=num_images_per_category,
        )
        create_ann_folder(
            root=anns_root,
            name=image_category_map.get(category, category),
            file_name_fn=lambda idx: f"annotation_{idx + 1:04d}.mat",
            num_examples=num_images_per_category,
        )

    (images_root / "BACKGROUND_Goodle").mkdir()
    make_tar(root, f"{images_root.name}.tar.gz", images_root, compression="gz")

    make_tar(root, f"{anns_root.name}.tar", anns_root)

    return num_images_per_category * len(categories)


@register_mock(configs=[dict()])
def caltech256(root, config):
    dir = root / "256_ObjectCategories"
    num_images_per_category = 2

    categories = [
        (1, "ak47"),
        (127, "laptop-101"),
        (198, "spider"),
        (257, "clutter"),
    ]

    for category_idx, category in categories:
        files = create_image_folder(
            dir,
            name=f"{category_idx:03d}.{category}",
            file_name_fn=lambda image_idx: f"{category_idx:03d}_{image_idx + 1:04d}.jpg",
            num_examples=num_images_per_category,
        )
        if category == "spider":
            open(files[0].parent / "RENAME2", "w").close()

    make_tar(root, f"{dir.name}.tar", dir)

    return num_images_per_category * len(categories)


@register_mock(configs=combinations_grid(split=("train", "val", "test")))
def imagenet(root, config):
    from scipy.io import savemat

    info = datasets.info("imagenet")

    if config["split"] == "train":
        num_samples = len(info["wnids"])
        archive_name = "ILSVRC2012_img_train.tar"

        files = []
        for wnid in info["wnids"]:
            create_image_folder(
                root=root,
                name=wnid,
                file_name_fn=lambda image_idx: f"{wnid}_{image_idx:04d}.JPEG",
                num_examples=1,
            )
            files.append(make_tar(root, f"{wnid}.tar"))
    elif config["split"] == "val":
        num_samples = 3
        archive_name = "ILSVRC2012_img_val.tar"
        files = [create_image_file(root, f"ILSVRC2012_val_{idx + 1:08d}.JPEG") for idx in range(num_samples)]

        devkit_root = root / "ILSVRC2012_devkit_t12"
        data_root = devkit_root / "data"
        data_root.mkdir(parents=True)

        with open(data_root / "ILSVRC2012_validation_ground_truth.txt", "w") as file:
            for label in torch.randint(0, len(info["wnids"]), (num_samples,)).tolist():
                file.write(f"{label}\n")

        num_children = 0
        synsets = [
            (idx, wnid, category, "", num_children, [], 0, 0)
            for idx, (category, wnid) in enumerate(zip(info["categories"], info["wnids"]), 1)
        ]
        num_children = 1
        synsets.extend((0, "", "", "", num_children, [], 0, 0) for _ in range(5))
        with warnings.catch_warnings():
            # The warning is not for savemat, but rather for some internals savemet is using
            warnings.filterwarnings("ignore", category=np.VisibleDeprecationWarning)
            savemat(data_root / "meta.mat", dict(synsets=synsets))

        make_tar(root, devkit_root.with_suffix(".tar.gz").name, compression="gz")
    else:  # config["split"] == "test"
        num_samples = 5
        archive_name = "ILSVRC2012_img_test_v10102019.tar"
        files = [create_image_file(root, f"ILSVRC2012_test_{idx + 1:08d}.JPEG") for idx in range(num_samples)]

    make_tar(root, archive_name, *files)

    return num_samples


class CocoMockData:
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
        split,
        year,
        num_samples,
    ):
        annotations_dir = root / "annotations"
        annotations_dir.mkdir()

        for split_ in ("train", "val"):
            config_name = f"{split_}{year}"

            images_meta = [
                dict(
                    file_name=f"{idx:012d}.jpg",
                    id=idx,
                    width=width,
                    height=height,
                )
                for idx, (height, width) in enumerate(
                    torch.randint(3, 11, size=(num_samples, 2), dtype=torch.int).tolist()
                )
            ]

            if split_ == split:
                create_image_folder(
                    root,
                    config_name,
                    file_name_fn=lambda idx: images_meta[idx]["file_name"],
                    num_examples=num_samples,
                    size=lambda idx: (3, images_meta[idx]["height"], images_meta[idx]["width"]),
                )
                make_zip(root, f"{config_name}.zip")

            cls._make_annotations(
                annotations_dir,
                config_name,
                images_meta=images_meta,
            )

        make_zip(root, f"annotations_trainval{year}.zip", annotations_dir)

        return num_samples


@register_mock(
    configs=combinations_grid(
        split=("train", "val"),
        year=("2017", "2014"),
        annotations=("instances", "captions", None),
    )
)
def coco(root, config):
    return CocoMockData.generate(root, split=config["split"], year=config["year"], num_samples=5)


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


@register_mock(configs=combinations_grid(split=("train", "val", "train_noval")))
def sbd(root, config):
    return SBDMockData.generate(root)[config["split"]]


@register_mock(configs=[dict()])
def semeion(root, config):
    num_samples = 3
    num_categories = 10

    images = torch.rand(num_samples, 256)
    labels = one_hot(torch.randint(num_categories, size=(num_samples,)), num_classes=num_categories)
    with open(root / "semeion.data", "w") as fh:
        for image, one_hot_label in zip(images, labels):
            image_columns = " ".join([f"{pixel.item():.4f}" for pixel in image])
            labels_columns = " ".join([str(label.item()) for label in one_hot_label])
            fh.write(f"{image_columns} {labels_columns} \n")

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
            child.text = str(text)
            return child

        def add_name(obj, name="dog"):
            add_child(obj, "name", name)

        def add_size(obj):
            obj = add_child(obj, "size")
            size = {"width": 0, "height": 0, "depth": 3}
            for name, text in size.items():
                add_child(obj, name, text)

        def add_bndbox(obj):
            obj = add_child(obj, "bndbox")
            bndbox = {"xmin": 1, "xmax": 2, "ymin": 3, "ymax": 4}
            for name, text in bndbox.items():
                add_child(obj, name, text)

        annotation = ET.Element("annotation")
        add_size(annotation)
        obj = add_child(annotation, "object")
        add_name(obj)
        add_bndbox(obj)

        with open(root / name, "wb") as fh:
            fh.write(ET.tostring(annotation))

    @classmethod
    def generate(cls, root, *, year, trainval):
        archive_folder = root
        if year == "2011":
            archive_folder = root / "TrainVal"
            data_folder = archive_folder / "VOCdevkit"
        else:
            archive_folder = data_folder = root / "VOCdevkit"
        data_folder = data_folder / f"VOC{year}"
        data_folder.mkdir(parents=True, exist_ok=True)

        ids, num_samples_map = cls._make_split_files(data_folder, year=year, trainval=trainval)
        for make_folder_fn, name, suffix in [
            (create_image_folder, "JPEGImages", ".jpg"),
            (create_image_folder, "SegmentationClass", ".png"),
            (cls._make_detection_anns_folder, "Annotations", ".xml"),
        ]:
            make_folder_fn(data_folder, name, file_name_fn=lambda idx: ids[idx] + suffix, num_examples=len(ids))
        make_tar(root, (cls._TRAIN_VAL_FILE_NAMES if trainval else cls._TEST_FILE_NAMES)[year], archive_folder)

        return num_samples_map


@register_mock(
    configs=[
        *combinations_grid(
            split=("train", "val", "trainval"),
            year=("2007", "2008", "2009", "2010", "2011", "2012"),
            task=("detection", "segmentation"),
        ),
        *combinations_grid(
            split=("test",),
            year=("2007",),
            task=("detection", "segmentation"),
        ),
    ],
)
def voc(root, config):
    trainval = config["split"] != "test"
    return VOCMockData.generate(root, year=config["year"], trainval=trainval)[config["split"]]


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


@register_mock(configs=combinations_grid(split=("train", "val", "test")))
def celeba(root, config):
    return CelebAMockData.generate(root)[config["split"]]


@register_mock(configs=combinations_grid(split=("train", "val", "test")))
def country211(root, config):
    split_folder = pathlib.Path(root, "country211", "valid" if config["split"] == "val" else config["split"])
    split_folder.mkdir(parents=True, exist_ok=True)

    num_examples = {
        "train": 3,
        "val": 4,
        "test": 5,
    }[config["split"]]

    classes = ("AD", "BS", "GR")
    for cls in classes:
        create_image_folder(
            split_folder,
            name=cls,
            file_name_fn=lambda idx: f"{idx}.jpg",
            num_examples=num_examples,
        )
    make_tar(root, f"{split_folder.parent.name}.tgz", split_folder.parent, compression="gz")
    return num_examples * len(classes)


@register_mock(configs=combinations_grid(split=("train", "test")))
def food101(root, config):
    data_folder = root / "food-101"

    num_images_per_class = 3
    image_folder = data_folder / "images"
    categories = ["apple_pie", "baby_back_ribs", "waffles"]
    image_ids = []
    for category in categories:
        image_files = create_image_folder(
            image_folder,
            category,
            file_name_fn=lambda idx: f"{idx:04d}.jpg",
            num_examples=num_images_per_class,
        )
        image_ids.extend(path.relative_to(path.parents[1]).with_suffix("").as_posix() for path in image_files)

    meta_folder = data_folder / "meta"
    meta_folder.mkdir()

    with open(meta_folder / "classes.txt", "w") as file:
        for category in categories:
            file.write(f"{category}\n")

    splits = ["train", "test"]
    num_samples_map = {}
    for offset, split in enumerate(splits):
        image_ids_in_split = image_ids[offset :: len(splits)]
        num_samples_map[split] = len(image_ids_in_split)
        with open(meta_folder / f"{split}.txt", "w") as file:
            for image_id in image_ids_in_split:
                file.write(f"{image_id}\n")

    make_tar(root, f"{data_folder.name}.tar.gz", compression="gz")

    return num_samples_map[config["split"]]


@register_mock(configs=combinations_grid(split=("train", "val", "test"), fold=(1, 4, 10)))
def dtd(root, config):
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

            num_samples_map[(split, fold)] = len(image_ids_in_config)

    make_tar(root, "dtd-r1.0.1.tar.gz", data_folder, compression="gz")

    return num_samples_map[config["split"], config["fold"]]


@register_mock(configs=combinations_grid(split=("train", "test")))
def fer2013(root, config):
    split = config["split"]
    num_samples = 5 if split == "train" else 3

    path = root / f"{split}.csv"
    with open(path, "w", newline="") as file:
        field_names = ["emotion"] if split == "train" else []
        field_names.append("pixels")

        file.write(",".join(field_names) + "\n")

        writer = csv.DictWriter(file, fieldnames=field_names, quotechar='"', quoting=csv.QUOTE_NONNUMERIC)
        for _ in range(num_samples):
            rowdict = {
                "pixels": " ".join([str(int(pixel)) for pixel in torch.randint(256, (48 * 48,), dtype=torch.uint8)])
            }
            if split == "train":
                rowdict["emotion"] = int(torch.randint(7, ()))
            writer.writerow(rowdict)

    make_zip(root, f"{path.name}.zip", path)

    return num_samples


@register_mock(configs=combinations_grid(split=("train", "test")))
def gtsrb(root, config):
    num_examples_per_class = 5 if config["split"] == "train" else 3
    classes = ("00000", "00042", "00012")
    num_examples = num_examples_per_class * len(classes)

    csv_columns = ["Filename", "Width", "Height", "Roi.X1", "Roi.Y1", "Roi.X2", "Roi.Y2", "ClassId"]

    def _make_ann_file(path, num_examples, class_idx):
        if class_idx == "random":
            class_idx = torch.randint(1, len(classes) + 1, size=(1,)).item()

        with open(path, "w") as csv_file:
            writer = csv.DictWriter(csv_file, fieldnames=csv_columns, delimiter=";")
            writer.writeheader()
            for image_idx in range(num_examples):
                writer.writerow(
                    {
                        "Filename": f"{image_idx:05d}.ppm",
                        "Width": torch.randint(1, 100, size=()).item(),
                        "Height": torch.randint(1, 100, size=()).item(),
                        "Roi.X1": torch.randint(1, 100, size=()).item(),
                        "Roi.Y1": torch.randint(1, 100, size=()).item(),
                        "Roi.X2": torch.randint(1, 100, size=()).item(),
                        "Roi.Y2": torch.randint(1, 100, size=()).item(),
                        "ClassId": class_idx,
                    }
                )

    archive_folder = root / "GTSRB"

    if config["split"] == "train":
        train_folder = archive_folder / "Training"
        train_folder.mkdir(parents=True)

        for class_idx in classes:
            create_image_folder(
                train_folder,
                name=class_idx,
                file_name_fn=lambda image_idx: f"{class_idx}_{image_idx:05d}.ppm",
                num_examples=num_examples_per_class,
            )
            _make_ann_file(
                path=train_folder / class_idx / f"GT-{class_idx}.csv",
                num_examples=num_examples_per_class,
                class_idx=int(class_idx),
            )
        make_zip(root, "GTSRB-Training_fixed.zip", archive_folder)
    else:
        test_folder = archive_folder / "Final_Test"
        test_folder.mkdir(parents=True)

        create_image_folder(
            test_folder,
            name="Images",
            file_name_fn=lambda image_idx: f"{image_idx:05d}.ppm",
            num_examples=num_examples,
        )

        make_zip(root, "GTSRB_Final_Test_Images.zip", archive_folder)

        _make_ann_file(
            path=root / "GT-final_test.csv",
            num_examples=num_examples,
            class_idx="random",
        )

        make_zip(root, "GTSRB_Final_Test_GT.zip", "GT-final_test.csv")

    return num_examples


@register_mock(configs=combinations_grid(split=("train", "val", "test")))
def clevr(root, config):
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

    return num_samples_map[config["split"]]


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


@register_mock(name="oxford-iiit-pet", configs=combinations_grid(split=("trainval", "test")))
def oxford_iiit_pet(root, config):
    return OxfordIIITPetMockData.generate(root)[config["split"]]


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


@register_mock(configs=combinations_grid(split=("train", "test"), year=("2010", "2011")))
def cub200(root, config):
    num_samples_map = (CUB2002011MockData if config["year"] == "2011" else CUB2002010MockData).generate(root)
    return num_samples_map[config["split"]]


@register_mock(configs=[dict()])
def eurosat(root, config):
    data_folder = root / "2750"
    data_folder.mkdir(parents=True)

    num_examples_per_class = 3
    categories = ["AnnualCrop", "Forest"]
    for category in categories:
        create_image_folder(
            root=data_folder,
            name=category,
            file_name_fn=lambda idx: f"{category}_{idx + 1}.jpg",
            num_examples=num_examples_per_class,
        )
    make_zip(root, "EuroSAT.zip", data_folder)
    return len(categories) * num_examples_per_class


@register_mock(configs=combinations_grid(split=("train", "test", "extra")))
def svhn(root, config):
    import scipy.io as sio

    num_samples = {
        "train": 2,
        "test": 3,
        "extra": 4,
    }[config["split"]]

    sio.savemat(
        root / f"{config['split']}_32x32.mat",
        {
            "X": np.random.randint(256, size=(32, 32, 3, num_samples), dtype=np.uint8),
            "y": np.random.randint(10, size=(num_samples,), dtype=np.uint8),
        },
    )
    return num_samples


@register_mock(configs=combinations_grid(split=("train", "val", "test")))
def pcam(root, config):
    import h5py

    num_images = {"train": 2, "test": 3, "val": 4}[config["split"]]

    split = "valid" if config["split"] == "val" else config["split"]

    images_io = io.BytesIO()
    with h5py.File(images_io, "w") as f:
        f["x"] = np.random.randint(0, 256, size=(num_images, 10, 10, 3), dtype=np.uint8)

    targets_io = io.BytesIO()
    with h5py.File(targets_io, "w") as f:
        f["y"] = np.random.randint(0, 2, size=(num_images, 1, 1, 1), dtype=np.uint8)

    # Create .gz compressed files
    images_file = root / f"camelyonpatch_level_2_split_{split}_x.h5.gz"
    targets_file = root / f"camelyonpatch_level_2_split_{split}_y.h5.gz"
    for compressed_file_name, uncompressed_file_io in ((images_file, images_io), (targets_file, targets_io)):
        compressed_data = gzip.compress(uncompressed_file_io.getbuffer())
        with open(compressed_file_name, "wb") as compressed_file:
            compressed_file.write(compressed_data)

    return num_images


@register_mock(name="stanford-cars", configs=combinations_grid(split=("train", "test")))
def stanford_cars(root, config):
    import scipy.io as io
    from numpy.core.records import fromarrays

    split = config["split"]
    num_samples = {"train": 5, "test": 7}[split]
    num_categories = 3

    if split == "train":
        images_folder_name = "cars_train"
        devkit = root / "devkit"
        devkit.mkdir()
        annotations_mat_path = devkit / "cars_train_annos.mat"
    else:
        images_folder_name = "cars_test"
        annotations_mat_path = root / "cars_test_annos_withlabels.mat"

    create_image_folder(
        root=root,
        name=images_folder_name,
        file_name_fn=lambda image_index: f"{image_index:5d}.jpg",
        num_examples=num_samples,
    )

    make_tar(root, f"cars_{split}.tgz", images_folder_name)
    bbox = np.random.randint(1, 200, num_samples, dtype=np.uint8)
    classes = np.random.randint(1, num_categories + 1, num_samples, dtype=np.uint8)
    fnames = [f"{i:5d}.jpg" for i in range(num_samples)]
    rec_array = fromarrays(
        [bbox, bbox, bbox, bbox, classes, fnames],
        names=["bbox_x1", "bbox_y1", "bbox_x2", "bbox_y2", "class", "fname"],
    )

    io.savemat(annotations_mat_path, {"annotations": rec_array})
    if split == "train":
        make_tar(root, "car_devkit.tgz", devkit, compression="gz")

    return num_samples


@register_mock(configs=combinations_grid(split=("train", "test")))
def usps(root, config):
    num_samples = {"train": 15, "test": 7}[config["split"]]

    with bz2.open(root / f"usps{'.t' if not config['split'] == 'train' else ''}.bz2", "wb") as fh:
        lines = []
        for _ in range(num_samples):
            label = make_tensor(1, low=1, high=11, dtype=torch.int)
            values = make_tensor(256, low=-1, high=1, dtype=torch.float)
            lines.append(
                " ".join([f"{int(label)}", *(f"{idx}:{float(value):.6f}" for idx, value in enumerate(values, 1))])
            )

        fh.write("\n".join(lines).encode())

    return num_samples
