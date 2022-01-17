import contextlib
import functools
import gzip
import itertools
import json
import lzma
import pathlib
import pickle
import tempfile
from collections import defaultdict, UserList

import numpy as np
import PIL.Image
import pytest
import torch
from datasets_utils import make_zip, make_tar, create_image_folder
from torch.testing import make_tensor as _make_tensor
from torchvision.prototype import datasets
from torchvision.prototype.datasets._api import DEFAULT_DECODER_MAP, DEFAULT_DECODER, find

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
        self.mock_data_fn = self._parse_mock_data(mock_data_fn)
        self.configs = configs or self.info._configs
        self._cache = {}

    @property
    def info(self):
        return self.dataset.info

    @property
    def name(self):
        return self.info.name

    def _parse_mock_data(self, mock_data_fn):
        def wrapper(info, root, config):
            mock_infos = mock_data_fn(info, root, config)

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

        return wrapper

    def _load_mock(self, config):
        with contextlib.suppress(KeyError):
            return self._cache[config]

        self.root.mkdir(exist_ok=True)
        for config_, mock_info in self.mock_data_fn(self.info, self.root, config).items():
            mock_resources = [
                ResourceMock(dataset_name=self.name, dataset_config=config_, file_name=resource.file_name)
                for resource in self.dataset.resources(config_)
            ]
            self._cache[config_] = (mock_resources, mock_info)

        return self._cache[config]

    def load(self, config, *, decoder=DEFAULT_DECODER):
        try:
            self.info.check_dependencies()
        except ModuleNotFoundError as error:
            pytest.skip(str(error))

        mock_resources, mock_info = self._load_mock(config)
        datapipe = self.dataset._make_datapipe(
            [resource.load(self.root) for resource in mock_resources],
            config=config,
            decoder=DEFAULT_DECODER_MAP.get(self.info.type) if decoder is DEFAULT_DECODER else decoder,
        )
        return datapipe, mock_info


class DatasetMocks(UserList):
    def append_named_callable(self, fn):
        mock_data_fn = fn.__func__ if isinstance(fn, classmethod) else fn
        self.data.append(DatasetMock(mock_data_fn.__name__, mock_data_fn))
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


@DATASET_MOCKS.append_named_callable
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


DATASET_MOCKS.extend([DatasetMock(name, mnist) for name in ["fashionmnist", "kmnist"]])


@DATASET_MOCKS.append_named_callable
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


@DATASET_MOCKS.append_named_callable
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


@DATASET_MOCKS.append_named_callable
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


@DATASET_MOCKS.append_named_callable
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


@DATASET_MOCKS.append_named_callable
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


@DATASET_MOCKS.append_named_callable
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


@DATASET_MOCKS.append_named_callable
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


@DATASET_MOCKS.append_named_callable
def coco(info, root, config):
    return dict(
        zip(
            [config_ for config_ in info._configs if config_.year == config.year],
            itertools.repeat(CocoMockData.generate(root, year=config.year, num_samples=5)),
        )
    )


def config_id(name, config):
    parts = [name]
    for name, value in config.items():
        if isinstance(value, bool):
            part = ("" if value else "no_") + name
        else:
            part = str(value)
        parts.append(part)
    return "-".join(parts)


def parametrize_dataset_mocks(datasets_mocks):
    return pytest.mark.parametrize(
        ("dataset_mock", "config"),
        [
            pytest.param(dataset_mock, config, id=config_id(dataset_mock.name, config))
            for dataset_mock in datasets_mocks
            for config in dataset_mock.configs
        ],
    )
