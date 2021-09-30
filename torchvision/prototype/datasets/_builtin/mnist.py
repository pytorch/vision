import abc
import codecs
import functools
import io
import operator
import pathlib
import string
from typing import Any, Callable, Dict, Iterator, List, Optional, Tuple, cast

import numpy as np
import torch
from torch.utils.data import IterDataPipe
from torch.utils.data.datapipes.iter import (
    Demultiplexer,
    Mapper,
    ZipArchiveReader,
    Zipper,
    Shuffler,
)

from torchvision.prototype.datasets.utils import (
    Dataset,
    DatasetConfig,
    DatasetInfo,
    HttpResource,
    OnlineResource,
)
from torchvision.prototype.datasets.utils._internal import (
    image_buffer_from_array,
    Decompressor,
    Slicer,
    INFINITE_BUFFER_SIZE,
)


__all__ = ["MNIST", "FashionMNIST", "KMNIST", "EMNIST", "QMNIST"]

prod = functools.partial(functools.reduce, operator.mul)


class MNISTFileReader(IterDataPipe):
    _DTYPE_MAP = {
        8: "u1",  # uint8
        9: "i1",  # int8
        11: "i2",  # int16
        12: "i4",  # int32
        13: "f4",  # float32
        14: "f8",  # float64
    }

    def __init__(self, datapipe: IterDataPipe) -> None:
        self.datapipe = datapipe

    @staticmethod
    def _decode(bytes):
        return int(codecs.encode(bytes, "hex"), 16)

    def __iter__(self) -> Iterator[np.ndarray]:
        for _, file in self.datapipe:
            magic = self._decode(file.read(4))
            dtype_type = self._DTYPE_MAP[magic // 256]
            ndim = magic % 256 - 1

            num_samples = self._decode(file.read(4))
            shape = [self._decode(file.read(4)) for _ in range(ndim)]

            in_dtype = np.dtype(f">{dtype_type}")
            out_dtype = np.dtype(dtype_type)
            chunk_size = (cast(int, prod(shape)) if shape else 1) * in_dtype.itemsize
            for _ in range(num_samples):
                chunk = file.read(chunk_size)
                yield np.frombuffer(chunk, dtype=in_dtype).astype(out_dtype).reshape(shape)


class _MNISTBase(Dataset):
    _FORMAT = "png"
    _URL_BASE: str

    @abc.abstractmethod
    def _files_and_checksums(self, config: DatasetConfig) -> Tuple[Tuple[str, str], Tuple[str, str]]:
        pass

    def resources(self, config: DatasetConfig) -> List[OnlineResource]:
        (images_file, images_sha256), (
            labels_file,
            labels_sha256,
        ) = self._files_and_checksums(config)

        images = HttpResource(f"{self._URL_BASE}/{images_file}", sha256=images_sha256)
        labels = HttpResource(f"{self._URL_BASE}/{labels_file}", sha256=labels_sha256)

        return [images, labels]

    def _collate_and_decode(
        self,
        data: Tuple[np.ndarray, np.ndarray],
        *,
        decoder: Optional[Callable[[io.IOBase], torch.Tensor]],
    ):
        image_array, label_array = data

        image_buffer = image_buffer_from_array(image_array)
        image = decoder(image_buffer) if decoder else image_buffer

        label = torch.tensor(label_array, dtype=torch.int64)
        category = self.info.categories[int(label)]

        return dict(image=image, label=label, category=category)

    def _make_datapipe(
        self,
        resource_dps: List[IterDataPipe],
        *,
        config: DatasetConfig,
        decoder: Optional[Callable[[io.IOBase], torch.Tensor]],
    ) -> IterDataPipe[Dict[str, Any]]:
        images_dp, labels_dp = resource_dps

        images_dp = Decompressor(images_dp)
        images_dp = MNISTFileReader(images_dp)

        labels_dp = Decompressor(labels_dp)
        labels_dp = MNISTFileReader(labels_dp)

        dp: IterDataPipe = Zipper(images_dp, labels_dp)
        dp = Shuffler(dp, buffer_size=INFINITE_BUFFER_SIZE)
        return Mapper(dp, self._collate_and_decode, fn_kwargs=dict(decoder=decoder))


class MNIST(_MNISTBase):
    @property
    def info(self):
        return DatasetInfo(
            "mnist",
            categories=10,
            homepage="http://yann.lecun.com/exdb/mnist",
            valid_options=dict(
                split=("train", "test"),
            ),
        )

    _URL_BASE = "http://yann.lecun.com/exdb/mnist"
    _CHECKSUMS = {
        "train-images-idx3-ubyte.gz": "440fcabf73cc546fa21475e81ea370265605f56be210a4024d2ca8f203523609",
        "train-labels-idx1-ubyte.gz": "3552534a0a558bbed6aed32b30c495cca23d567ec52cac8be1a0730e8010255c",
        "t10k-images-idx3-ubyte.gz": "8d422c7b0a1c1c79245a5bcf07fe86e33eeafee792b84584aec276f5a2dbc4e6",
        "t10k-labels-idx1-ubyte.gz": "f7ae60f92e00ec6debd23a6088c31dbd2371eca3ffa0defaefb259924204aec6",
    }

    def _files_and_checksums(self, config: DatasetConfig) -> Tuple[Tuple[str, str], Tuple[str, str]]:
        prefix = "train" if config.split == "train" else "t10k"
        images_file = f"{prefix}-images-idx3-ubyte.gz"
        labels_file = f"{prefix}-labels-idx1-ubyte.gz"
        return (images_file, self._CHECKSUMS[images_file]), (
            labels_file,
            self._CHECKSUMS[labels_file],
        )


class FashionMNIST(MNIST):
    @property
    def info(self):
        return DatasetInfo(
            "fashionmnist",
            categories=(
                "T-shirt/top",
                "Trouser",
                "Pullover",
                "Dress",
                "Coat",
                "Sandal",
                "Shirt",
                "Sneaker",
                "Bag",
                "Ankle boot",
            ),
            homepage="https://github.com/zalandoresearch/fashion-mnist",
            valid_options=dict(
                split=("train", "test"),
            ),
        )

    _URL_BASE = "fashion-mnist.s3-website.eu-central-1.amazonaws.com/"
    _CHECKSUMS = {
        "train-images-idx3-ubyte.gz": "3aede38d61863908ad78613f6a32ed271626dd12800ba2636569512369268a84",
        "train-labels-idx1-ubyte.gz": "a04f17134ac03560a47e3764e11b92fc97de4d1bfaf8ba1a3aa29af54cc90845",
        "t10k-images-idx3-ubyte.gz": "346e55b948d973a97e58d2351dde16a484bd415d4595297633bb08f03db6a073",
        "t10k-labels-idx1-ubyte.gz": "67da17c76eaffca5446c3361aaab5c3cd6d1c2608764d35dfb1850b086bf8dd5",
    }


class KMNIST(MNIST):
    @property
    def info(self):
        return DatasetInfo(
            "kmnist",
            categories=["o", "ki", "su", "tsu", "na", "ha", "ma", "ya", "re", "wo"],
            homepage="http://codh.rois.ac.jp/kmnist/index.html.en",
            valid_options=dict(
                split=("train", "test"),
            ),
        )

    _URL_BASE = "http://codh.rois.ac.jp/kmnist/index.html.en"
    _CHECKSUMS = {
        "train-images-idx3-ubyte.gz": "51467d22d8cc72929e2a028a0428f2086b092bb31cfb79c69cc0a90ce135fde4",
        "train-labels-idx1-ubyte.gz": "e38f9ebcd0f3ebcdec7fc8eabdcdaef93bb0df8ea12bee65224341c8183d8e17",
        "t10k-images-idx3-ubyte.gz": "edd7a857845ad6bb1d0ba43fe7e794d164fe2dce499a1694695a792adfac43c5",
        "t10k-labels-idx1-ubyte.gz": "20bb9a0ef54c7db3efc55a92eef5582c109615df22683c380526788f98e42a1c",
    }


class EMNIST(_MNISTBase):
    @property
    def info(self):
        return DatasetInfo(
            "emnist",
            # FIXME: shift the labels at runtime to always return a static label
            categories=list(string.digits + string.ascii_letters),
            homepage="https://www.westernsydney.edu.au/icns/reproducible_research/publication_support_materials/emnist",
            valid_options=dict(
                split=("train", "test"),
                image_set=(
                    "mnist",
                    "byclass",
                    "bymerge",
                    "balanced",
                    "digits",
                    "letters",
                ),
            ),
        )

    _URL_BASE = "https://rds.westernsydney.edu.au/Institutes/MARCS/BENS/EMNIST"

    def _files_and_checksums(self, config: DatasetConfig) -> Tuple[Tuple[str, str], Tuple[str, str]]:
        prefix = f"emnist-{config.image_set}-{config.split}"
        images_file = f"{prefix}-images-idx3-ubyte.gz"
        labels_file = f"{prefix}-labels-idx1-ubyte.gz"
        # Since EMNIST provides the data files inside an archive, we don't need provide checksums for them
        return (images_file, ""), (labels_file, "")

    def resources(self, config: Optional[DatasetConfig] = None) -> List[OnlineResource]:
        return [
            HttpResource(
                f"{self._URL_BASE}/emnist-gzip.zip",
                sha256="909a2a39c5e86bdd7662425e9b9c4a49bb582bf8d0edad427f3c3a9d0c6f7259",
            )
        ]

    def _classify_archive(self, data: Tuple[str, Any], *, config: DatasetConfig) -> Optional[int]:
        path = pathlib.Path(data[0])
        (images_file, _), (labels_file, _) = self._files_and_checksums(config)
        if path.name == images_file:
            return 0
        elif path.name == labels_file:
            return 1
        else:
            return None

    def _make_datapipe(
        self,
        resource_dps: List[IterDataPipe],
        *,
        config: DatasetConfig,
        decoder: Optional[Callable[[io.IOBase], torch.Tensor]],
    ) -> IterDataPipe[Dict[str, Any]]:
        archive_dp = resource_dps[0]
        archive_dp = ZipArchiveReader(archive_dp)
        images_dp, labels_dp = Demultiplexer(
            archive_dp,
            2,
            functools.partial(self._classify_archive, config=config),  # type:ignore[arg-type]
            drop_none=True,
            buffer_size=INFINITE_BUFFER_SIZE,
        )
        return super()._make_datapipe([images_dp, labels_dp], config=config, decoder=decoder)


class QMNIST(_MNISTBase):
    @property
    def info(self):
        return DatasetInfo(
            "qmnist",
            categories=10,
            homepage="https://github.com/facebookresearch/qmnist",
            valid_options=dict(
                split=("train", "test", "test10k", "test50k", "nist"),
            ),
        )

    _URL_BASE = "https://raw.githubusercontent.com/facebookresearch/qmnist/master"
    _CHECKSUMS = {
        "qmnist-train-images-idx3-ubyte.gz": "9e26a7bf1683614e065d7b76460ccd52807165b3f22561fb782bd9f38c52b51d",
        "qmnist-train-labels-idx2-int.gz": "2c05dc77f6b916b38e455e97ab129a42a444f3dbef09b278a366f82904e0dd9f",
        "qmnist-test-images-idx3-ubyte.gz": "43fc22bf7498b8fc98de98369d72f752d0deabc280a43a7bcc364ab19e57b375",
        "qmnist-test-labels-idx2-int.gz": "9fbcbe594c3766fdf4f0b15c5165dc0d1e57ac604e01422608bb72c906030d06",
        "xnist-images-idx3-ubyte.xz": "f075553993026d4359ded42208eff77a1941d3963c1eff49d6015814f15f0984",
        "xnist-labels-idx2-int.xz": "db042968723ec2b7aed5f1beac25d2b6e983b9286d4f4bf725f1086e5ae55c4f",
    }

    def _files_and_checksums(self, config: DatasetConfig) -> Tuple[Tuple[str, str], Tuple[str, str]]:
        prefix = "xnist" if config.split == "nist" else f"qmnist-{'train' if config.split== 'train' else 'test'}"
        suffix = "xz" if config.split == "nist" else "gz"
        images_file = f"{prefix}-images-idx3-ubyte.{suffix}"
        labels_file = f"{prefix}-labels-idx2-int.{suffix}"
        return (images_file, self._CHECKSUMS[images_file]), (
            labels_file,
            self._CHECKSUMS[labels_file],
        )

    def _split_label(self, sample: Dict[str, Any]) -> Dict[str, Any]:
        parts = [part.squeeze(0) for part in sample.pop("label").split(1)]
        sample.update(
            dict(
                zip(
                    (
                        "label",
                        "nist_hsf_series",
                        "nist_writer_id",
                        "digit_index",
                        "nist_label",
                        "global_digit_index",
                    ),
                    parts[:6],
                )
            )
        )
        sample.update(dict(zip(("duplicate", "unused"), [bool(value) for value in parts[-2:]])))
        return sample

    def _collate_and_decode(
        self,
        data: Tuple[np.ndarray, np.ndarray],
        *,
        decoder: Optional[Callable[[io.IOBase], torch.Tensor]],
    ):
        image_array, label_array = data
        label_parts = label_array.tolist()
        sample = super()._collate_and_decode((image_array, label_parts[0]), decoder=decoder)

        sample.update(
            dict(
                zip(
                    ("nist_hsf_series", "nist_writer_id", "digit_index", "nist_label", "global_digit_index"),
                    label_parts[1:6],
                )
            )
        )
        sample.update(dict(zip(("duplicate", "unused"), [bool(value) for value in label_parts[-2:]])))
        return sample

    def _make_datapipe(
        self,
        resource_dps: List[IterDataPipe],
        *,
        config: DatasetConfig,
        decoder: Optional[Callable[[io.IOBase], torch.Tensor]],
    ) -> IterDataPipe[Dict[str, Any]]:
        dp = super()._make_datapipe(resource_dps, config=config, decoder=decoder)
        # dp = Mapper(dp, self._split_label)
        if config.split not in ("test10k", "test50k"):
            return dp

        start: Optional[int]
        stop: Optional[int]
        if config.split == "test10k":
            start = 0
            stop = 10000
        else:  # config.split == "test50k"
            start = 10000
            stop = None

        return Slicer(dp, start=start, stop=stop)
