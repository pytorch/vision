import abc
import functools
import io
import operator
import pathlib
import string
from typing import Any, Callable, Dict, Iterator, List, Optional, Tuple, cast, BinaryIO

import torch
from torchdata.datapipes.iter import (
    IterDataPipe,
    Demultiplexer,
    Mapper,
    Zipper,
    Shuffler,
)
from torchvision.prototype.datasets.decoder import raw
from torchvision.prototype.datasets.utils import (
    Dataset,
    DatasetType,
    DatasetConfig,
    DatasetInfo,
    HttpResource,
    OnlineResource,
)
from torchvision.prototype.datasets.utils._internal import (
    image_buffer_from_array,
    Decompressor,
    INFINITE_BUFFER_SIZE,
    fromfile,
)
from torchvision.prototype.features import Image, Label


__all__ = ["MNIST", "FashionMNIST", "KMNIST", "EMNIST", "QMNIST"]

prod = functools.partial(functools.reduce, operator.mul)


class MNISTFileReader(IterDataPipe[torch.Tensor]):
    _DTYPE_MAP = {
        8: torch.uint8,
        9: torch.int8,
        11: torch.int16,
        12: torch.int32,
        13: torch.float32,
        14: torch.float64,
    }

    def __init__(
        self, datapipe: IterDataPipe[Tuple[Any, BinaryIO]], *, start: Optional[int], stop: Optional[int]
    ) -> None:
        self.datapipe = datapipe
        self.start = start
        self.stop = stop

    def __iter__(self) -> Iterator[torch.Tensor]:
        for _, file in self.datapipe:
            read = functools.partial(fromfile, file, byte_order="big")

            magic = int(read(dtype=torch.int32, count=1))
            dtype = self._DTYPE_MAP[magic // 256]
            ndim = magic % 256 - 1

            num_samples = int(read(dtype=torch.int32, count=1))
            shape = cast(List[int], read(dtype=torch.int32, count=ndim).tolist()) if ndim else []
            count = prod(shape) if shape else 1

            start = self.start or 0
            stop = min(self.stop, num_samples) if self.stop else num_samples

            if start:
                num_bytes_per_value = (torch.finfo if dtype.is_floating_point else torch.iinfo)(dtype).bits // 8
                file.seek(num_bytes_per_value * count * start, 1)

            for _ in range(stop - start):
                yield read(dtype=dtype, count=count).reshape(shape)


class _MNISTBase(Dataset):
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

    def start_and_stop(self, config: DatasetConfig) -> Tuple[Optional[int], Optional[int]]:
        return None, None

    def _collate_and_decode(
        self,
        data: Tuple[torch.Tensor, torch.Tensor],
        *,
        config: DatasetConfig,
        decoder: Optional[Callable[[io.IOBase], torch.Tensor]],
    ) -> Dict[str, Any]:
        image, label = data

        if decoder is raw:
            image = Image(image)
        else:
            image_buffer = image_buffer_from_array(image.numpy())
            image = decoder(image_buffer) if decoder else image_buffer  # type: ignore[assignment]

        label = Label(label, dtype=torch.int64, category=self.info.categories[int(label)])

        return dict(image=image, label=label)

    def _make_datapipe(
        self,
        resource_dps: List[IterDataPipe],
        *,
        config: DatasetConfig,
        decoder: Optional[Callable[[io.IOBase], torch.Tensor]],
    ) -> IterDataPipe[Dict[str, Any]]:
        images_dp, labels_dp = resource_dps
        start, stop = self.start_and_stop(config)

        images_dp = Decompressor(images_dp)
        images_dp = MNISTFileReader(images_dp, start=start, stop=stop)

        labels_dp = Decompressor(labels_dp)
        labels_dp = MNISTFileReader(labels_dp, start=start, stop=stop)

        dp = Zipper(images_dp, labels_dp)
        dp = Shuffler(dp, buffer_size=INFINITE_BUFFER_SIZE)
        return Mapper(dp, self._collate_and_decode, fn_kwargs=dict(config=config, decoder=decoder))


class MNIST(_MNISTBase):
    def _make_info(self) -> DatasetInfo:
        return DatasetInfo(
            "mnist",
            type=DatasetType.RAW,
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
    def _make_info(self) -> DatasetInfo:
        return DatasetInfo(
            "fashionmnist",
            type=DatasetType.RAW,
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
    def _make_info(self) -> DatasetInfo:
        return DatasetInfo(
            "kmnist",
            type=DatasetType.RAW,
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
    def _make_info(self) -> DatasetInfo:
        return DatasetInfo(
            "emnist",
            type=DatasetType.RAW,
            categories=list(string.digits + string.ascii_uppercase + string.ascii_lowercase),
            homepage="https://www.westernsydney.edu.au/icns/reproducible_research/publication_support_materials/emnist",
            valid_options=dict(
                split=("train", "test"),
                image_set=(
                    "Balanced",
                    "By_Merge",
                    "By_Class",
                    "Letters",
                    "Digits",
                    "MNIST",
                ),
            ),
        )

    _URL_BASE = "https://rds.westernsydney.edu.au/Institutes/MARCS/BENS/EMNIST"

    def _files_and_checksums(self, config: DatasetConfig) -> Tuple[Tuple[str, str], Tuple[str, str]]:
        prefix = f"emnist-{config.image_set.replace('_', '').lower()}-{config.split}"
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

    _LABEL_OFFSETS = {
        38: 1,
        39: 1,
        40: 1,
        41: 1,
        42: 1,
        43: 6,
        44: 8,
        45: 8,
        46: 9,
    }

    def _collate_and_decode(
        self,
        data: Tuple[torch.Tensor, torch.Tensor],
        *,
        config: DatasetConfig,
        decoder: Optional[Callable[[io.IOBase], torch.Tensor]],
    ) -> Dict[str, Any]:
        # In these two splits, some lowercase letters are merged into their uppercase ones (see Fig 2. in the paper).
        # That means for example that there is 'D', 'd', and 'C', but not 'c'. Since the labels are nevertheless dense,
        # i.e. no gaps between 0 and 46 for 47 total classes, we need to add an offset to create this gaps. For example,
        # since there is no 'c', 'd' corresponds to
        # label 38 (10 digits + 26 uppercase letters + 3rd unmerged lower case letter - 1 for zero indexing),
        # and at the same time corresponds to
        # index 39 (10 digits + 26 uppercase letters + 4th lower case letter - 1 for zero indexing)
        # in self.categories. Thus, we need to add 1 to the label to correct this.
        if config.image_set in ("Balanced", "By_Merge"):
            image, label = data
            label += self._LABEL_OFFSETS.get(int(label), 0)
            data = (image, label)
        return super()._collate_and_decode(data, config=config, decoder=decoder)

    def _make_datapipe(
        self,
        resource_dps: List[IterDataPipe],
        *,
        config: DatasetConfig,
        decoder: Optional[Callable[[io.IOBase], torch.Tensor]],
    ) -> IterDataPipe[Dict[str, Any]]:
        archive_dp = resource_dps[0]
        images_dp, labels_dp = Demultiplexer(
            archive_dp,
            2,
            functools.partial(self._classify_archive, config=config),
            drop_none=True,
            buffer_size=INFINITE_BUFFER_SIZE,
        )
        return super()._make_datapipe([images_dp, labels_dp], config=config, decoder=decoder)


class QMNIST(_MNISTBase):
    def _make_info(self) -> DatasetInfo:
        return DatasetInfo(
            "qmnist",
            type=DatasetType.RAW,
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

    def start_and_stop(self, config: DatasetConfig) -> Tuple[Optional[int], Optional[int]]:
        start: Optional[int]
        stop: Optional[int]
        if config.split == "test10k":
            start = 0
            stop = 10000
        elif config.split == "test50k":
            start = 10000
            stop = None
        else:
            start = stop = None

        return start, stop

    def _collate_and_decode(
        self,
        data: Tuple[torch.Tensor, torch.Tensor],
        *,
        config: DatasetConfig,
        decoder: Optional[Callable[[io.IOBase], torch.Tensor]],
    ) -> Dict[str, Any]:
        image, ann = data
        label, *extra_anns = ann
        sample = super()._collate_and_decode((image, label), config=config, decoder=decoder)

        sample.update(
            dict(
                zip(
                    ("nist_hsf_series", "nist_writer_id", "digit_index", "nist_label", "global_digit_index"),
                    [int(value) for value in extra_anns[:5]],
                )
            )
        )
        sample.update(dict(zip(("duplicate", "unused"), [bool(value) for value in extra_anns[-2:]])))
        return sample
