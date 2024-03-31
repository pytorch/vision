import abc
import functools
import operator
import pathlib
import string
from typing import Any, BinaryIO, cast, Dict, Iterator, List, Optional, Sequence, Tuple, Union

import torch
from torchdata.datapipes.iter import Decompressor, Demultiplexer, IterDataPipe, Mapper, Zipper
from torchvision.prototype.datasets.utils import Dataset, HttpResource, OnlineResource
from torchvision.prototype.datasets.utils._internal import hint_sharding, hint_shuffling, INFINITE_BUFFER_SIZE
from torchvision.prototype.tv_tensors import Label
from torchvision.prototype.utils._internal import fromfile
from torchvision.tv_tensors import Image

from .._api import register_dataset, register_info


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
            try:
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
            finally:
                file.close()


class _MNISTBase(Dataset):
    _URL_BASE: Union[str, Sequence[str]]

    @abc.abstractmethod
    def _files_and_checksums(self) -> Tuple[Tuple[str, str], Tuple[str, str]]:
        pass

    def _resources(self) -> List[OnlineResource]:
        (images_file, images_sha256), (
            labels_file,
            labels_sha256,
        ) = self._files_and_checksums()

        url_bases = self._URL_BASE
        if isinstance(url_bases, str):
            url_bases = (url_bases,)

        images_urls = [f"{url_base}/{images_file}" for url_base in url_bases]
        images = HttpResource(images_urls[0], sha256=images_sha256, mirrors=images_urls[1:])

        labels_urls = [f"{url_base}/{labels_file}" for url_base in url_bases]
        labels = HttpResource(labels_urls[0], sha256=labels_sha256, mirrors=labels_urls[1:])

        return [images, labels]

    def start_and_stop(self) -> Tuple[Optional[int], Optional[int]]:
        return None, None

    _categories: List[str]

    def _prepare_sample(self, data: Tuple[torch.Tensor, torch.Tensor]) -> Dict[str, Any]:
        image, label = data
        return dict(
            image=Image(image),
            label=Label(label, dtype=torch.int64, categories=self._categories),
        )

    def _datapipe(self, resource_dps: List[IterDataPipe]) -> IterDataPipe[Dict[str, Any]]:
        images_dp, labels_dp = resource_dps
        start, stop = self.start_and_stop()

        images_dp = Decompressor(images_dp)
        images_dp = MNISTFileReader(images_dp, start=start, stop=stop)

        labels_dp = Decompressor(labels_dp)
        labels_dp = MNISTFileReader(labels_dp, start=start, stop=stop)

        dp = Zipper(images_dp, labels_dp)
        dp = hint_shuffling(dp)
        dp = hint_sharding(dp)
        return Mapper(dp, self._prepare_sample)


@register_info("mnist")
def _mnist_info() -> Dict[str, Any]:
    return dict(
        categories=[str(label) for label in range(10)],
    )


@register_dataset("mnist")
class MNIST(_MNISTBase):
    """
    - **homepage**: http://yann.lecun.com/exdb/mnist
    """

    def __init__(
        self,
        root: Union[str, pathlib.Path],
        *,
        split: str = "train",
        skip_integrity_check: bool = False,
    ) -> None:
        self._split = self._verify_str_arg(split, "split", ("train", "test"))
        super().__init__(root, skip_integrity_check=skip_integrity_check)

    _URL_BASE: Union[str, Sequence[str]] = (
        "http://yann.lecun.com/exdb/mnist",
        "https://ossci-datasets.s3.amazonaws.com/mnist",
    )
    _CHECKSUMS = {
        "train-images-idx3-ubyte.gz": "440fcabf73cc546fa21475e81ea370265605f56be210a4024d2ca8f203523609",
        "train-labels-idx1-ubyte.gz": "3552534a0a558bbed6aed32b30c495cca23d567ec52cac8be1a0730e8010255c",
        "t10k-images-idx3-ubyte.gz": "8d422c7b0a1c1c79245a5bcf07fe86e33eeafee792b84584aec276f5a2dbc4e6",
        "t10k-labels-idx1-ubyte.gz": "f7ae60f92e00ec6debd23a6088c31dbd2371eca3ffa0defaefb259924204aec6",
    }

    def _files_and_checksums(self) -> Tuple[Tuple[str, str], Tuple[str, str]]:
        prefix = "train" if self._split == "train" else "t10k"
        images_file = f"{prefix}-images-idx3-ubyte.gz"
        labels_file = f"{prefix}-labels-idx1-ubyte.gz"
        return (images_file, self._CHECKSUMS[images_file]), (
            labels_file,
            self._CHECKSUMS[labels_file],
        )

    _categories = _mnist_info()["categories"]

    def __len__(self) -> int:
        return 60_000 if self._split == "train" else 10_000


@register_info("fashionmnist")
def _fashionmnist_info() -> Dict[str, Any]:
    return dict(
        categories=[
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
        ],
    )


@register_dataset("fashionmnist")
class FashionMNIST(MNIST):
    """
    - **homepage**: https://github.com/zalandoresearch/fashion-mnist
    """

    _URL_BASE = "http://fashion-mnist.s3-website.eu-central-1.amazonaws.com"
    _CHECKSUMS = {
        "train-images-idx3-ubyte.gz": "3aede38d61863908ad78613f6a32ed271626dd12800ba2636569512369268a84",
        "train-labels-idx1-ubyte.gz": "a04f17134ac03560a47e3764e11b92fc97de4d1bfaf8ba1a3aa29af54cc90845",
        "t10k-images-idx3-ubyte.gz": "346e55b948d973a97e58d2351dde16a484bd415d4595297633bb08f03db6a073",
        "t10k-labels-idx1-ubyte.gz": "67da17c76eaffca5446c3361aaab5c3cd6d1c2608764d35dfb1850b086bf8dd5",
    }

    _categories = _fashionmnist_info()["categories"]


@register_info("kmnist")
def _kmnist_info() -> Dict[str, Any]:
    return dict(
        categories=["o", "ki", "su", "tsu", "na", "ha", "ma", "ya", "re", "wo"],
    )


@register_dataset("kmnist")
class KMNIST(MNIST):
    """
    - **homepage**: http://codh.rois.ac.jp/kmnist/index.html.en
    """

    _URL_BASE = "http://codh.rois.ac.jp/kmnist/dataset/kmnist"
    _CHECKSUMS = {
        "train-images-idx3-ubyte.gz": "51467d22d8cc72929e2a028a0428f2086b092bb31cfb79c69cc0a90ce135fde4",
        "train-labels-idx1-ubyte.gz": "e38f9ebcd0f3ebcdec7fc8eabdcdaef93bb0df8ea12bee65224341c8183d8e17",
        "t10k-images-idx3-ubyte.gz": "edd7a857845ad6bb1d0ba43fe7e794d164fe2dce499a1694695a792adfac43c5",
        "t10k-labels-idx1-ubyte.gz": "20bb9a0ef54c7db3efc55a92eef5582c109615df22683c380526788f98e42a1c",
    }

    _categories = _kmnist_info()["categories"]


@register_info("emnist")
def _emnist_info() -> Dict[str, Any]:
    return dict(
        categories=list(string.digits + string.ascii_uppercase + string.ascii_lowercase),
    )


@register_dataset("emnist")
class EMNIST(_MNISTBase):
    """
    - **homepage**: https://www.westernsydney.edu.au/icns/reproducible_research/publication_support_materials/emnist
    """

    def __init__(
        self,
        root: Union[str, pathlib.Path],
        *,
        split: str = "train",
        image_set: str = "Balanced",
        skip_integrity_check: bool = False,
    ) -> None:
        self._split = self._verify_str_arg(split, "split", ("train", "test"))
        self._image_set = self._verify_str_arg(
            image_set, "image_set", ("Balanced", "By_Merge", "By_Class", "Letters", "Digits", "MNIST")
        )
        super().__init__(root, skip_integrity_check=skip_integrity_check)

    _URL_BASE = "https://rds.westernsydney.edu.au/Institutes/MARCS/BENS/EMNIST"

    def _files_and_checksums(self) -> Tuple[Tuple[str, str], Tuple[str, str]]:
        prefix = f"emnist-{self._image_set.replace('_', '').lower()}-{self._split}"
        images_file = f"{prefix}-images-idx3-ubyte.gz"
        labels_file = f"{prefix}-labels-idx1-ubyte.gz"
        # Since EMNIST provides the data files inside an archive, we don't need to provide checksums for them
        return (images_file, ""), (labels_file, "")

    def _resources(self) -> List[OnlineResource]:
        return [
            HttpResource(
                f"{self._URL_BASE}/emnist-gzip.zip",
                sha256="909a2a39c5e86bdd7662425e9b9c4a49bb582bf8d0edad427f3c3a9d0c6f7259",
            )
        ]

    def _classify_archive(self, data: Tuple[str, Any]) -> Optional[int]:
        path = pathlib.Path(data[0])
        (images_file, _), (labels_file, _) = self._files_and_checksums()
        if path.name == images_file:
            return 0
        elif path.name == labels_file:
            return 1
        else:
            return None

    _categories = _emnist_info()["categories"]

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

    def _prepare_sample(self, data: Tuple[torch.Tensor, torch.Tensor]) -> Dict[str, Any]:
        # In these two splits, some lowercase letters are merged into their uppercase ones (see Fig 2. in the paper).
        # That means for example that there is 'D', 'd', and 'C', but not 'c'. Since the labels are nevertheless dense,
        # i.e. no gaps between 0 and 46 for 47 total classes, we need to add an offset to create these gaps. For
        # example, since there is no 'c', 'd' corresponds to
        # label 38 (10 digits + 26 uppercase letters + 3rd unmerged lower case letter - 1 for zero indexing),
        # and at the same time corresponds to
        # index 39 (10 digits + 26 uppercase letters + 4th lower case letter - 1 for zero indexing)
        # in self._categories. Thus, we need to add 1 to the label to correct this.
        if self._image_set in ("Balanced", "By_Merge"):
            image, label = data
            label += self._LABEL_OFFSETS.get(int(label), 0)
            data = (image, label)
        return super()._prepare_sample(data)

    def _datapipe(self, resource_dps: List[IterDataPipe]) -> IterDataPipe[Dict[str, Any]]:
        archive_dp = resource_dps[0]
        images_dp, labels_dp = Demultiplexer(
            archive_dp,
            2,
            self._classify_archive,
            drop_none=True,
            buffer_size=INFINITE_BUFFER_SIZE,
        )
        return super()._datapipe([images_dp, labels_dp])

    def __len__(self) -> int:
        return {
            ("train", "Balanced"): 112_800,
            ("train", "By_Merge"): 697_932,
            ("train", "By_Class"): 697_932,
            ("train", "Letters"): 124_800,
            ("train", "Digits"): 240_000,
            ("train", "MNIST"): 60_000,
            ("test", "Balanced"): 18_800,
            ("test", "By_Merge"): 116_323,
            ("test", "By_Class"): 116_323,
            ("test", "Letters"): 20_800,
            ("test", "Digits"): 40_000,
            ("test", "MNIST"): 10_000,
        }[(self._split, self._image_set)]


@register_info("qmnist")
def _qmnist_info() -> Dict[str, Any]:
    return dict(
        categories=[str(label) for label in range(10)],
    )


@register_dataset("qmnist")
class QMNIST(_MNISTBase):
    """
    - **homepage**: https://github.com/facebookresearch/qmnist
    """

    def __init__(
        self,
        root: Union[str, pathlib.Path],
        *,
        split: str = "train",
        skip_integrity_check: bool = False,
    ) -> None:
        self._split = self._verify_str_arg(split, "split", ("train", "test", "test10k", "test50k", "nist"))
        super().__init__(root, skip_integrity_check=skip_integrity_check)

    _URL_BASE = "https://raw.githubusercontent.com/facebookresearch/qmnist/master"
    _CHECKSUMS = {
        "qmnist-train-images-idx3-ubyte.gz": "9e26a7bf1683614e065d7b76460ccd52807165b3f22561fb782bd9f38c52b51d",
        "qmnist-train-labels-idx2-int.gz": "2c05dc77f6b916b38e455e97ab129a42a444f3dbef09b278a366f82904e0dd9f",
        "qmnist-test-images-idx3-ubyte.gz": "43fc22bf7498b8fc98de98369d72f752d0deabc280a43a7bcc364ab19e57b375",
        "qmnist-test-labels-idx2-int.gz": "9fbcbe594c3766fdf4f0b15c5165dc0d1e57ac604e01422608bb72c906030d06",
        "xnist-images-idx3-ubyte.xz": "f075553993026d4359ded42208eff77a1941d3963c1eff49d6015814f15f0984",
        "xnist-labels-idx2-int.xz": "db042968723ec2b7aed5f1beac25d2b6e983b9286d4f4bf725f1086e5ae55c4f",
    }

    def _files_and_checksums(self) -> Tuple[Tuple[str, str], Tuple[str, str]]:
        prefix = "xnist" if self._split == "nist" else f"qmnist-{'train' if self._split == 'train' else 'test'}"
        suffix = "xz" if self._split == "nist" else "gz"
        images_file = f"{prefix}-images-idx3-ubyte.{suffix}"
        labels_file = f"{prefix}-labels-idx2-int.{suffix}"
        return (images_file, self._CHECKSUMS[images_file]), (
            labels_file,
            self._CHECKSUMS[labels_file],
        )

    def start_and_stop(self) -> Tuple[Optional[int], Optional[int]]:
        start: Optional[int]
        stop: Optional[int]
        if self._split == "test10k":
            start = 0
            stop = 10000
        elif self._split == "test50k":
            start = 10000
            stop = None
        else:
            start = stop = None

        return start, stop

    _categories = _emnist_info()["categories"]

    def _prepare_sample(self, data: Tuple[torch.Tensor, torch.Tensor]) -> Dict[str, Any]:
        image, ann = data
        label, *extra_anns = ann
        sample = super()._prepare_sample((image, label))

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

    def __len__(self) -> int:
        return {
            "train": 60_000,
            "test": 60_000,
            "test10k": 10_000,
            "test50k": 50_000,
            "nist": 402_953,
        }[self._split]
