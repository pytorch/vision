import abc
import io
import pathlib
import pickle
from typing import Any, BinaryIO, cast, Dict, Iterator, List, Optional, Tuple, Union

import numpy as np
from torchdata.datapipes.iter import Filter, IterDataPipe, Mapper
from torchvision.prototype.datasets.utils import Dataset, HttpResource, OnlineResource
from torchvision.prototype.datasets.utils._internal import (
    hint_sharding,
    hint_shuffling,
    path_comparator,
    read_categories_file,
)
from torchvision.prototype.tv_tensors import Label
from torchvision.tv_tensors import Image

from .._api import register_dataset, register_info


class CifarFileReader(IterDataPipe[Tuple[np.ndarray, int]]):
    def __init__(self, datapipe: IterDataPipe[Dict[str, Any]], *, labels_key: str) -> None:
        self.datapipe = datapipe
        self.labels_key = labels_key

    def __iter__(self) -> Iterator[Tuple[np.ndarray, int]]:
        for mapping in self.datapipe:
            image_arrays = mapping["data"].reshape((-1, 3, 32, 32))
            category_idcs = mapping[self.labels_key]
            yield from iter(zip(image_arrays, category_idcs))


class _CifarBase(Dataset):
    _FILE_NAME: str
    _SHA256: str
    _LABELS_KEY: str
    _META_FILE_NAME: str
    _CATEGORIES_KEY: str
    _categories: List[str]

    def __init__(
        self,
        root: Union[str, pathlib.Path],
        *,
        split: str = "train",
        skip_integrity_check: bool = False,
    ) -> None:
        self._split = self._verify_str_arg(split, "split", ("train", "test"))
        super().__init__(root, skip_integrity_check=skip_integrity_check)

    @abc.abstractmethod
    def _is_data_file(self, data: Tuple[str, BinaryIO]) -> Optional[int]:
        pass

    def _resources(self) -> List[OnlineResource]:
        return [
            HttpResource(
                f"https://www.cs.toronto.edu/~kriz/{self._FILE_NAME}",
                sha256=self._SHA256,
            )
        ]

    def _unpickle(self, data: Tuple[str, io.BytesIO]) -> Dict[str, Any]:
        _, file = data
        content = cast(Dict[str, Any], pickle.load(file, encoding="latin1"))
        file.close()
        return content

    def _prepare_sample(self, data: Tuple[np.ndarray, int]) -> Dict[str, Any]:
        image_array, category_idx = data
        return dict(
            image=Image(image_array),
            label=Label(category_idx, categories=self._categories),
        )

    def _datapipe(self, resource_dps: List[IterDataPipe]) -> IterDataPipe[Dict[str, Any]]:
        dp = resource_dps[0]
        dp = Filter(dp, self._is_data_file)
        dp = Mapper(dp, self._unpickle)
        dp = CifarFileReader(dp, labels_key=self._LABELS_KEY)
        dp = hint_shuffling(dp)
        dp = hint_sharding(dp)
        return Mapper(dp, self._prepare_sample)

    def __len__(self) -> int:
        return 50_000 if self._split == "train" else 10_000

    def _generate_categories(self) -> List[str]:
        resources = self._resources()

        dp = resources[0].load(self._root)
        dp = Filter(dp, path_comparator("name", self._META_FILE_NAME))
        dp = Mapper(dp, self._unpickle)

        return cast(List[str], next(iter(dp))[self._CATEGORIES_KEY])


@register_info("cifar10")
def _cifar10_info() -> Dict[str, Any]:
    return dict(categories=read_categories_file("cifar10"))


@register_dataset("cifar10")
class Cifar10(_CifarBase):
    """
    - **homepage**: https://www.cs.toronto.edu/~kriz/cifar.html
    """

    _FILE_NAME = "cifar-10-python.tar.gz"
    _SHA256 = "6d958be074577803d12ecdefd02955f39262c83c16fe9348329d7fe0b5c001ce"
    _LABELS_KEY = "labels"
    _META_FILE_NAME = "batches.meta"
    _CATEGORIES_KEY = "label_names"
    _categories = _cifar10_info()["categories"]

    def _is_data_file(self, data: Tuple[str, Any]) -> bool:
        path = pathlib.Path(data[0])
        return path.name.startswith("data" if self._split == "train" else "test")


@register_info("cifar100")
def _cifar100_info() -> Dict[str, Any]:
    return dict(categories=read_categories_file("cifar100"))


@register_dataset("cifar100")
class Cifar100(_CifarBase):
    """
    - **homepage**: https://www.cs.toronto.edu/~kriz/cifar.html
    """

    _FILE_NAME = "cifar-100-python.tar.gz"
    _SHA256 = "85cd44d02ba6437773c5bbd22e183051d648de2e7d6b014e1ef29b855ba677a7"
    _LABELS_KEY = "fine_labels"
    _META_FILE_NAME = "meta"
    _CATEGORIES_KEY = "fine_label_names"
    _categories = _cifar100_info()["categories"]

    def _is_data_file(self, data: Tuple[str, Any]) -> bool:
        path = pathlib.Path(data[0])
        return path.name == self._split
