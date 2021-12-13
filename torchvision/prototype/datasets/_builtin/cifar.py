import abc
import functools
import io
import pathlib
import pickle
from typing import Any, Callable, Dict, List, Optional, Tuple, Union, Iterator, cast

import numpy as np
import torch
from torchdata.datapipes.iter import (
    IterDataPipe,
    Filter,
    Mapper,
    Shuffler,
)
from torchvision.prototype.datasets.decoder import raw
from torchvision.prototype.datasets.utils import (
    Dataset,
    DatasetConfig,
    DatasetInfo,
    HttpResource,
    OnlineResource,
    DatasetType,
)
from torchvision.prototype.datasets.utils._internal import (
    INFINITE_BUFFER_SIZE,
    image_buffer_from_array,
    path_comparator,
)
from torchvision.prototype.features import Label, Image

__all__ = ["Cifar10", "Cifar100"]


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
    _LABELS_KEY: str
    _META_FILE_NAME: str
    _CATEGORIES_KEY: str

    @abc.abstractmethod
    def _is_data_file(self, data: Tuple[str, io.IOBase], *, config: DatasetConfig) -> Optional[int]:
        pass

    def _unpickle(self, data: Tuple[str, io.BytesIO]) -> Dict[str, Any]:
        _, file = data
        return cast(Dict[str, Any], pickle.load(file, encoding="latin1"))

    def _collate_and_decode(
        self,
        data: Tuple[np.ndarray, int],
        *,
        decoder: Optional[Callable[[io.IOBase], torch.Tensor]],
    ) -> Dict[str, Any]:
        image_array, category_idx = data

        image: Union[Image, io.BytesIO]
        if decoder is raw:
            image = Image(image_array)
        else:
            image_buffer = image_buffer_from_array(image_array.transpose((1, 2, 0)))
            image = decoder(image_buffer) if decoder else image_buffer  # type: ignore[assignment]

        label = Label(category_idx, category=self.categories[category_idx])

        return dict(image=image, label=label)

    def _make_datapipe(
        self,
        resource_dps: List[IterDataPipe],
        *,
        config: DatasetConfig,
        decoder: Optional[Callable[[io.IOBase], torch.Tensor]],
    ) -> IterDataPipe[Dict[str, Any]]:
        dp = resource_dps[0]
        dp = Filter(dp, functools.partial(self._is_data_file, config=config))
        dp = Mapper(dp, self._unpickle)
        dp = CifarFileReader(dp, labels_key=self._LABELS_KEY)
        dp = Shuffler(dp, buffer_size=INFINITE_BUFFER_SIZE)
        return Mapper(dp, self._collate_and_decode, fn_kwargs=dict(decoder=decoder))

    def _generate_categories(self, root: pathlib.Path) -> List[str]:
        dp = self.resources(self.default_config)[0].load(pathlib.Path(root) / self.name)
        dp = Filter(dp, path_comparator("name", self._META_FILE_NAME))
        dp = Mapper(dp, self._unpickle)
        return cast(List[str], next(iter(dp))[self._CATEGORIES_KEY])


class Cifar10(_CifarBase):
    _LABELS_KEY = "labels"
    _META_FILE_NAME = "batches.meta"
    _CATEGORIES_KEY = "label_names"

    def _is_data_file(self, data: Tuple[str, Any], *, config: DatasetConfig) -> bool:
        path = pathlib.Path(data[0])
        return path.name.startswith("data" if config.split == "train" else "test")

    def _make_info(self) -> DatasetInfo:
        return DatasetInfo(
            "cifar10",
            type=DatasetType.RAW,
            homepage="https://www.cs.toronto.edu/~kriz/cifar.html",
        )

    def resources(self, config: DatasetConfig) -> List[OnlineResource]:
        return [
            HttpResource(
                "https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz",
                sha256="6d958be074577803d12ecdefd02955f39262c83c16fe9348329d7fe0b5c001ce",
            )
        ]


class Cifar100(_CifarBase):
    _LABELS_KEY = "fine_labels"
    _META_FILE_NAME = "meta"
    _CATEGORIES_KEY = "fine_label_names"

    def _is_data_file(self, data: Tuple[str, Any], *, config: DatasetConfig) -> bool:
        path = pathlib.Path(data[0])
        return path.name == cast(str, config.split)

    def _make_info(self) -> DatasetInfo:
        return DatasetInfo(
            "cifar100",
            type=DatasetType.RAW,
            homepage="https://www.cs.toronto.edu/~kriz/cifar.html",
            valid_options=dict(
                split=("train", "test"),
            ),
        )

    def resources(self, config: DatasetConfig) -> List[OnlineResource]:
        return [
            HttpResource(
                "https://www.cs.toronto.edu/~kriz/cifar-100-python.tar.gz",
                sha256="85cd44d02ba6437773c5bbd22e183051d648de2e7d6b014e1ef29b855ba677a7",
            )
        ]
