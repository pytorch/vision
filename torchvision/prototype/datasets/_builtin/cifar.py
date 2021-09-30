import abc
import functools
import io
import pathlib
import pickle
from typing import Any, Callable, Dict, List, Optional, Tuple, Union, TypeVar

import numpy as np

import torch
from torch.utils.data import IterDataPipe
from torch.utils.data.datapipes.iter import (
    Demultiplexer,
    Filter,
    Mapper,
    TarArchiveReader,
    Shuffler,
)
from torchdata.datapipes.iter import KeyZipper

from torchvision.prototype.datasets.utils import (
    Dataset,
    DatasetConfig,
    DatasetInfo,
    HttpResource,
    OnlineResource,
)
from torchvision.prototype.datasets.utils._internal import (
    create_categories_file,
    MappingIterator,
    SequenceIterator,
    INFINITE_BUFFER_SIZE,
    image_buffer_from_array,
    Enumerator,
)

__all__ = ["Cifar10", "Cifar100"]

HERE = pathlib.Path(__file__).parent

D = TypeVar("D")


class _CifarBase(Dataset):
    @abc.abstractmethod
    def _is_data_file(
        self, data: Tuple[str, io.IOBase], *, config: DatasetConfig
    ) -> Optional[int]:
        pass

    @abc.abstractmethod
    def _split_data_file(self, data: Tuple[str, Any]) -> Optional[int]:
        pass

    def _unpickle(self, data: Tuple[str, io.BytesIO]) -> Dict[str, Any]:
        _, file = data
        return pickle.load(file, encoding="latin1")

    def _remove_data_dict_key(self, data: Tuple[str, D]) -> D:
        return data[1]

    def _key_fn(self, data: Tuple[int, Any]) -> int:
        return data[0]

    def _collate_and_decode(
        self, data: Tuple[Tuple[int, int], Tuple[int, np.ndarray]],
        *,
        decoder: Optional[Callable[[io.IOBase], torch.Tensor]],
    ) -> Dict[str, Any]:
        (_, category_idx), (_, image_array_flat) = data

        image_array = image_array_flat.reshape((3, 32, 32)).transpose(1, 2, 0)
        image_buffer = image_buffer_from_array(image_array)

        category = self.categories[category_idx]
        label = torch.tensor(category_idx)

        return dict(image=decoder(image_buffer) if decoder else image_buffer, label=label, category=category)

    def _make_datapipe(
        self,
        resource_dps: List[IterDataPipe],
        *,
        config: DatasetConfig,
        decoder: Optional[Callable[[io.IOBase], torch.Tensor]],
    ) -> IterDataPipe[Dict[str, Any]]:
        archive_dp = resource_dps[0]
        archive_dp = TarArchiveReader(archive_dp)
        archive_dp = Filter(
            archive_dp, functools.partial(self._is_data_file, config=config)
        )
        archive_dp = Mapper(archive_dp, self._unpickle)
        archive_dp = MappingIterator(archive_dp)
        images_dp, labels_dp = Demultiplexer(
            archive_dp,
            2,
            self._split_data_file,  # type: ignore[arg-type]
            drop_none=True,
            buffer_size=INFINITE_BUFFER_SIZE,
        )

        labels_dp = Mapper(labels_dp, self._remove_data_dict_key)
        labels_dp = SequenceIterator(labels_dp)
        labels_dp = Enumerator(labels_dp)
        labels_dp = Shuffler(labels_dp, buffer_size=INFINITE_BUFFER_SIZE)

        images_dp = Mapper(images_dp, self._remove_data_dict_key)
        images_dp = SequenceIterator(images_dp)
        images_dp = Enumerator(images_dp)

        dp = KeyZipper(labels_dp, images_dp, self._key_fn, buffer_size=INFINITE_BUFFER_SIZE)
        return Mapper(dp, self._collate_and_decode, fn_kwargs=dict(decoder=decoder))

    @property
    @abc.abstractmethod
    def _meta_file_name(self) -> str:
        pass

    @property
    @abc.abstractmethod
    def _categories_key(self) -> str:
        pass

    def _is_meta_file(self, data: Tuple[str, Any]) -> bool:
        path = pathlib.Path(data[0])
        return path.name == self._meta_file_name

    def generate_categories_file(
        self, root: Union[str, pathlib.Path]
    ) -> None:
        dp = self.resources(self.default_config)[0].to_datapipe(
            pathlib.Path(root) / self.name
        )
        dp = TarArchiveReader(dp)
        dp = Filter(dp, self._is_meta_file)
        dp = Mapper(dp, self._unpickle)
        categories = next(iter(dp))[self._categories_key]
        create_categories_file(HERE, self.name, categories)


class Cifar10(_CifarBase):
    @property
    def info(self) -> DatasetInfo:
        return DatasetInfo(
            "cifar10",
            categories=HERE / "cifar10.categories",
            homepage="https://www.cs.toronto.edu/~kriz/cifar.html",
        )

    def resources(self, config: DatasetConfig) -> List[OnlineResource]:
        return [
            HttpResource(
                "https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz",
                sha256="6d958be074577803d12ecdefd02955f39262c83c16fe9348329d7fe0b5c001ce",
            )
        ]

    def _is_data_file(self, data: Tuple[str, Any], *, config: DatasetConfig) -> bool:
        path = pathlib.Path(data[0])
        return path.name.startswith("data" if config.split == "train" else "test")

    def _split_data_file(self, data: Tuple[str, Any]) -> Optional[int]:
        key, _ = data
        if key == "data":
            return 0
        elif key == "labels":
            return 1
        else:
            return None

    @property
    def _meta_file_name(self) -> str:
        return "batches.meta"

    @property
    def _categories_key(self) -> str:
        return "label_names"


class Cifar100(_CifarBase):
    @property
    def info(self) -> DatasetInfo:
        return DatasetInfo(
            "cifar100",
            categories=HERE / "cifar100.categories",
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

    def _is_data_file(self, data: Tuple[str, io.IOBase], *, config: DatasetConfig) -> bool:
        path = pathlib.Path(data[0])
        return path.name == config.split

    def _split_data_file(self, data: Tuple[str, Any]) -> Optional[int]:
        key, _ = data
        if key == "data":
            return 0
        elif key == "fine_labels":
            return 1
        else:
            return None

    @property
    def _meta_file_name(self) -> str:
        return "meta"

    @property
    def _categories_key(self) -> str:
        return "fine_label_names"


if __name__ == "__main__":
    from torchvision.prototype.datasets import home

    root = home()
    Cifar10().generate_categories_file(root)
    Cifar100().generate_categories_file(root)
