import abc
import functools
import io
import os.path
import pathlib
import pickle
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import numpy as np
import PIL.Image
import torch
from torch.utils.data import IterDataPipe
from torch.utils.data.datapipes.iter import (
    Demultiplexer,
    Filter,
    Mapper,
    TarArchiveReader,
    Zipper,
)

from torchvision.prototype.datasets.datapipes import MappingIterator, SequenceIterator
from torchvision.prototype.datasets.utils import (
    Dataset,
    DatasetConfig,
    DatasetInfo,
    HttpResource,
    OnlineResource,
)
from torchvision.prototype.datasets.utils._internal import create_categories_file

__all__ = ["Cifar10", "Cifar100"]

HERE = pathlib.Path(__file__).parent


class _CifarBase(Dataset):
    @abc.abstractmethod
    def _is_data_file(
        self, data: Tuple[str, io.BufferedIOBase], *, config: DatasetConfig
    ) -> Optional[int]:
        pass

    @abc.abstractmethod
    def _split_data_file(self, data: Tuple[str, Any]) -> Optional[int]:
        pass

    def _unpickle(self, data: Tuple[str, io.BytesIO]) -> Dict[str, Any]:
        _, file = data
        return pickle.load(file, encoding="latin1")

    def _parse_image(self, data: np.ndarray) -> io.BytesIO:
        image = PIL.Image.fromarray(data.reshape(3, 32, 32).transpose(1, 2, 0))
        buffer = io.BytesIO()
        image.save(buffer, format="png")
        buffer.seek(0)
        return buffer

    def _parse_label(self, data: int) -> torch.Tensor:
        return torch.tensor(data)

    def _collate_and_decode(
        self,
        data: Tuple[Any, torch.Tensor],
        *,
        decoder: Optional[Callable[[str, io.BufferedIOBase], torch.Tensor]],
    ) -> Dict[str, Any]:
        image, label = data
        if decoder:
            image = decoder(".png", image)
        return dict(image=image, label=label)

    def _make_datapipe(
        self,
        resource_dps: List[IterDataPipe],
        *,
        config: DatasetConfig,
        decoder: Optional[Callable[[str, io.BufferedIOBase], torch.Tensor]],
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
            # FIXME
            buffer_size=1_000_000,
        )

        images_dp = Mapper(images_dp, lambda data: data[1])
        images_dp = SequenceIterator(images_dp)
        images_dp = Mapper(images_dp, self._parse_image)

        labels_dp = Mapper(labels_dp, lambda data: data[1])
        labels_dp = SequenceIterator(labels_dp)
        labels_dp = Mapper(labels_dp, self._parse_label)

        dp: IterDataPipe = Zipper(images_dp, labels_dp)
        return Mapper(dp, self._collate_and_decode, fn_kwargs=dict(decoder=decoder))

    def generate_categories_file(
        self, root: Union[str, pathlib.Path], *, file_name: str, categories_key: str
    ) -> None:
        dp = self.resources(self.default_config)[0].to_datapipe(
            pathlib.Path(root) / self.name
        )
        dp = TarArchiveReader(dp)
        dp = Filter(dp, lambda data: os.path.basename(data[0]) == file_name)
        dp = Mapper(dp, self._unpickle)
        categories = next(iter(dp))[categories_key]
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

    def _is_data_file(
        self, data: Tuple[str, io.BufferedIOBase], *, config: DatasetConfig
    ) -> bool:
        path, _ = data
        name = os.path.basename(path)
        return name.startswith("data" if config.split == "train" else "test")

    def _split_data_file(self, data: Tuple[str, Any]) -> Optional[int]:
        key, _ = data
        if key == "data":
            return 0
        elif key == "labels":
            return 1
        else:
            return None

    def generate_categories_file(
        self,
        root: Union[str, pathlib.Path],
        *,
        file_name: str = "batches.meta",
        categories_key: str = "label_names",
    ) -> None:
        super().generate_categories_file(
            root, file_name=file_name, categories_key=categories_key
        )


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

    def _is_data_file(
        self, data: Tuple[str, io.BufferedIOBase], *, config: DatasetConfig
    ) -> bool:
        path, _ = data
        name = os.path.basename(path)
        return name == config.split

    def _split_data_file(self, data: Tuple[str, Any]) -> Optional[int]:
        key, _ = data
        if key == "data":
            return 0
        elif key == "fine_labels":
            return 1
        else:
            return None

    def generate_categories_file(
        self,
        root: Union[str, pathlib.Path],
        *,
        file_name: str = "meta",
        categories_key: str = "fine_label_names",
    ) -> None:
        super().generate_categories_file(
            root, file_name=file_name, categories_key=categories_key
        )


if __name__ == "__main__":
    from torchvision.prototype.datasets import home

    root = home()
    Cifar10().generate_categories_file(root)
    Cifar100().generate_categories_file(root)
