import io
from typing import Any, Callable, Dict, List, Optional, Tuple, Iterator, Iterable
import pathlib
from functools import partial
import re

import torch
from PIL import Image
import numpy as np

from torchdata.datapipes.iter import (
    IterDataPipe,
    Demultiplexer,
    Mapper,
    Shuffler,
    Zipper,
    LineReader,
    ZipArchiveReader,
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
    image_buffer_from_array,
    Decompressor,
    INFINITE_BUFFER_SIZE,
)
from torchvision import transforms


class FlowDatasetReader(IterDataPipe[torch.Tensor]):
    def __init__(
        self,
        images_datapipe: IterDataPipe[Tuple[Any, io.IOBase]],
        labels_datapipe: IterDataPipe[Tuple[Any, io.IOBase]]
    ) -> None:
        self.images_datapipe = images_datapipe
        self.labels_datapipe = labels_datapipe

    def __iter__(self) -> Iterator[torch.Tensor]:
        count_images = 0
        for _, file in self.images_datapipe:
            yield self._read_image(file)

        count_labels = 0
        for _, file in self.labels_datapipe:
            count_labels += 1

        print(count_images, count_labels)
        pass


class SINTEL(Dataset):
    def _make_info(self) -> DatasetInfo:
        return DatasetInfo(
            "sintel",
            type=DatasetType.IMAGE,
            homepage="",
            valid_options=dict(
                split=("train", "test"),
                pass_=("clean", "final"),
            )
        )

    def resources(self, config: DatasetConfig) -> List[OnlineResource]:
        archive = HttpResource(
            "http://sintel.cs.washington.edu/MPI-Sintel-complete.zip",
            sha256="",
        )
        # return [training_images_archive, testing_images_archive], self._collate_and_decode_sample, fn_kwargs=dict(decoder=decoder))
        return [archive]

    def _collate_and_decode_sample(
        self,
        data: Tuple[str, ...],
        *,
        decoder: Optional[Callable[[io.IOBase], torch.Tensor]],
    ) -> Dict[str, Any]:
        return data

    def _classify_train_test(self, data: Dict[str, Any], *, config: DatasetConfig):
        path = pathlib.Path(data[0])
        path_str = str(path.absolute())
        if "/training/" in path_str:
            return 0
        elif "/test/" in path_str:
            return 1
        else:
            return None

    def _classify_archive(self, data: Dict[str, Any], *, config: DatasetConfig):
        path = pathlib.Path(data[0])
        path_str = str(path.absolute())
        if config.pass_ in path_str and ".png" in path_str:
            return 0
        elif ".flo" in path_str:
            return 1
        else:
            return None

    def read_images(self, data: IterDataPipe[Tuple[Any, io.IOBase]]) -> Iterable[torch.Tensor]:
        count_images = 0
        for _, file in data:
            img = Image.open(file)
            to_tensor = transforms.ToTensor()
            count_images += 1
            yield to_tensor(img)

    def read_flo(self, data: IterDataPipe[Tuple[Any, io.IOBase]], config) -> Iterable[np.ndarray]:
        count_flo = 0
        for _, file in data:
            with open(file.name, "rb") as f:
                magic = np.fromfile(f, np.float32, count=1)
                if 202021.25 != magic:
                    raise ValueError("Magic number incorrect. Invalid .flo file")

                w = int(np.fromfile(f, np.int32, count=1))
                h = int(np.fromfile(f, np.int32, count=1))
                _data = np.fromfile(f, np.float32, count=2 * w * h)
                count_flo += 1
                yield _data.reshape(2, h, w)

    def _make_datapipe(
        self,
        resource_dps: List[IterDataPipe],
        *,
        config: DatasetConfig,
        decoder: Optional[Callable[[io.IOBase], torch.Tensor]]
    ) -> IterDataPipe[Dict[str, Any]]:
        dp = resource_dps[0]
        archive_dp = ZipArchiveReader(dp)

        train_dp, test_dp = Demultiplexer(
            archive_dp,
            2,
            partial(self._classify_train_test, config=config),
            drop_none=True,
            buffer_size=INFINITE_BUFFER_SIZE,
        )

        if config.split == "train":
            curr_split = train_dp
        else:
            curr_split = test_dp

        pass_images_dp, flo_dp = Demultiplexer(
            curr_split,
            2,
            partial(self._classify_archive, config=config),
            drop_none=True,
            buffer_size=INFINITE_BUFFER_SIZE,
        )
        pass_images_dp = self.read_images(pass_images_dp)
        flo_dp = self.read_flo(flo_dp, config)

        images_dp = Shuffler(pass_images_dp, buffer_size=INFINITE_BUFFER_SIZE)
        flo_dp = Shuffler(flo_dp, buffer_size=INFINITE_BUFFER_SIZE)
        zipped_dp = Zipper(images_dp, flo_dp)
        return Mapper(zipped_dp, self._collate_and_decode_sample, fn_kwargs=dict(decoder=decoder))
