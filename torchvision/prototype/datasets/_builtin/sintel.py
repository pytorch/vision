import io
import pathlib
import re
from functools import partial
from typing import Any, Callable, Dict, List, Optional, Tuple, Iterator, Iterable, TypeVar, Union

import numpy as np
import torch
from torchdata.datapipes.iter import (
    IterDataPipe,
    Demultiplexer,
    Mapper,
    Shuffler,
    Filter,
    IterKeyZipper,
    ZipArchiveReader,
)
from torchvision.prototype.datasets.utils import (
    Dataset,
    DatasetConfig,
    DatasetInfo,
    HttpResource,
    OnlineResource,
    DatasetType,
)
from torchvision.prototype.datasets.utils._internal import INFINITE_BUFFER_SIZE

T = TypeVar("T")

FILE_NAME_PATTERN = re.compile(r"(frame|image)_(?P<idx>\d+)[.](flo|png)")

try:
    from itertools import pairwise  # type: ignore
except ImportError:
    from itertools import tee

    def pairwise(iterable: Iterable[T]) -> Iterable[Tuple[T, T]]:
        a, b = tee(iterable)
        next(b, None)
        return zip(a, b)


class InSceneGrouper(IterDataPipe[Tuple[Tuple[str, T], Tuple[str, T]]]):
    def __init__(self, datapipe: IterDataPipe[Tuple[str, T]]) -> None:
        self.datapipe = datapipe

    def __iter__(self) -> Iterator[Tuple[Tuple[str, Any], Tuple[str, Any]]]:
        for item1, item2 in pairwise(sorted(self.datapipe)):
            if pathlib.Path(item1[0]).parent != pathlib.Path(item2[0]).parent:
                continue

            yield item1, item2


class SINTEL(Dataset):
    def _make_info(self) -> DatasetInfo:
        return DatasetInfo(
            "sintel",
            type=DatasetType.IMAGE,
            homepage="http://sintel.is.tue.mpg.de/",
            valid_options=dict(
                split=("train", "test"),
                pass_name=("clean", "final"),
            ),
        )

    def resources(self, config: DatasetConfig) -> List[OnlineResource]:
        archive = HttpResource(
            "http://files.is.tue.mpg.de/sintel/MPI-Sintel-complete.zip",
            sha256="bdc80abbe6ae13f96f6aa02e04d98a251c017c025408066a00204cd2c7104c5f",
        )
        return [archive]

    def _filter_split(self, data: Tuple[str, Any], *, config: DatasetConfig) -> bool:
        path = pathlib.Path(data[0])
        return config.split in str(path.parent)

    def _classify_archive(self, data: Tuple[str, Any], *, config: DatasetConfig) -> Optional[int]:
        path = pathlib.Path(data[0])
        if config.pass_ == path.parent.parent.name and path.suffix == ".png":
            return 0
        elif path.suffix == ".flo":
            return 1
        else:
            return None

    def _read_flo(self, file: io.IOBase) -> torch.Tensor:
        magic = file.read(4)
        if magic != b"PIEH":
            raise ValueError("Magic number incorrect. Invalid .flo file")
        w = int.from_bytes(file.read(4), "little")
        h = int.from_bytes(file.read(4), "little")
        data = file.read(2 * w * h * 4)
        data = np.frombuffer(data, dtype=np.float32)
        return data.reshape(h, w, 2).transpose(2, 0, 1)

    def _flows_key(self, data: Tuple[str, Any]) -> Tuple[str, int]:
        path = pathlib.Path(data[0])
        category = path.parent.name
        idx = int(FILE_NAME_PATTERN.match(path.name).group("idx"))  # type: ignore[union-attr]
        return category, idx

    def _images_key(self, data: Tuple[Tuple[str, Any], Tuple[str, Any]]) -> Tuple[str, int]:
        return self._flows_key(data[0])

    def _collate_and_decode_sample(
        self,
        data: Tuple[Tuple[str, io.IOBase], Any],
        *,
        decoder: Optional[Callable[[io.IOBase], torch.Tensor]],
        config: DatasetConfig,
    ) -> Dict[str, Any]:
        if config.split == "train":
            flo, images = data
            img1, img2 = images
            flow_arr = self._read_flo(flo[1])
            del images, flo
        else:
            # When split is `test`
            img1, img2 = data

        del data

        path1, buffer1 = img1
        path2, buffer2 = img2

        return dict(
            image1=decoder(buffer1) if decoder else buffer1,
            image2=decoder(buffer2) if decoder else buffer2,
            flow=flow_arr if config.split == "train" else None,
        )

    def _make_datapipe(
        self,
        resource_dps: List[IterDataPipe],
        *,
        config: DatasetConfig,
        decoder: Optional[Callable[[io.IOBase], torch.Tensor]],
    ) -> IterDataPipe[Dict[str, Any]]:
        dp = resource_dps[0]
        archive_dp = ZipArchiveReader(dp)

        curr_split = Filter(archive_dp, self._classify_train_test, fn_kwargs=dict(split=split))

        pass_images_dp, flo_dp = Demultiplexer(
            curr_split,
            2,
            partial(self._classify_archive, config=config),
            drop_none=True,
            buffer_size=INFINITE_BUFFER_SIZE,
        )
        flo_dp = Shuffler(flo_dp, buffer_size=INFINITE_BUFFER_SIZE)

        pass_images_dp: IterDataPipe[Tuple[str, Any], Tuple[stry, Any]] = IntCategoryGrouper(pass_images_dp)
        if config.split == "train":
            zipped_dp = IterKeyZipper(
                flo_dp,
                pass_images_dp,
                key_fn=self._flows_key,
                ref_key_fn=self._images_key,
            )
            return Mapper(zipped_dp, self._collate_and_decode_sample, fn_kwargs=dict(decoder=decoder, config=config))
        else:
            # When split is `test`, flo_dp will be empty and thus should not be zipped with pass_images_dp
            return Mapper(
                pass_images_dp, self._collate_and_decode_sample, fn_kwargs=dict(decoder=decoder, config=config)
            )
