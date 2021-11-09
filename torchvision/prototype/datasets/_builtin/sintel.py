import io
from typing import Any, Callable, Dict, List, Optional, Tuple, Iterator, Iterable, TypeVar
import pathlib
from functools import partial
import re

import torch
from PIL import Image
import numpy as np

from torchdata.datapipes.iter import (
    IterDataPipe,
    IterableWrapper,
    Demultiplexer,
    Mapper,
    Shuffler,
    Filter,
    Zipper,
    KeyZipper,
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

T = TypeVar("T")

FILE_NAME_PATTERN = re.compile(r"(frame|image)_(?P<idx>\d+)[.](flo|png)")

try:
    from itertools import pairwise
except ImportError:
    from itertools import tee

    def pairwise(iterable: Iterable[T]) -> Iterable[Tuple[T, T]]:
        a, b = tee(iterable)
        next(b, None)
        return zip(a, b)

class IntCategoryGrouper(IterDataPipe[Tuple[Tuple[str, T], Tuple[str, T]]]):
    def __init__(self, datapipe: IterDataPipe[Tuple[str, T]]) -> None:
        self.datapipe = datapipe

    def __iter__(self):
        for item1, item2 in pairwise(sorted(self.datapipe)):
            if pathlib.Path(item1[0]).parent != pathlib.Path(item2[0]).parent:
                continue

            yield item1, item2


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
        return [archive]

    def _collate_and_decode_sample(
        self,
        data: Tuple[str, ...],
        *,
        decoder: Optional[Callable[[io.IOBase], torch.Tensor]],
    ) -> Dict[str, Any]:
        # Read images and flo file here
        # Use decoder for images if available
        # Return dict
        flo, images = data
        img1, img2 = images

        print(img1)
        path1, buffer1 = img1
        path2, buffer2 = img2

        flow_arr = self.read_flo(flo)
        obj = Image.open(buffer1)

        return dict(
            image1=decoder(buffer1) if decoder else buffer1,
            image2=decoder(buffer2) if decoder else buffer2,
            label=flow_arr,
        )

    def _classify_train_test(self, data: Dict[str, Any], *, config: DatasetConfig):
        path = pathlib.Path(data[0])
        return config.split in str(path.parent)

    def _classify_archive(self, data: Dict[str, Any], *, config: DatasetConfig):
        path = pathlib.Path(data[0])
        path_str = str(path.absolute())
        if config.pass_ in path_str and ".png" in path_str:
            return 0
        elif ".flo" in path_str:
            return 1
        else:
            return None

    def read_flo(self, data: IterDataPipe[Tuple[Any, io.IOBase]]) -> Iterable[np.ndarray]:
        count_flo = 0
        for _, file in data:
            f = file.file_obj
            magic = np.fromfile(f, np.float32, count=1)
            if 202021.25 != magic:
                raise ValueError("Magic number incorrect. Invalid .flo file")

            w = int(np.fromfile(f, np.int32, count=1))
            h = int(np.fromfile(f, np.int32, count=1))
            _data = np.fromfile(f, np.float32, count=2 * w * h)
            count_flo += 1
            yield _data.reshape(2, h, w)

    def flows_key(self, data: Tuple[str, Any]) -> Tuple[str, int]:
        path = pathlib.Path(data[0])
        category = path.parent.name
        idx = int(FILE_NAME_PATTERN.match(path.name).group("idx"))  # type: ignore[union-attr]
        return category, idx

    def images_key(self, data: Tuple[Tuple[str, Any], Tuple[str, Any]]) -> Tuple[str, int]:
        return self.flows_key(data[0])

    def _make_datapipe(
        self,
        resource_dps: List[IterDataPipe],
        *,
        config: DatasetConfig,
        decoder: Optional[Callable[[io.IOBase], torch.Tensor]]
    ) -> IterDataPipe[Tuple[Dict[str, Any], Dict[str, Any]]]:
        dp = resource_dps[0]
        archive_dp = ZipArchiveReader(dp)

        curr_split = Filter(archive_dp, partial(self._classify_train_test, config=config))

        pass_images_dp, flo_dp = Demultiplexer(
            curr_split,
            2,
            partial(self._classify_archive, config=config),
            drop_none=True,
            buffer_size=INFINITE_BUFFER_SIZE,
        )
        flo_dp = Shuffler(flo_dp, buffer_size=INFINITE_BUFFER_SIZE)

        pass_images_dp = IntCategoryGrouper(pass_images_dp)
        zipped_dp = KeyZipper(
            flo_dp,
            pass_images_dp,
            key_fn=self.flows_key,
            ref_key_fn=self.images_key,
        )

        return Mapper(zipped_dp, self._collate_and_decode_sample, fn_kwargs=dict(decoder=decoder))
