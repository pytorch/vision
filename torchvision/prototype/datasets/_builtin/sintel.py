import io
import pathlib
import re
from functools import partial
from typing import Any, Callable, Dict, List, Optional, Tuple, Iterator, Iterable, TypeVar

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

try:
    from itertools import pairwise  # type: ignore[attr-defined]
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

    _FILE_NAME_PATTERN = re.compile(r"(frame|image)_(?P<idx>\d+)[.](flo|png)")

    def _make_info(self) -> DatasetInfo:
        return DatasetInfo(
            "sintel",
            type=DatasetType.IMAGE,
            homepage="http://sintel.is.tue.mpg.de/",
            valid_options=dict(
                split=("train", "test"),
                pass_name=("clean", "final", "both"),
            ),
        )

    def resources(self, config: DatasetConfig) -> List[OnlineResource]:
        archive = HttpResource(
            "http://files.is.tue.mpg.de/sintel/MPI-Sintel-complete.zip",
            sha256="bdc80abbe6ae13f96f6aa02e04d98a251c017c025408066a00204cd2c7104c5f",
        )
        return [archive]

    def _filter_split(self, data: Tuple[str, Any], *, split: str) -> bool:
        path = pathlib.Path(data[0])
        # The dataset contains has the folder "training", while allowed options for `split` are
        # "train" and "test", we don't check for equality here ("train" != "training") and instead
        # check if split is in the folder name
        return split in path.parents[2].name

    def _filter_images(self, data: Tuple[str, Any], *, pass_name: str) -> bool:
        path = pathlib.Path(data[0])
        if pass_name == "both":
            matched = path.parents[1].name in ["clean", "final"]
        else:
            matched = path.parents[1].name == pass_name
        return matched and path.suffix == ".png"

    def _classify_archive(self, data: Tuple[str, Any], *, pass_name: str) -> Optional[int]:
        path = pathlib.Path(data[0])
        suffix = path.suffix
        if suffix == ".flo":
            return 0
        elif suffix == ".png":
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
        data_arr = np.frombuffer(data, dtype="<f4")

        # Creating a copy of the underlying array, to avoid UserWarning: "The given NumPy array
        #     is not writeable, and PyTorch does not support non-writeable tensors."
        return torch.from_numpy(np.copy(data_arr.reshape(h, w, 2).transpose(2, 0, 1)))

    def _flows_key(self, data: Tuple[str, Any]) -> Tuple[str, int]:
        path = pathlib.Path(data[0])
        category = path.parent.name
        idx = int(self._FILE_NAME_PATTERN.match(path.name).group("idx"))  # type: ignore[union-attr]
        return category, idx

    def _add_fake_flow_data(self, data: Tuple[str, Any]) -> Tuple[tuple, Tuple[str, Any]]:
        return ((None, None), data)

    def _images_key(self, data: Tuple[Tuple[str, Any], Tuple[str, Any]]) -> Tuple[str, int]:
        return self._flows_key(data[0])

    def _collate_and_decode_sample(
        self,
        data: Tuple[Tuple[str, io.IOBase], Any],
        *,
        decoder: Optional[Callable[[io.IOBase], torch.Tensor]],
        config: DatasetConfig,
    ) -> Dict[str, Any]:
        flo, images = data
        img1, img2 = images
        flow_arr = self._read_flo(flo[1]) if flo[1] else None

        path1, buffer1 = img1
        path2, buffer2 = img2

        return dict(
            image1=decoder(buffer1) if decoder else buffer1,
            image1_path=path1,
            image2=decoder(buffer2) if decoder else buffer2,
            image2_path=path2,
            flow=flow_arr,
            flow_path=flo[0],
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

        curr_split = Filter(archive_dp, self._filter_split, fn_kwargs=dict(split=config.split))
        filtered_curr_split = Filter(curr_split, self._filter_images, fn_kwargs=dict(pass_name=config.pass_name))
        if config.split == "train":
            flo_dp, pass_images_dp = Demultiplexer(
                filtered_curr_split,
                2,
                partial(self._classify_archive, pass_name=config.pass_name),
                drop_none=True,
                buffer_size=INFINITE_BUFFER_SIZE,
            )
            flo_dp = Shuffler(flo_dp, buffer_size=INFINITE_BUFFER_SIZE)
            pass_images_dp: IterDataPipe[Tuple[str, Any], Tuple[stry, Any]] = InSceneGrouper(pass_images_dp)
            zipped_dp = IterKeyZipper(
                flo_dp,
                pass_images_dp,
                key_fn=self._flows_key,
                ref_key_fn=self._images_key,
            )
        else:
            pass_images_dp = Shuffler(filtered_curr_split, buffer_size=INFINITE_BUFFER_SIZE)
            pass_images_dp = InSceneGrouper(pass_images_dp)
            zipped_dp = Mapper(pass_images_dp, self._add_fake_flow_data)

        return Mapper(zipped_dp, self._collate_and_decode_sample, fn_kwargs=dict(decoder=decoder, config=config))
