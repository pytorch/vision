# Sintel Optical Flow Dataset
import io
from typing import Any, Callable, Dict, List, Optional, Tuple, Iterator
import pathlib

import torch
from torchdata.datapipes.iter import (
    IterDataPipe,
    Demultiplexer,
    Mapper,
    Shuffler,
    Zipper,
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


# WIP
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
            count_images += 1

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
            ),
        )

    def resources(self, config: DatasetConfig) -> List[OnlineResource]:
        # training_images_archive = HttpResource(
        #     "",
        #     sha256="",
        # )
        # testing_images_archive = HttpResource(
        #     "",
        #     sha256="",
        # )
        archive = HttpResource(
            "http://sintel.cs.washington.edu/MPI-Sintel-complete.zip",
            sha256="",
        )
        # return [training_images_archive, testing_images_archive]
        return [archive]

    def _collate_and_decode_sample(
        self,
        data: Tuple[str, ...],
        *,
        decoder: Optional[Callable[[io.IOBase], torch.Tensor]],
    ) -> Dict[str, Any]:
        print(data)

    def _classify_archive(self, data: Dict[str, Any]):
        path = pathlib.Path(data[0])
        if ".png" in path.name:
            return 0
        elif ".flo" in path.name:
            return 1
        else:
            return None

    def _make_datapipe(
        self,
        resource_dps: List[IterDataPipe],
        *,
        config: DatasetConfig,
        decoder: Optional[Callable[[io.IOBase], torch.Tensor]]
    ) -> IterDataPipe[Dict[str, Any]]:
        dp = resource_dps[0]
        dp = ZipArchiveReader(dp)
        images_dp, flo_dp = Demultiplexer(
            dp,
            2,
            self._classify_archive,
            drop_none=True,
            buffer_size=INFINITE_BUFFER_SIZE,
        )

        # images_dp = Decompressor(images_dp)
        # flo_dp = Decompressor(flo_dp)
        # images_dp, flo_dp = FlowDatasetReader(images_dp, flo_dp)

        # dp = Zipper(images_dp, flo_dp)
        images_dp = Shuffler(images_dp, buffer_size=INFINITE_BUFFER_SIZE)
        flo_dp = Shuffler(flo_dp, buffer_size=INFINITE_BUFFER_SIZE)
        # images_dp =  Mapper(images_dp, self._collate_and_decode_sample, fn_kwargs=dict(decoder=decoder))
        return [images_dp, flo_dp]
