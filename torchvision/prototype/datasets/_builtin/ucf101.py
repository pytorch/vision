import abc
import functools
import io
import pathlib
from pathlib import Path
import pickle
from typing import Any, Callable, Dict, List, Optional, Tuple, Union, Iterator

from torchvision.prototype.datasets.utils._internal import RarArchiveReader, INFINITE_BUFFER_SIZE

import numpy as np
import torch
from torchdata.datapipes.iter import CSVParser, KeyZipper
from torch.utils.data import IterDataPipe
from torch.utils.data.datapipes.iter import (
    Filter,
    Mapper,
    TarArchiveReader,
    ZipArchiveReader,
    Shuffler,
)
from torchdata.datapipes.iter.util.combining import KeyZipperIterDataPipe
from torchvision.prototype.datasets.decoder import raw
from torchvision.prototype.datasets.utils import (
    Dataset,
    DatasetConfig,
    DatasetInfo,
    HttpResource,
    OnlineResource,
    DatasetType,
)


class ucf101(Dataset):
    """This is a base datapipe that returns a file handler of the video.
    What we want to do is implement either several decoder options or additional
    datapipe extensions to make this work.

    We would want 3 different outcomes:
        0. Datapipe that returns file handle (decoder=None)
        1. Datapipe that simply returns keyframes of the video
        2. Datapipe that returns batched clips
        3. Datapipe that returns a number of random frames from the video.

    Args:
        Dataset ([type]): [description]

    Returns:
        [type]: [description]
    """
    @property
    def info(self) -> DatasetInfo:
        return DatasetInfo(
            "ucf101",
            type=DatasetType.VIDEO,
            valid_options={'split': ["train", "test"], 'fold': ["1", "2", "3"]}
            # categories=HERE / "ucf101.categories",
            # homepage="https://www.cs.toronto.edu/~kriz/cifar.html",
        )
    
    def resources(self, config: DatasetConfig) -> List[OnlineResource]:
        return [
            HttpResource(
                "https://www.crcv.ucf.edu/data/UCF101/UCF101TrainTestSplits-RecognitionTask.zip",
                sha256="",
            ),
            HttpResource(
                "https://www.crcv.ucf.edu/data/UCF101/UCF101.rar",
                sha256="",
            )
        ]

    def _collate_and_decode(
        self,
        data: Tuple[np.ndarray, int],
        *,
        decoder: Optional[Callable[[io.IOBase], Dict[str, Any]]],
    ) -> Dict[str, Any]:
        annotations_d, file_d = data

        label = annotations_d[1]
        _path, file_handle = file_d

        if decoder.__name__ == "av_kf":
            pass
        else:
            # by default return just a file handle
            return {"path": _path, "file": file_handle, "target":label}
        


    def _make_datapipe(
        self,
        resource_dps: List[IterDataPipe],
        *,
        config: DatasetConfig,
        decoder: Optional[Callable[[io.IOBase], torch.Tensor]],
    ) -> IterDataPipe[Dict[str, Any]]:
        
        annotations = resource_dps[0]
        files = resource_dps[1]

        annotations_dp = ZipArchiveReader(annotations)
        annotations_dp = Filter(annotations_dp, lambda x: Path(x[0]).name == f"{config.split}list0{config.fold}.txt")
        annotations_dp = CSVParser(annotations_dp, delimiter=" ")
        # COMMENT FOR TESTING
        # annotations_dp = Shuffler(annotations_dp, buffer_size=INFINITE_BUFFER_SIZE)

        
        files_dp = RarArchiveReader(files)
        dp = KeyZipper(annotations_dp, files_dp, lambda x: Path(x[0]).name, lambda y: Path(y[0]).name)        
        return Mapper(dp, self._collate_and_decode, fn_kwargs=dict(decoder=decoder))