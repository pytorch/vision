import io
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

from torchvision.prototype.datasets.utils._internal import RarArchiveReader, INFINITE_BUFFER_SIZE

import numpy as np
import torch
from torchdata.datapipes.iter import CSVParser, KeyZipper
from torch.utils.data import IterDataPipe
from torch.utils.data.datapipes.iter import (
    Filter,
    Mapper,
    ZipArchiveReader,
    Shuffler,
)
from torchvision.prototype.datasets.utils._internal import path_accessor, path_comparator
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
    """
    @property
    def info(self) -> DatasetInfo:
        return DatasetInfo(
            "ucf101",
            type=DatasetType.VIDEO,
            valid_options={'split': ["train", "test"], 'fold': ["1", "2", "3"]},
            homepage="https://www.crcv.ucf.edu/data/UCF101.php",
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
        decoder: Optional[Callable[[io.IOBase], Dict[str, Any]]]=None,
    ) -> Dict[str, Any]:
        annotations_d, file_d = data

        label = annotations_d[1]
        _path, file_handle = file_d
        file = decoder(file_handle) if decoder else file_handle
        return {"path": _path, "file": file, "target": label}

    def _filtername(self, data, *, tgt):
        return Path(data[0]).name == tgt

    def _getname(self, data):
        return Path(data[0]).name

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
        annotations_dp = Filter(annotations_dp, path_comparator("name", f"{config.split}list0{config.fold}.txt"))
        annotations_dp = CSVParser(annotations_dp, delimiter=" ")
        # COMMENT OUT FOR TESTING
        annotations_dp = Shuffler(annotations_dp, buffer_size=INFINITE_BUFFER_SIZE)

        files_dp = RarArchiveReader(files)
        dp = KeyZipper(annotations_dp, files_dp, path_accessor("name"))
        return Mapper(dp, self._collate_and_decode, fn_kwargs=dict(decoder=decoder))
