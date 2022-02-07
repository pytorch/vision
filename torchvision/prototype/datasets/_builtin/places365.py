import io
from typing import Any, Callable, Dict, List, Optional

import torch
from torchdata.datapipes.iter import (
    IterDataPipe,
    Mapper,
    Filter,
    IterKeyZipper,
)
from torchvision.prototype.datasets.utils import (
    Dataset,
    DatasetConfig,
    DatasetInfo,
    HttpResource,
    OnlineResource,
    DatasetType,
)

from torchvision.prototype.datasets.utils._internal import INFINITE_BUFFER_SIZE, read_mat, hint_sharding, hint_shuffling

class Places365Standard(Dataset):
    def _make_info(self) -> DatasetInfo:
        return DatasetInfo(
            name="places365standard",
            type=DatasetType.IMAGE,
            homepage="http://places2.csail.mit.edu/index.html",
        )
    
    def resources(self, config: DatasetConfig) -> List[OnlineResource]:
        train_images = HttpResource(
            "http://data.csail.mit.edu/places/places365/train_256_places365standard.tar",
            sha256="",
        )
        val_images = HttpResource(
            "http://data.csail.mit.edu/places/places365/val_256.tar",
            sha256="",
        )
        test_images = HttpResource(
            "http://data.csail.mit.edu/places/places365/test_256.tar",
            sha256="",
        )
        file_list = (
            "http://data.csail.mit.edu/places/places365/filelist_places365-standard.tar",
            sha256="",
         )
        return [train_images, val_images, test_images, file_list]
    
    

class Places365Challenge(Dataset):
    def _make_info(self) -> DatasetInfo:
        return DatasetInfo(
            name="places365challenge",
            type=DatasetType.IMAGE,
            homepage="http://places2.csail.mit.edu/index.html",
        )
    
     def resources(self, config: DatasetConfig) -> List[OnlineResource]:
        train_images = HttpResource(
            "http://data.csail.mit.edu/places/places365/train_large_places365challenge.tar",
            sha256="",
        )
        val_images = HttpResource(
            "http://data.csail.mit.edu/places/places365/val_256.tar",
            sha256="",
        )
        test_images = HttpResource(
            "http://data.csail.mit.edu/places/places365/test_256.tar",
            sha256="",
        )
        file_list = (
            "http://data.csail.mit.edu/places/places365/filelist_places365-challenge.tar",
            sha256="",
         )
        return [train_images, val_images, test_images, file_list] 