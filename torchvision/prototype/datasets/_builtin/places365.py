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
            dependencies=("resizeimage","Pillow",),
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
    
    def _make_datapipe(
        self,
        resource_dps: List[IterDataPipe],
        *,
        config: DatasetConfig,
        decoder: Optional[Callable[[io.IOBase], torch.Tensor]],
    ) -> IterDataPipe[Dict[str, Any]]:
        train_images_dp, val_images_dp, test_images_dp,file_list_dp= resource_dps

        #images_dp = Filter(images_dp, self._is_not_background_image)
        train_images_dp = hint_sharding(train_images_dp)
        train_images_dp = hint_shuffling(train_images_dp)
        val_images_dp = hint_sharding(val_images_dp)
        val_images_dp = hint_shuffling(val_images_dp)
        test_images_dp = hint_sharding(test_images_dp)
        test_images_dp = hint_shuffling(test_images_dp)
        
        #anns_dp = Filter(anns_dp, self._is_ann)

        dp = IterKeyZipper(
            train_images_dp,
            val_images_dp,
            test_images_dp,
            #key_fn=self._images_key_fn,
            #ref_key_fn=self._anns_key_fn,
            buffer_size=INFINITE_BUFFER_SIZE,
            keep_key=True,
        )
        return resource_dps[0]#Mapper(dp,functools.partial(self._collate_and_decode_sample, decoder=decoder))

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
python setup.py develop            sha256="",
        )
        file_list = (
            "http://data.csail.mit.edu/places/places365/filelist_places365-challenge.tar",
            sha256="",
         )
        return [train_images, val_images, test_images, file_list]
    
    def _make_datapipe(
        self,
        resource_dps: List[IterDataPipe],
        *,
        config: DatasetConfig,
        decoder: Optional[Callable[[io.IOBase], torch.Tensor]],
    ) -> IterDataPipe[Dict[str, Any]]:
        train_images_dp, val_images_dp, test_images_dp,file_list_dp= resource_dps

        #images_dp = Filter(images_dp, self._is_not_background_image)
        train_images_dp = hint_sharding(train_images_dp)
        train_images_dp = hint_shuffling(train_images_dp)
        val_images_dp = hint_sharding(val_images_dp)
        val_images_dp = hint_shuffling(val_images_dp)
        test_images_dp = hint_sharding(test_images_dp)
        test_images_dp = hint_shuffling(test_images_dp)

        #anns_dp = Filter(anns_dp, self._is_ann)

        dp = IterKeyZipper(
            train_images_dp,
            val_images_dp,
            test_images_dp,
            #key_fn=self._images_key_fn,
            #ref_key_fn=self._anns_key_fn,
            buffer_size=INFINITE_BUFFER_SIZE,
            keep_key=True,
        )
        return resource_dps[0]#Mapper(dp, functools.partial(self._collate_and_decode_sample, decoder=decoder))