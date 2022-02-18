
from cProfile import label
from hashlib import sha256
import io
from collections import namedtuple
from pickletools import read_string1
from typing import Any, Callable, Dict, List, Optional, Tuple, Iterator
import pathlib
from torchdata.datapipes.iter import (
    IterDataPipe,
    Mapper,
    Filter,
    )

from torchvision.prototype.datasets.utils._internal import read_mat,hint_sharding, hint_shuffling, path_comparator
import torch
from torchdata.datapipes.iter import IterDataPipe, Mapper, Zipper
from torchvision.prototype import features
from torchvision.prototype.features import Label, BoundingBox, _Feature, EncodedImage
from torchvision.prototype.datasets.utils import (
    Dataset,
    DatasetConfig,
    DatasetInfo,
    OnlineResource,
    #DatasetType,
    HttpResource
)


from torchvision.prototype.datasets.utils._internal import (
    hint_sharding,
    hint_shuffling,
)
from torchvision.prototype.features import Label


class StanfordCars(Dataset):
    def _make_info(self) -> DatasetInfo:
        return DatasetInfo(
            name="stanford_cars",
            #type=DatasetType.IMAGE,
            homepage="https://ai.stanford.edu/~jkrause/cars/car_dataset.html",
            dependencies=("scipy",),
            valid_options=dict(split=("test","train"),
                               ),
        )

    _URL_ROOT = "https://ai.stanford.edu/~jkrause/"
    _URLS = {
        "train":f"{_URL_ROOT}car196/cars_train.tgz",
        "test":f"{_URL_ROOT}car196/cars_test.tgz",
        "test_ground_truth":f"{_URL_ROOT}car196/cars_test_annos_withlabels.mat",
        "devkit":f"{_URL_ROOT}cars/car_devkit.tgz"
    }

    _CHECKSUM = {
        "train": "512b227b30e2f0a8aab9e09485786ab4479582073a144998da74d64b801fd288",
        "test": "bffea656d6f425cba3c91c6d83336e4c5f86c6cffd8975b0f375d3a10da8e243",
        "test_ground_truth": "790f75be8ea34eeded134cc559332baf23e30e91367e9ddca97d26ed9b895f05",
        "devkit": "512b227b30e2f0a8aab9e09485786ab4479582073a144998da74d64b801fd288",
    }

    def resources(self, config: DatasetConfig) -> List[OnlineResource]:
        resources = [HttpResource(self._URLS[config.split] , sha256=self._CHECKSUM[config.split])]
        print(len(resources))
        if config.split == "test":
            resources.append(HttpResource(self._URLS["test_ground_truth"],sha256=self._CHECKSUM["test_ground_truth"]))
        else:
            resources.append(HttpResource(url=self._URLS["devkit"],sha256=self._CHECKSUM["devkit"]))
        return resources

    def _make_datapipe(
        self,
        resource_dps: List[IterDataPipe],
        *,
        config: DatasetConfig,
    ) -> IterDataPipe[Dict[str, Any]] :

        images_dp, targets_dp = resource_dps

        if config.split == "train":
            targets_dp = Filter(targets_dp, path_comparator("name", "cars_train_annos.mat"))


        #right now our both dps are of .tgz and .mat format; for all configs(train and test.)
        targets_dp = Mapper(targets_dp,self._read_labels)
        dp =Zipper(images_dp,targets_dp)
        dp=hint_sharding(dp)
        dp=hint_shuffling(dp)


        return Mapper(dp,self._collate_and_decode) 

    def _collate_and_decode(self, data: Tuple[Any, Any]) -> Dict[str, Any]:
        image, target = data  # They're both numpy arrays at this point

        image_path, image_buffer = image

        image = EncodedImage.from_file(image_buffer)
       

    def _read_labels(self,data):
        labels = data[0]
        labels = read_mat(labels,squeeze_me=True)
        labels=labels["annotations"]
        print("\n"*10)
        print(len(labels["class"]))
        bboxs_and_class = zip(labels["bbox_x1"],labels["bbox_y1"],labels["bbox_x2"],labels["bbox_y2"],labels["class"])

        return  bboxs_and_class
