from typing import Any, Dict, List, Tuple, Iterator

import numpy as np
from torchdata.datapipes.iter import Filter, IterDataPipe, Mapper, Zipper
from torchvision.prototype.datasets.utils import Dataset, DatasetConfig, DatasetInfo, HttpResource
from torchvision.prototype.datasets.utils._internal import hint_sharding, hint_shuffling, path_comparator, read_mat
from torchvision.prototype.features import BoundingBox, EncodedImage


class _StanfordCarsLabelReader(IterDataPipe[Tuple[np.ndarray, int]]):
    def __init__(self, datapipe: IterDataPipe[Dict[str, Any]]) -> None:
        self.datapipe = datapipe

    def __iter__(self) -> Iterator[Tuple[str]]:
        for _, file in self.datapipe:
            file = iter(read_mat(file, squeeze_me=True)["annotations"])
            for line in file:
                yield line


class StanfordCars(Dataset):
    def _make_info(self) -> DatasetInfo:
        return DatasetInfo(
            name="stanford-cars",
            homepage="https://ai.stanford.edu/~jkrause/cars/car_dataset.html",
            dependencies=("scipy",),
            valid_options=dict(
                split=("test", "train"),
            ),
        )

    _URL_ROOT = "https://ai.stanford.edu/~jkrause/"
    _URLS = {
        "train": f"{_URL_ROOT}car196/cars_train.tgz",
        "test": f"{_URL_ROOT}car196/cars_test.tgz",
        "cars_test_annos_withlabels": f"{_URL_ROOT}car196/cars_test_annos_withlabels.mat",
        "car_devkit": f"{_URL_ROOT}cars/car_devkit.tgz",
    }

    _CHECKSUM = {
        "train": "b97deb463af7d58b6bfaa18b2a4de9829f0f79e8ce663dfa9261bf7810e9accd",
        "test": "bffea656d6f425cba3c91c6d83336e4c5f86c6cffd8975b0f375d3a10da8e243",
        "cars_test_annos_withlabels": "790f75be8ea34eeded134cc559332baf23e30e91367e9ddca97d26ed9b895f05",
        "car_devkit": "512b227b30e2f0a8aab9e09485786ab4479582073a144998da74d64b801fd288",
    }

    def resources(self, config: DatasetConfig) -> List[HttpResource]:
        resources = [HttpResource(self._URLS[config.split], sha256=self._CHECKSUM[config.split])]
        if config.split == "test":
            resources.append(
                HttpResource(
                    self._URLS["cars_test_annos_withlabels"], sha256=self._CHECKSUM["cars_test_annos_withlabels"]
                )
            )
        else:
            resources.append(HttpResource(url=self._URLS["car_devkit"], sha256=self._CHECKSUM["car_devkit"]))

        return resources

    def _prepare_sample(self, data: Tuple[IterDataPipe, Tuple[Any]]) -> Dict[str, Any]:
        image, target = data
        image_path, image_buffer = image
        image = EncodedImage.from_file(image_buffer)
        index = image_path[-9:-4]
        index = int(image_path[-9:-4]) - 1

        return dict(
            index=index,
            image_path=image_path,
            image=image,
            classification_label=target[4] - 1,
            bounding_box=BoundingBox(
                [
                    target[0],
                    target[1],
                    target[2],
                    target[3],
                ],
                format="xyxy",
                image_size=image.image_size,
            ),
        )

    def _make_datapipe(
        self,
        resource_dps: List[IterDataPipe],
        *,
        config: DatasetConfig,
    ) -> IterDataPipe[Dict[str, Any]]:

        images_dp, targets_dp = resource_dps
        if config.split == "train":
            targets_dp = Filter(targets_dp, path_comparator("name", "cars_train_annos.mat"))
        targets_dp = _StanfordCarsLabelReader(targets_dp)
        dp = Zipper(images_dp, targets_dp)
        dp = hint_sharding(dp)
        dp = hint_shuffling(dp)
        return Mapper(dp, self._prepare_sample)
