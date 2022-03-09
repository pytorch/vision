import pathlib
from typing import Any, Dict, List, Tuple, Iterator, BinaryIO

from torchdata.datapipes.iter import Filter, IterDataPipe, Mapper, Zipper
from torchvision.prototype.datasets.utils import Dataset, DatasetConfig, DatasetInfo, HttpResource, OnlineResource
from torchvision.prototype.datasets.utils._internal import hint_sharding, hint_shuffling, path_comparator, read_mat
from torchvision.prototype.features import BoundingBox, EncodedImage, Label


class StanfordCarsLabelReader(IterDataPipe[Tuple[int, int, int, int, int, str]]):
    def __init__(self, datapipe: IterDataPipe[Dict[str, Any]]) -> None:
        self.datapipe = datapipe

    def __iter__(self) -> Iterator[Tuple[int, int, int, int, int, str]]:
        for _, file in self.datapipe:
            data = read_mat(file, squeeze_me=True)
            for ann in data["annotations"]:
                yield tuple(ann)  # type: ignore[misc]


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

    def resources(self, config: DatasetConfig) -> List[OnlineResource]:
        resources: List[OnlineResource] = [HttpResource(self._URLS[config.split], sha256=self._CHECKSUM[config.split])]
        if config.split == "train":
            resources.append(HttpResource(url=self._URLS["car_devkit"], sha256=self._CHECKSUM["car_devkit"]))

        else:
            resources.append(
                HttpResource(
                    self._URLS["cars_test_annos_withlabels"], sha256=self._CHECKSUM["cars_test_annos_withlabels"]
                )
            )
        return resources

    def _prepare_sample(self, data: Tuple[Tuple[str, BinaryIO], Tuple[int, int, int, int, int, str]]) -> Dict[str, Any]:
        image, target = data
        path, buffer = image
        image = EncodedImage.from_file(buffer)

        return dict(
            path=path,
            image=image,
            label=Label(target[4] - 1, categories=self.categories),
            bounding_box=BoundingBox(target[:4], format="xyxy", image_size=image.image_size),
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
        targets_dp = StanfordCarsLabelReader(targets_dp)
        dp = Zipper(images_dp, targets_dp)
        dp = hint_sharding(dp)
        dp = hint_shuffling(dp)
        return Mapper(dp, self._prepare_sample)

    def _generate_categories(self, root: pathlib.Path) -> List[str]:
        config = self.info.make_config(split="train")
        resources = self.resources(config)

        devkit_dp = resources[1].load(root)
        meta_dp = Filter(devkit_dp, path_comparator("name", "cars_meta.mat"))
        _, meta_file = next(iter(meta_dp))

        return list(read_mat(meta_file, squeeze_me=True)["class_names"])
