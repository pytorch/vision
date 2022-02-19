from typing import Any, List, Dict
from torchdata.datapipes.iter import (
    IterDataPipe,
    Mapper,
    Filter,
    Zipper,
)
from torchvision.prototype.features import BoundingBox, EncodedImage
from torchvision.prototype.datasets.utils import (
    Dataset,
    DatasetConfig,
    DatasetInfo,
    OnlineResource,
    HttpResource
)
from torchvision.prototype.datasets.utils._internal import (
    hint_sharding,
    hint_shuffling,
    read_mat,
    path_comparator
)


class StanfordCars(Dataset):
    def _make_info(self) -> DatasetInfo:
        return DatasetInfo(
            name="stanford_cars",
            homepage="https://ai.stanford.edu/~jkrause/cars/car_dataset.html",
            dependencies=("scipy",),
            valid_options=dict(split=("test", "train"),),
        )

    _URL_ROOT = "https://ai.stanford.edu/~jkrause/"
    _URLS = {
        "train": f"{_URL_ROOT}car196/cars_train.tgz",
        "test": f"{_URL_ROOT}car196/cars_test.tgz",
        "test_ground_truth": f"{_URL_ROOT}car196/cars_test_annos_withlabels.mat",
        "devkit": f"{_URL_ROOT}cars/car_devkit.tgz",
    }

    _CHECKSUM = {
        "train": "512b227b30e2f0a8aab9e09485786ab4479582073a144998da74d64b801fd288",
        "test": "bffea656d6f425cba3c91c6d83336e4c5f86c6cffd8975b0f375d3a10da8e243",
        "test_ground_truth": "790f75be8ea34eeded134cc559332baf23e30e91367e9ddca97d26ed9b895f05",
        "devkit": "512b227b30e2f0a8aab9e09485786ab4479582073a144998da74d64b801fd288",
    }

    def resources(self, config: DatasetConfig) -> List[OnlineResource]:
        resources = [HttpResource(self._URLS[config.split], sha256=self._CHECKSUM[config.split])]
        if config.split == "test":
            resources.append(HttpResource(self._URLS["test_ground_truth"], sha256=self._CHECKSUM["test_ground_truth"]))
        else:
            resources.append(HttpResource(url=self._URLS["devkit"], sha256=self._CHECKSUM["devkit"]))
        return resources

    def _make_datapipe(
        self,
        resource_dps: List[IterDataPipe],
        *,
        config: DatasetConfig,
    ) -> IterDataPipe[Dict[str, Any]] :

        images_dp, targets_dp = resource_dps
        print(config.split)
        if config.split == "train":
            targets_dp = Filter(targets_dp, path_comparator("name", "cars_train_annos.mat"))
        dp = Zipper(images_dp, targets_dp)
        dp = hint_sharding(dp)
        dp = hint_shuffling(dp)
        return Mapper(dp, self._read_images_and_labels)

    def _read_images_and_labels(self, data):
        image, target = data
        image_path, image_buffer = image
        labels_path, labels_buffer = target

        image = EncodedImage.from_file(image_buffer)
        labels = read_mat(labels_buffer, squeeze_me=True)["annotations"]
        index = image_path[-9:-4]
        index = int(image_path[-9:-4]) - 1

        return dict(
            index=index,
            image_path=image_path,
            image=image,
            labels_path=labels_path,
            classification_label=labels["class"][index] - 1,
            bounding_box=BoundingBox([labels["bbox_x1"][index], labels["bbox_y1"][index], labels["bbox_x2"][index], labels["bbox_y2"][index]], format="xyxy", image_size=image.image_size)
        )
