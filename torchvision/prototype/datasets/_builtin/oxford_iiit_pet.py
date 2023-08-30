import enum
import pathlib
from typing import Any, BinaryIO, Dict, List, Optional, Tuple, Union

from torchdata.datapipes.iter import CSVDictParser, Demultiplexer, Filter, IterDataPipe, IterKeyZipper, Mapper
from torchvision.prototype.datasets.utils import Dataset, EncodedImage, HttpResource, OnlineResource
from torchvision.prototype.datasets.utils._internal import (
    getitem,
    hint_sharding,
    hint_shuffling,
    INFINITE_BUFFER_SIZE,
    path_accessor,
    path_comparator,
    read_categories_file,
)
from torchvision.prototype.tv_tensors import Label

from .._api import register_dataset, register_info


NAME = "oxford-iiit-pet"


class OxfordIIITPetDemux(enum.IntEnum):
    SPLIT_AND_CLASSIFICATION = 0
    SEGMENTATIONS = 1


@register_info(NAME)
def _info() -> Dict[str, Any]:
    return dict(categories=read_categories_file(NAME))


@register_dataset(NAME)
class OxfordIIITPet(Dataset):
    """Oxford IIIT Pet Dataset
    homepage="https://www.robots.ox.ac.uk/~vgg/data/pets/",
    """

    def __init__(
        self, root: Union[str, pathlib.Path], *, split: str = "trainval", skip_integrity_check: bool = False
    ) -> None:
        self._split = self._verify_str_arg(split, "split", {"trainval", "test"})
        self._categories = _info()["categories"]
        super().__init__(root, skip_integrity_check=skip_integrity_check)

    def _resources(self) -> List[OnlineResource]:
        images = HttpResource(
            "https://www.robots.ox.ac.uk/~vgg/data/pets/data/images.tar.gz",
            sha256="67195c5e1c01f1ab5f9b6a5d22b8c27a580d896ece458917e61d459337fa318d",
            preprocess="decompress",
        )
        anns = HttpResource(
            "https://www.robots.ox.ac.uk/~vgg/data/pets/data/annotations.tar.gz",
            sha256="52425fb6de5c424942b7626b428656fcbd798db970a937df61750c0f1d358e91",
            preprocess="decompress",
        )
        return [images, anns]

    def _classify_anns(self, data: Tuple[str, Any]) -> Optional[int]:
        return {
            "annotations": OxfordIIITPetDemux.SPLIT_AND_CLASSIFICATION,
            "trimaps": OxfordIIITPetDemux.SEGMENTATIONS,
        }.get(pathlib.Path(data[0]).parent.name)

    def _filter_images(self, data: Tuple[str, Any]) -> bool:
        return pathlib.Path(data[0]).suffix == ".jpg"

    def _filter_segmentations(self, data: Tuple[str, Any]) -> bool:
        return not pathlib.Path(data[0]).name.startswith(".")

    def _prepare_sample(
        self, data: Tuple[Tuple[Dict[str, str], Tuple[str, BinaryIO]], Tuple[str, BinaryIO]]
    ) -> Dict[str, Any]:
        ann_data, image_data = data
        classification_data, segmentation_data = ann_data
        segmentation_path, segmentation_buffer = segmentation_data
        image_path, image_buffer = image_data

        return dict(
            label=Label(int(classification_data["label"]) - 1, categories=self._categories),
            species="cat" if classification_data["species"] == "1" else "dog",
            segmentation_path=segmentation_path,
            segmentation=EncodedImage.from_file(segmentation_buffer),
            image_path=image_path,
            image=EncodedImage.from_file(image_buffer),
        )

    def _datapipe(self, resource_dps: List[IterDataPipe]) -> IterDataPipe[Dict[str, Any]]:
        images_dp, anns_dp = resource_dps

        images_dp = Filter(images_dp, self._filter_images)

        split_and_classification_dp, segmentations_dp = Demultiplexer(
            anns_dp,
            2,
            self._classify_anns,
            drop_none=True,
            buffer_size=INFINITE_BUFFER_SIZE,
        )

        split_and_classification_dp = Filter(split_and_classification_dp, path_comparator("name", f"{self._split}.txt"))
        split_and_classification_dp = CSVDictParser(
            split_and_classification_dp, fieldnames=("image_id", "label", "species"), delimiter=" "
        )
        split_and_classification_dp = hint_shuffling(split_and_classification_dp)
        split_and_classification_dp = hint_sharding(split_and_classification_dp)

        segmentations_dp = Filter(segmentations_dp, self._filter_segmentations)

        anns_dp = IterKeyZipper(
            split_and_classification_dp,
            segmentations_dp,
            key_fn=getitem("image_id"),
            ref_key_fn=path_accessor("stem"),
            buffer_size=INFINITE_BUFFER_SIZE,
        )

        dp = IterKeyZipper(
            anns_dp,
            images_dp,
            key_fn=getitem(0, "image_id"),
            ref_key_fn=path_accessor("stem"),
            buffer_size=INFINITE_BUFFER_SIZE,
        )
        return Mapper(dp, self._prepare_sample)

    def _filter_split_and_classification_anns(self, data: Tuple[str, Any]) -> bool:
        return self._classify_anns(data) == OxfordIIITPetDemux.SPLIT_AND_CLASSIFICATION

    def _generate_categories(self) -> List[str]:
        resources = self._resources()

        dp = resources[1].load(self._root)
        dp = Filter(dp, self._filter_split_and_classification_anns)
        dp = Filter(dp, path_comparator("name", "trainval.txt"))
        dp = CSVDictParser(dp, fieldnames=("image_id", "label"), delimiter=" ")

        raw_categories_and_labels = {(data["image_id"].rsplit("_", 1)[0], data["label"]) for data in dp}
        raw_categories, _ = zip(
            *sorted(raw_categories_and_labels, key=lambda raw_category_and_label: int(raw_category_and_label[1]))
        )
        return [" ".join(part.title() for part in raw_category.split("_")) for raw_category in raw_categories]

    def __len__(self) -> int:
        return 3_680 if self._split == "trainval" else 3_669
