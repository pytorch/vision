import enum
import pathlib
from typing import Any, Dict, List, Optional, Tuple, BinaryIO

from torchdata.datapipes.iter import IterDataPipe, Mapper, Filter, IterKeyZipper, Demultiplexer, CSVDictParser
from torchvision.prototype.datasets.utils import (
    Dataset,
    DatasetConfig,
    DatasetInfo,
    HttpResource,
    OnlineResource,
)
from torchvision.prototype.datasets.utils._internal import (
    INFINITE_BUFFER_SIZE,
    hint_sharding,
    hint_shuffling,
    getitem,
    path_accessor,
    path_comparator,
)
from torchvision.prototype.features import Label, EncodedImage


class OxfordIITPetDemux(enum.IntEnum):
    SPLIT_AND_CLASSIFICATION = 0
    SEGMENTATIONS = 1


class OxfordIITPet(Dataset):
    def _make_info(self) -> DatasetInfo:
        return DatasetInfo(
            "oxford-iiit-pet",
            homepage="https://www.robots.ox.ac.uk/~vgg/data/pets/",
            valid_options=dict(
                split=("trainval", "test"),
            ),
        )

    def resources(self, config: DatasetConfig) -> List[OnlineResource]:
        images = HttpResource(
            "https://www.robots.ox.ac.uk/~vgg/data/pets/data/images.tar.gz",
            sha256="67195c5e1c01f1ab5f9b6a5d22b8c27a580d896ece458917e61d459337fa318d",
            decompress=True,
        )
        anns = HttpResource(
            "https://www.robots.ox.ac.uk/~vgg/data/pets/data/annotations.tar.gz",
            sha256="52425fb6de5c424942b7626b428656fcbd798db970a937df61750c0f1d358e91",
            decompress=True,
        )
        return [images, anns]

    def _classify_anns(self, data: Tuple[str, Any]) -> Optional[int]:
        return {
            "annotations": OxfordIITPetDemux.SPLIT_AND_CLASSIFICATION,
            "trimaps": OxfordIITPetDemux.SEGMENTATIONS,
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
            label=Label(int(classification_data["label"]) - 1, categories=self.categories),
            species="cat" if classification_data["species"] == "1" else "dog",
            segmentation_path=segmentation_path,
            segmentation=EncodedImage.from_file(segmentation_buffer),
            image_path=image_path,
            image=EncodedImage.from_file(image_buffer),
        )

    def _make_datapipe(
        self, resource_dps: List[IterDataPipe], *, config: DatasetConfig
    ) -> IterDataPipe[Dict[str, Any]]:
        images_dp, anns_dp = resource_dps

        images_dp = Filter(images_dp, self._filter_images)

        split_and_classification_dp, segmentations_dp = Demultiplexer(
            anns_dp,
            2,
            self._classify_anns,
            drop_none=True,
            buffer_size=INFINITE_BUFFER_SIZE,
        )

        split_and_classification_dp = Filter(
            split_and_classification_dp, path_comparator("name", f"{config.split}.txt")
        )
        split_and_classification_dp = CSVDictParser(
            split_and_classification_dp, fieldnames=("image_id", "label", "species"), delimiter=" "
        )
        split_and_classification_dp = hint_sharding(split_and_classification_dp)
        split_and_classification_dp = hint_shuffling(split_and_classification_dp)

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
        return self._classify_anns(data) == OxfordIITPetDemux.SPLIT_AND_CLASSIFICATION

    def _generate_categories(self, root: pathlib.Path) -> List[str]:
        config = self.default_config
        resources = self.resources(config)

        dp = resources[1].load(root)
        dp = Filter(dp, self._filter_split_and_classification_anns)
        dp = Filter(dp, path_comparator("name", f"{config.split}.txt"))
        dp = CSVDictParser(dp, fieldnames=("image_id", "label"), delimiter=" ")

        raw_categories_and_labels = {(data["image_id"].rsplit("_", 1)[0], data["label"]) for data in dp}
        raw_categories, _ = zip(
            *sorted(raw_categories_and_labels, key=lambda raw_category_and_label: int(raw_category_and_label[1]))
        )
        return [" ".join(part.title() for part in raw_category.split("_")) for raw_category in raw_categories]
