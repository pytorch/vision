import json
from collections import namedtuple
from functools import partial
from pathlib import Path
from typing import cast, Any, Dict, List, Tuple

from torchdata.datapipes.iter import Grouper, IterDataPipe, Mapper, Filter, IterKeyZipper
from torchvision.prototype.datasets.utils import (
    Dataset,
    DatasetInfo,
    DatasetConfig,
    ManualDownloadResource,
    OnlineResource,
)
from torchvision.prototype.datasets.utils._internal import INFINITE_BUFFER_SIZE, hint_sharding, hint_shuffling
from torchvision.prototype.features import EncodedImage
from torchvision.prototype.utils._internal import FrozenMapping


class CityscapesDatasetInfo(DatasetInfo):
    def __init__(self, *args: Any, **kwargs: Any):
        super().__init__(*args, **kwargs)
        self._configs = tuple(
            config
            for config in self._configs
            if not (
                (config.split == "test" and config.mode == "coarse")
                or (config.split == "train_extra" and config.mode == "fine")
            )
        )

    def make_config(self, **options: Any) -> DatasetConfig:
        config = super().make_config(**options)
        if config.split == "test" and config.mode == "coarse":
            raise ValueError("`split='test'` is only available for `mode='fine'`")
        if config.split == "train_extra" and config.mode == "fine":
            raise ValueError("`split='train_extra'` is only available for `mode='coarse'`")

        return config


class CityscapesResource(ManualDownloadResource):
    def __init__(self, **kwargs: Any) -> None:
        super().__init__(
            "Register on https://www.cityscapes-dataset.com/login/ and follow the instructions there.", **kwargs
        )


CityscapesClass = namedtuple(
    "CityscapesClass",
    ["name", "id", "train_id", "category", "category_id", "has_instances", "ignore_in_eval", "color"],
)


class Cityscapes(Dataset):

    categories_to_details: FrozenMapping = FrozenMapping(
        {
            "unlabeled": CityscapesClass("unlabeled", 0, 255, "void", 0, False, True, (0, 0, 0)),
            "ego vehicle": CityscapesClass("ego vehicle", 1, 255, "void", 0, False, True, (0, 0, 0)),
            "rectification border": CityscapesClass("rectification border", 2, 255, "void", 0, False, True, (0, 0, 0)),
            "out of roi": CityscapesClass("out of roi", 3, 255, "void", 0, False, True, (0, 0, 0)),
            "static": CityscapesClass("static", 4, 255, "void", 0, False, True, (0, 0, 0)),
            "dynamic": CityscapesClass("dynamic", 5, 255, "void", 0, False, True, (111, 74, 0)),
            "ground": CityscapesClass("ground", 6, 255, "void", 0, False, True, (81, 0, 81)),
            "road": CityscapesClass("road", 7, 0, "flat", 1, False, False, (128, 64, 128)),
            "sidewalk": CityscapesClass("sidewalk", 8, 1, "flat", 1, False, False, (244, 35, 232)),
            "parking": CityscapesClass("parking", 9, 255, "flat", 1, False, True, (250, 170, 160)),
            "rail track": CityscapesClass("rail track", 10, 255, "flat", 1, False, True, (230, 150, 140)),
            "building": CityscapesClass("building", 11, 2, "construction", 2, False, False, (70, 70, 70)),
            "wall": CityscapesClass("wall", 12, 3, "construction", 2, False, False, (102, 102, 156)),
            "fence": CityscapesClass("fence", 13, 4, "construction", 2, False, False, (190, 153, 153)),
            "guard rail": CityscapesClass("guard rail", 14, 255, "construction", 2, False, True, (180, 165, 180)),
            "bridge": CityscapesClass("bridge", 15, 255, "construction", 2, False, True, (150, 100, 100)),
            "tunnel": CityscapesClass("tunnel", 16, 255, "construction", 2, False, True, (150, 120, 90)),
            "pole": CityscapesClass("pole", 17, 5, "object", 3, False, False, (153, 153, 153)),
            "polegroup": CityscapesClass("polegroup", 18, 255, "object", 3, False, True, (153, 153, 153)),
            "traffic light": CityscapesClass("traffic light", 19, 6, "object", 3, False, False, (250, 170, 30)),
            "traffic sign": CityscapesClass("traffic sign", 20, 7, "object", 3, False, False, (220, 220, 0)),
            "vegetation": CityscapesClass("vegetation", 21, 8, "nature", 4, False, False, (107, 142, 35)),
            "terrain": CityscapesClass("terrain", 22, 9, "nature", 4, False, False, (152, 251, 152)),
            "sky": CityscapesClass("sky", 23, 10, "sky", 5, False, False, (70, 130, 180)),
            "person": CityscapesClass("person", 24, 11, "human", 6, True, False, (220, 20, 60)),
            "rider": CityscapesClass("rider", 25, 12, "human", 6, True, False, (255, 0, 0)),
            "car": CityscapesClass("car", 26, 13, "vehicle", 7, True, False, (0, 0, 142)),
            "truck": CityscapesClass("truck", 27, 14, "vehicle", 7, True, False, (0, 0, 70)),
            "bus": CityscapesClass("bus", 28, 15, "vehicle", 7, True, False, (0, 60, 100)),
            "caravan": CityscapesClass("caravan", 29, 255, "vehicle", 7, True, True, (0, 0, 90)),
            "trailer": CityscapesClass("trailer", 30, 255, "vehicle", 7, True, True, (0, 0, 110)),
            "train": CityscapesClass("train", 31, 16, "vehicle", 7, True, False, (0, 80, 100)),
            "motorcycle": CityscapesClass("motorcycle", 32, 17, "vehicle", 7, True, False, (0, 0, 230)),
            "bicycle": CityscapesClass("bicycle", 33, 18, "vehicle", 7, True, False, (119, 11, 32)),
            "license plate": CityscapesClass("license plate", -1, -1, "vehicle", 7, False, True, (0, 0, 142)),
        }
    )

    def _make_info(self) -> DatasetInfo:
        name = "cityscapes"

        return CityscapesDatasetInfo(
            name,
            categories=list(self.categories_to_details.keys()),
            homepage="http://www.cityscapes-dataset.com/",
            valid_options=dict(
                split=("train", "val", "test", "train_extra"),
                mode=("fine", "coarse"),
            ),
            extra=dict(categories_to_details=self.categories_to_details),
        )

    _FILES_CHECKSUMS = {
        "gtCoarse.zip": "3555e09349ed49127053d940eaa66a87a79a175662b329c1a26a58d47e602b5b",
        "gtFine_trainvaltest.zip": "40461a50097844f400fef147ecaf58b18fd99e14e4917fb7c3bf9c0d87d95884",
        "leftImg8bit_trainextra.zip": "e41cc14c0c06aad051d52042465d9b8c22bacf6e4c93bb98de273ed7177b7133",
        "leftImg8bit_trainvaltest.zip": "3ccff9ac1fa1d80a6a064407e589d747ed0657aac7dc495a4403ae1235a37525",
    }

    def resources(self, config: DatasetConfig) -> List[OnlineResource]:

        if config.mode == "fine":
            filenames = ("leftImg8bit_trainvaltest.zip", "gtFine_trainvaltest.zip")
        else:
            split_label = "trainextra" if config.split == "train_extra" else "trainvaltest"
            filenames = (f"leftImg8bit_{split_label}.zip", "gtCoarse.zip")

        return cast(
            List[OnlineResource],
            [
                CityscapesResource(file_name=file_name, sha256=self._FILES_CHECKSUMS[file_name])
                for file_name in filenames
            ],
        )

    def _filter_split_images(self, data: Tuple[str, Any], *, req_split: str) -> bool:
        path = Path(data[0])
        split = path.parent.parts[-2]
        return split == req_split and ".png" == path.suffix

    def _get_key_from_path(self, data: Tuple[Any, Any]) -> str:
        data0 = data[0]
        if isinstance(data0, tuple):
            data0 = data0[0]
        path = Path(data0)
        # The pathes for images and targets are
        # - leftImg8bit/val/frankfurt/frankfurt_000001_066574_leftImg8bit.png
        # - gtFine/val/frankfurt/frankfurt_000001_066574_gtFine_polygons.json
        # - gtFine/val/frankfurt/frankfurt_000001_066574_gtFine_labelIds.png
        # we transform them into "frankfurt_000001_066574"
        return "_".join(path.name.split("_", 3)[:3])

    def _prepare_sample(self, data: Tuple[Tuple[str, Any], Any]) -> Dict[str, Any]:
        (img_path, img_data), target_data = data

        output = dict(image_path=img_path, image=EncodedImage.from_file(img_data))
        # reorder inside group of targets and setup output dictionary:
        for path, data in target_data:
            stem = Path(path).stem
            for target_type in ["instance", "label", "polygon", "color"]:
                if target_type in stem:
                    if target_type == "polygon":
                        enc_data = json.loads(data.read())
                    elif target_type == "label":
                        # TODO: We need an EncodedSegmentationMask feature now that we also
                        # have a separate SegementationMask.
                        enc_data = EncodedImage.from_file(data)
                    else:
                        enc_data = EncodedImage.from_file(data)
                    output[target_type] = enc_data
                    output[f"{target_type}_path"] = path
                    break

        return output

    def _make_datapipe(
        self,
        resource_dps: List[IterDataPipe],
        *,
        config: DatasetConfig,
    ) -> IterDataPipe[Dict[str, Any]]:
        archive_images, archive_targets = resource_dps

        images_dp = Filter(archive_images, filter_fn=partial(self._filter_split_images, req_split=config.split))

        images_dp = hint_sharding(images_dp)
        images_dp = hint_shuffling(images_dp)

        # As city names are unique per split we can group targets by
        # keys like "frankfurt_000001_066574"
        targets_dp = Grouper(
            archive_targets,
            group_key_fn=self._get_key_from_path,
            buffer_size=INFINITE_BUFFER_SIZE,
            group_size=4,
        )

        samples = IterKeyZipper(
            images_dp,
            targets_dp,
            key_fn=self._get_key_from_path,
            buffer_size=INFINITE_BUFFER_SIZE,
        )
        return Mapper(samples, fn=self._prepare_sample)
