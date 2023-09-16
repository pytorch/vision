import pathlib
import re
from typing import Any, BinaryIO, Dict, List, Tuple, Union

import numpy as np

import torch
from torchdata.datapipes.iter import Filter, IterDataPipe, IterKeyZipper, Mapper
from torchvision.prototype.datasets.utils import Dataset, EncodedImage, GDriveResource, OnlineResource
from torchvision.prototype.datasets.utils._internal import (
    hint_sharding,
    hint_shuffling,
    INFINITE_BUFFER_SIZE,
    read_categories_file,
    read_mat,
)
from torchvision.prototype.tv_tensors import Label
from torchvision.tv_tensors import BoundingBoxes

from .._api import register_dataset, register_info


@register_info("caltech101")
def _caltech101_info() -> Dict[str, Any]:
    return dict(categories=read_categories_file("caltech101"))


@register_dataset("caltech101")
class Caltech101(Dataset):
    """
    - **homepage**: https://data.caltech.edu/records/20086
    - **dependencies**:
        - <scipy `https://scipy.org/`>_
    """

    def __init__(
        self,
        root: Union[str, pathlib.Path],
        skip_integrity_check: bool = False,
    ) -> None:
        self._categories = _caltech101_info()["categories"]

        super().__init__(
            root,
            dependencies=("scipy",),
            skip_integrity_check=skip_integrity_check,
        )

    def _resources(self) -> List[OnlineResource]:
        images = GDriveResource(
            "137RyRjvTBkBiIfeYBNZBtViDHQ6_Ewsp",
            file_name="101_ObjectCategories.tar.gz",
            sha256="af6ece2f339791ca20f855943d8b55dd60892c0a25105fcd631ee3d6430f9926",
            preprocess="decompress",
        )
        anns = GDriveResource(
            "175kQy3UsZ0wUEHZjqkUDdNVssr7bgh_m",
            file_name="Annotations.tar",
            sha256="1717f4e10aa837b05956e3f4c94456527b143eec0d95e935028b30aff40663d8",
        )
        return [images, anns]

    _IMAGES_NAME_PATTERN = re.compile(r"image_(?P<id>\d+)[.]jpg")
    _ANNS_NAME_PATTERN = re.compile(r"annotation_(?P<id>\d+)[.]mat")
    _ANNS_CATEGORY_MAP = {
        "Faces_2": "Faces",
        "Faces_3": "Faces_easy",
        "Motorbikes_16": "Motorbikes",
        "Airplanes_Side_2": "airplanes",
    }

    def _is_not_background_image(self, data: Tuple[str, Any]) -> bool:
        path = pathlib.Path(data[0])
        return path.parent.name != "BACKGROUND_Google"

    def _is_ann(self, data: Tuple[str, Any]) -> bool:
        path = pathlib.Path(data[0])
        return bool(self._ANNS_NAME_PATTERN.match(path.name))

    def _images_key_fn(self, data: Tuple[str, Any]) -> Tuple[str, str]:
        path = pathlib.Path(data[0])

        category = path.parent.name
        id = self._IMAGES_NAME_PATTERN.match(path.name).group("id")  # type: ignore[union-attr]

        return category, id

    def _anns_key_fn(self, data: Tuple[str, Any]) -> Tuple[str, str]:
        path = pathlib.Path(data[0])

        category = path.parent.name
        if category in self._ANNS_CATEGORY_MAP:
            category = self._ANNS_CATEGORY_MAP[category]

        id = self._ANNS_NAME_PATTERN.match(path.name).group("id")  # type: ignore[union-attr]

        return category, id

    def _prepare_sample(
        self, data: Tuple[Tuple[str, str], Tuple[Tuple[str, BinaryIO], Tuple[str, BinaryIO]]]
    ) -> Dict[str, Any]:
        key, (image_data, ann_data) = data
        category, _ = key
        image_path, image_buffer = image_data
        ann_path, ann_buffer = ann_data

        image = EncodedImage.from_file(image_buffer)
        ann = read_mat(ann_buffer)

        return dict(
            label=Label.from_category(category, categories=self._categories),
            image_path=image_path,
            image=image,
            ann_path=ann_path,
            bounding_boxes=BoundingBoxes(
                ann["box_coord"].astype(np.int64).squeeze()[[2, 0, 3, 1]],
                format="xyxy",
                spatial_size=image.spatial_size,
            ),
            contour=torch.as_tensor(ann["obj_contour"].T),
        )

    def _datapipe(self, resource_dps: List[IterDataPipe]) -> IterDataPipe[Dict[str, Any]]:
        images_dp, anns_dp = resource_dps

        images_dp = Filter(images_dp, self._is_not_background_image)
        images_dp = hint_shuffling(images_dp)
        images_dp = hint_sharding(images_dp)

        anns_dp = Filter(anns_dp, self._is_ann)

        dp = IterKeyZipper(
            images_dp,
            anns_dp,
            key_fn=self._images_key_fn,
            ref_key_fn=self._anns_key_fn,
            buffer_size=INFINITE_BUFFER_SIZE,
            keep_key=True,
        )
        return Mapper(dp, self._prepare_sample)

    def __len__(self) -> int:
        return 8677

    def _generate_categories(self) -> List[str]:
        resources = self._resources()

        dp = resources[0].load(self._root)
        dp = Filter(dp, self._is_not_background_image)

        return sorted({pathlib.Path(path).parent.name for path, _ in dp})


@register_info("caltech256")
def _caltech256_info() -> Dict[str, Any]:
    return dict(categories=read_categories_file("caltech256"))


@register_dataset("caltech256")
class Caltech256(Dataset):
    """
    - **homepage**: https://data.caltech.edu/records/20087
    """

    def __init__(
        self,
        root: Union[str, pathlib.Path],
        skip_integrity_check: bool = False,
    ) -> None:
        self._categories = _caltech256_info()["categories"]

        super().__init__(root, skip_integrity_check=skip_integrity_check)

    def _resources(self) -> List[OnlineResource]:
        return [
            GDriveResource(
                "1r6o0pSROcV1_VwT4oSjA2FBUSCWGuxLK",
                file_name="256_ObjectCategories.tar",
                sha256="08ff01b03c65566014ae88eb0490dbe4419fc7ac4de726ee1163e39fd809543e",
            )
        ]

    def _is_not_rogue_file(self, data: Tuple[str, Any]) -> bool:
        path = pathlib.Path(data[0])
        return path.name != "RENAME2"

    def _prepare_sample(self, data: Tuple[str, BinaryIO]) -> Dict[str, Any]:
        path, buffer = data

        return dict(
            path=path,
            image=EncodedImage.from_file(buffer),
            label=Label(int(pathlib.Path(path).parent.name.split(".", 1)[0]) - 1, categories=self._categories),
        )

    def _datapipe(self, resource_dps: List[IterDataPipe]) -> IterDataPipe[Dict[str, Any]]:
        dp = resource_dps[0]
        dp = Filter(dp, self._is_not_rogue_file)
        dp = hint_shuffling(dp)
        dp = hint_sharding(dp)
        return Mapper(dp, self._prepare_sample)

    def __len__(self) -> int:
        return 30607

    def _generate_categories(self) -> List[str]:
        resources = self._resources()

        dp = resources[0].load(self._root)
        dir_names = {pathlib.Path(path).parent.name for path, _ in dp}

        return [name.split(".")[1] for name in sorted(dir_names)]
