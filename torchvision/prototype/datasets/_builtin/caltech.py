import io
import pathlib
import re
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import numpy as np
import torch
from torch.utils.data import IterDataPipe
from torch.utils.data.datapipes.iter import (
    Mapper,
    TarArchiveReader,
    Shuffler,
    Filter,
)
from torchdata.datapipes.iter import KeyZipper
from torchvision.prototype.datasets.utils import (
    Dataset,
    DatasetConfig,
    DatasetInfo,
    HttpResource,
    OnlineResource,
)
from torchvision.prototype.datasets.utils._internal import create_categories_file, INFINITE_BUFFER_SIZE, read_mat

HERE = pathlib.Path(__file__).parent


class Caltech101(Dataset):
    @property
    def info(self) -> DatasetInfo:
        return DatasetInfo(
            "caltech101",
            type="image",
            categories=HERE / "caltech101.categories",
            homepage="http://www.vision.caltech.edu/Image_Datasets/Caltech101",
        )

    def resources(self, config: DatasetConfig) -> List[OnlineResource]:
        images = HttpResource(
            "http://www.vision.caltech.edu/Image_Datasets/Caltech101/101_ObjectCategories.tar.gz",
            sha256="af6ece2f339791ca20f855943d8b55dd60892c0a25105fcd631ee3d6430f9926",
        )
        anns = HttpResource(
            "http://www.vision.caltech.edu/Image_Datasets/Caltech101/Annotations.tar",
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

    def _collate_and_decode_sample(
        self, data, *, decoder: Optional[Callable[[io.IOBase], Dict[str, Any]]]
    ) -> Dict[str, Any]:
        key, image_data, ann_data = data
        category, _ = key
        image_path, image_buffer = image_data
        ann_path, ann_buffer = ann_data

        label = self.info.categories.index(category)

        ann = read_mat(ann_buffer)
        bbox = torch.as_tensor(ann["box_coord"].astype(np.int64))
        contour = torch.as_tensor(ann["obj_contour"])

        sample = dict(
            category=category,
            label=label,
            image_path=image_path,
            bbox=bbox,
            contour=contour,
            ann_path=ann_path,
        )
        sample.update(decoder(image_buffer) if decoder else dict(image=image_buffer))
        return sample

    def _make_datapipe(
        self,
        resource_dps: List[IterDataPipe],
        *,
        config: DatasetConfig,
        decoder: Optional[Callable[[io.IOBase], Dict[str, Any]]],
    ) -> IterDataPipe[Dict[str, Any]]:
        images_dp, anns_dp = resource_dps

        images_dp = TarArchiveReader(images_dp)
        images_dp = Filter(images_dp, self._is_not_background_image)
        images_dp = Shuffler(images_dp, buffer_size=INFINITE_BUFFER_SIZE)

        anns_dp = TarArchiveReader(anns_dp)
        anns_dp = Filter(anns_dp, self._is_ann)

        dp = KeyZipper(
            images_dp,
            anns_dp,
            key_fn=self._images_key_fn,
            ref_key_fn=self._anns_key_fn,
            buffer_size=INFINITE_BUFFER_SIZE,
            keep_key=True,
        )
        return Mapper(dp, self._collate_and_decode_sample, fn_kwargs=dict(decoder=decoder))

    def generate_categories_file(self, root: Union[str, pathlib.Path]) -> None:
        dp = self.resources(self.default_config)[0].to_datapipe(pathlib.Path(root) / self.name)
        dp = TarArchiveReader(dp)
        dp = Filter(dp, self._is_not_background_image)
        dir_names = {pathlib.Path(path).parent.name for path, _ in dp}
        create_categories_file(HERE, self.name, sorted(dir_names))


class Caltech256(Dataset):
    @property
    def info(self) -> DatasetInfo:
        return DatasetInfo(
            "caltech256",
            type="image",
            categories=HERE / "caltech256.categories",
            homepage="http://www.vision.caltech.edu/Image_Datasets/Caltech256",
        )

    def resources(self, config: DatasetConfig) -> List[OnlineResource]:
        return [
            HttpResource(
                "http://www.vision.caltech.edu/Image_Datasets/Caltech256/256_ObjectCategories.tar",
                sha256="08ff01b03c65566014ae88eb0490dbe4419fc7ac4de726ee1163e39fd809543e",
            )
        ]

    def _is_not_rogue_file(self, data: Tuple[str, Any]) -> bool:
        path = pathlib.Path(data[0])
        return path.name != "RENAME2"

    def _collate_and_decode_sample(
        self,
        data: Tuple[str, io.IOBase],
        *,
        decoder: Optional[Callable[[io.IOBase], Dict[str, Any]]],
    ) -> Dict[str, Any]:
        path, buffer = data

        dir_name = pathlib.Path(path).parent.name
        label_str, category = dir_name.split(".")
        label = torch.tensor(int(label_str))

        sample = dict(label=label, category=category)
        sample.update(decoder(buffer) if decoder else dict(image=buffer))
        return sample

    def _make_datapipe(
        self,
        resource_dps: List[IterDataPipe],
        *,
        config: DatasetConfig,
        decoder: Optional[Callable[[io.IOBase], Dict[str, Any]]],
    ) -> IterDataPipe[Dict[str, Any]]:
        dp = resource_dps[0]
        dp = TarArchiveReader(dp)
        dp = Filter(dp, self._is_not_rogue_file)
        dp = Shuffler(dp, buffer_size=INFINITE_BUFFER_SIZE)
        return Mapper(dp, self._collate_and_decode_sample, fn_kwargs=dict(decoder=decoder))

    def generate_categories_file(self, root: Union[str, pathlib.Path]) -> None:
        dp = self.resources(self.default_config)[0].to_datapipe(pathlib.Path(root) / self.name)
        dp = TarArchiveReader(dp)
        dir_names = {pathlib.Path(path).parent.name for path, _ in dp}
        categories = [name.split(".")[1] for name in sorted(dir_names)]
        create_categories_file(HERE, self.name, categories)


if __name__ == "__main__":
    from torchvision.prototype.datasets import home

    root = home()
    Caltech101().generate_categories_file(root)
    Caltech256().generate_categories_file(root)
