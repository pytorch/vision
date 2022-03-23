import pathlib
from typing import Any, Dict, List, Tuple

from torchdata.datapipes.iter import IterDataPipe, Mapper, Filter, LineReader, IterKeyZipper
from torchvision.prototype.datasets.utils import Dataset, DatasetConfig, DatasetInfo, HttpResource, OnlineResource
from torchvision.prototype.datasets.utils._internal import (
    path_comparator,
    hint_sharding,
    hint_shuffling,
    INFINITE_BUFFER_SIZE,
)
from torchvision.prototype.features import EncodedImage, Label


class SUN397(Dataset):

    _DATA_ROOT_DIR = "SUN397"
    _SPLIT_NAME_MAPPER = {
        "train": "Training",
        "test": "Testing",
    }

    def _make_info(self) -> DatasetInfo:
        return DatasetInfo(
            "sun397",
            homepage="https://vision.princeton.edu/projects/2010/SUN/",
            valid_options=dict(
                split=("train", "test"),
                fold=tuple(range(1, 11)),
            ),
        )

    def resources(self, config: DatasetConfig) -> List[OnlineResource]:
        images = HttpResource(
            "http://vision.princeton.edu/projects/2010/SUN/SUN397.tar.gz",
            sha256="f404130965a7ad77bed5ececc71e720ab50f3cc1e1bb257257610d38f3b928ec",
        )
        splits = HttpResource(
            "https://vision.princeton.edu/projects/2010/SUN/download/Partitions.zip",
            sha256="e3311ea5f812d4cc6faef194688343d1111f6cd6f70f2c62f393c2b8d6ba1ec9",
        )
        return [images, splits]

    def _get_category_from_path(self, path_str: str) -> str:
        root_index = 0
        for i, fname in enumerate(pathlib.Path(path_str).parts):
            if fname == self._DATA_ROOT_DIR:
                root_index = i
        category = "/".join(pathlib.Path(path_str).parts[root_index + 2 : -1])
        return category

    def _prepare_sample(self, data: Tuple[str, Any]) -> Dict[str, Any]:
        path_str, buffer = data
        category = self._get_category_from_path(path_str)
        return dict(
            label=Label.from_category(category, categories=self.categories),
            path=path_str,
            image=EncodedImage.from_file(buffer),
        )

    def _image_key_fn(self, data: Tuple[str, Any]) -> str:
        path_str, _ = data
        category = self._get_category_from_path(path_str)
        filename = pathlib.Path(path_str).name
        return "/".join([category, filename])

    def _split_key_fn(self, partial_path: str) -> str:
        return partial_path[3:]

    def _merge_fn(self, split_data: str, img_data: Tuple[str, Any]) -> Tuple[str, Any]:
        return img_data

    def _make_datapipe(
        self, resource_dps: List[IterDataPipe], *, config: DatasetConfig
    ) -> IterDataPipe[Dict[str, Any]]:
        images_dp, splits_dp = resource_dps

        splits_dp = Filter(
            splits_dp,
            path_comparator(
                "name",
                "{split_name}_{fold:02d}.txt".format(
                    split_name=self._SPLIT_NAME_MAPPER[config.split], fold=config.fold
                ),
            ),
        )
        splits_dp = LineReader(splits_dp, decode=True, return_path=False)
        splits_dp = hint_sharding(splits_dp)
        splits_dp = hint_shuffling(splits_dp)

        images_dp = Filter(images_dp, path_comparator("suffix", ".jpg"))

        # Filter images that appeared in splits_dp
        images_dp = IterKeyZipper(
            splits_dp,
            images_dp,
            key_fn=self._split_key_fn,
            ref_key_fn=self._image_key_fn,
            merge_fn=self._merge_fn,
            buffer_size=INFINITE_BUFFER_SIZE,
        )
        return Mapper(images_dp, self._prepare_sample)

    def _generate_categories(self, root: pathlib.Path) -> List[str]:
        resources = self.resources(self.default_config)
        dp = resources[0].load(root)
        return sorted({self._get_category_from_path(path_str) for path_str, _ in dp})
