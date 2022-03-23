import pathlib
import re
from typing import Any, Dict, List, Tuple

from torchdata.datapipes.iter import IterDataPipe, Mapper, Filter
from torchvision.prototype.datasets.utils import Dataset, DatasetConfig, DatasetInfo, HttpResource, OnlineResource
from torchvision.prototype.datasets.utils._internal import path_comparator, hint_sharding, hint_shuffling
from torchvision.prototype.features import EncodedImage, Label


class SUN397(Dataset):

    _IMAGE_FILENAME_PATTERN = re.compile(r"sun_.*\.jpg")
    _DATA_ROOT_DIR = "SUN397"

    def _make_info(self) -> DatasetInfo:
        return DatasetInfo(
            "sun397",
            homepage="https://vision.princeton.edu/projects/2010/SUN/",
        )

    def resources(self, config: DatasetConfig) -> List[OnlineResource]:
        return [
            HttpResource(
                "http://vision.princeton.edu/projects/2010/SUN/SUN397.tar.gz",
                sha256="f404130965a7ad77bed5ececc71e720ab50f3cc1e1bb257257610d38f3b928ec",
            )
        ]

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

    def _filter_image_files(self, data: Tuple[str, Any]) -> bool:
        match = self._IMAGE_FILENAME_PATTERN.match(pathlib.Path(data[0]).name)
        return bool(match)

    def _make_datapipe(
        self, resource_dps: List[IterDataPipe], *, config: DatasetConfig
    ) -> IterDataPipe[Dict[str, Any]]:
        dp = resource_dps[0]
        dp = Filter(dp, self._filter_image_files)
        dp = hint_sharding(dp)
        dp = hint_shuffling(dp)
        return Mapper(dp, self._prepare_sample)

    def _generate_categories(self, root: pathlib.Path) -> List[str]:
        resources = self.resources(self.default_config)
        dp = resources[0].load(root)
        return sorted({self._get_category_from_path(path) for path, _ in dp})
