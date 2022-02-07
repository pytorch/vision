import io
import pathlib
from typing import Any, Callable, Dict, List, Optional

import torch
from torchdata.datapipes.iter import IterDataPipe, Filter, LineReader
from torchvision.prototype.datasets.utils import (
    Dataset,
    DatasetConfig,
    DatasetInfo,
    HttpResource,
    OnlineResource,
    DatasetType,
)
from torchvision.prototype.datasets.utils._internal import hint_sharding, hint_shuffling, path_comparator


class Places365(Dataset):
    def _make_info(self) -> DatasetInfo:
        return DatasetInfo(
            name="places365",
            type=DatasetType.IMAGE,
            homepage="http://places2.csail.mit.edu/index.html",
            valid_options=dict(split=("train", "val", "test")),
        )

    _RESOURCES = {
        "train": ("train_large_places365standard.tar", ""),
        "val": ("val_large.tar", "ddd71c418592a4c230645e238f9e52de77247461d68cd9a14a080a9ca6f27df6"),
        "test": ("test_large.tar", "4fae1d859035fe800a7697c27e5e69d78eb292d4cf12d84798c497b23b46b8e1"),
    }

    def resources(self, config: DatasetConfig) -> List[OnlineResource]:
        url_root = "http://data.csail.mit.edu/places/places365"

        meta = HttpResource(
            f"{url_root}/filelist_places365-standard.tar",
            sha256="520699e00d69b63ddc29fac54645aa00ce1c10ca42e288960aa1cf791d6e9aa9",
        )

        filename, sha256 = self._RESOURCES[config.split]
        images = HttpResource(f"{url_root}/{filename}", sha256=sha256)

        return [meta, images]

    def _make_datapipe(
        self,
        resource_dps: List[IterDataPipe],
        *,
        config: DatasetConfig,
        decoder: Optional[Callable[[io.IOBase], torch.Tensor]],
    ) -> IterDataPipe[Dict[str, Any]]:
        meta_dp, images_dp = resource_dps

        images_dp = hint_sharding(images_dp)
        images_dp = hint_shuffling(images_dp)

        return images_dp

    def _generate_categories(self, root: pathlib.Path) -> List[str]:
        resources = self.resources(self.info.make_config(split="val"))

        meta_dp = resources[0].load(root)
        categories_dp = Filter(meta_dp, path_comparator("name", "categories_places365.txt"))
        categories_dp = LineReader(categories_dp, decode=True, return_path=False)

        return [posix_path_and_label.split()[0][3:] for posix_path_and_label in categories_dp]
