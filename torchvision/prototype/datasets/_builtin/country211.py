import pathlib
from typing import Any, Dict, List, Tuple

from torchdata.datapipes.iter import IterDataPipe, Mapper, Filter
from torchvision.prototype.datasets.utils import Dataset, DatasetConfig, DatasetInfo, HttpResource, OnlineResource
from torchvision.prototype.datasets.utils._internal import path_comparator, hint_sharding, hint_shuffling
from torchvision.prototype.features import EncodedImage, Label


class Country211(Dataset):
    def _make_info(self) -> DatasetInfo:
        return DatasetInfo(
            "country211",
            homepage="https://github.com/openai/CLIP/blob/main/data/country211.md",
            valid_options=dict(split=("train", "val", "test")),
        )

    def resources(self, config: DatasetConfig) -> List[OnlineResource]:
        return [
            HttpResource(
                "https://openaipublic.azureedge.net/clip/data/country211.tgz",
                sha256="c011343cdc1296a8c31ff1d7129cf0b5e5b8605462cffd24f89266d6e6f4da3c",
            )
        ]

    _SPLIT_NAME_MAPPER = {
        "train": "train",
        "val": "valid",
        "test": "test",
    }

    def _prepare_sample(self, data: Tuple[str, Any]) -> Dict[str, Any]:
        path, buffer = data
        category = pathlib.Path(path).parent.name
        return dict(
            label=Label.from_category(category, categories=self.categories),
            path=path,
            image=EncodedImage.from_file(buffer),
        )

    def _filter_split(self, data: Tuple[str, Any], *, split: str) -> bool:
        return pathlib.Path(data[0]).parent.parent.name == split

    def _make_datapipe(
        self, resource_dps: List[IterDataPipe], *, config: DatasetConfig
    ) -> IterDataPipe[Dict[str, Any]]:
        dp = resource_dps[0]
        dp = Filter(dp, path_comparator("parent.parent.name", self._SPLIT_NAME_MAPPER[config.split]))
        dp = hint_sharding(dp)
        dp = hint_shuffling(dp)
        return Mapper(dp, self._prepare_sample)

    def _generate_categories(self, root: pathlib.Path) -> List[str]:
        resources = self.resources(self.default_config)
        dp = resources[0].load(root)
        return sorted({pathlib.Path(path).parent.name for path, _ in dp})
