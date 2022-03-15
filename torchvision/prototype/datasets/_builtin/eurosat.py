import pathlib
from typing import Any, Dict, List, Tuple

from torchdata.datapipes.iter import IterDataPipe, Mapper
from torchvision.prototype.datasets.utils import Dataset, DatasetConfig, DatasetInfo, HttpResource, OnlineResource
from torchvision.prototype.datasets.utils._internal import hint_sharding, hint_shuffling
from torchvision.prototype.features import EncodedImage, Label


class EuroSAT(Dataset):
    def _make_info(self) -> DatasetInfo:
        return DatasetInfo(
            "eurosat",
            homepage="https://github.com/phelber/eurosat",
            categories=(
                "AnnualCrop",
                "Forest",
                "HerbaceousVegetation",
                "Highway",
                "Industrial," "Pasture",
                "PermanentCrop",
                "Residential",
                "River",
                "SeaLake",
            ),
        )

    def resources(self, config: DatasetConfig) -> List[OnlineResource]:
        return [
            HttpResource(
                "https://madm.dfki.de/files/sentinel/EuroSAT.zip",
                sha256="8ebea626349354c5328b142b96d0430e647051f26efc2dc974c843f25ecf70bd",
            )
        ]

    def _prepare_sample(self, data: Tuple[str, Any]) -> Dict[str, Any]:
        path, buffer = data
        category = pathlib.Path(path).parent.name
        return dict(
            label=Label.from_category(category, categories=self.categories),
            path=path,
            image=EncodedImage.from_file(buffer),
        )

    def _make_datapipe(
        self, resource_dps: List[IterDataPipe], *, config: DatasetConfig
    ) -> IterDataPipe[Dict[str, Any]]:
        dp = resource_dps[0]
        dp = hint_sharding(dp)
        dp = hint_shuffling(dp)
        return Mapper(dp, self._prepare_sample)
