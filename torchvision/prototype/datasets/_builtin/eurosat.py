import pathlib
from typing import Any, Dict, List, Tuple, Union

from torchdata.datapipes.iter import IterDataPipe, Mapper
from torchvision.prototype.datasets.utils import Dataset, EncodedImage, HttpResource, OnlineResource
from torchvision.prototype.datasets.utils._internal import hint_sharding, hint_shuffling
from torchvision.prototype.tv_tensors import Label

from .._api import register_dataset, register_info

NAME = "eurosat"


@register_info(NAME)
def _info() -> Dict[str, Any]:
    return dict(
        categories=(
            "AnnualCrop",
            "Forest",
            "HerbaceousVegetation",
            "Highway",
            "Industrial",
            "Pasture",
            "PermanentCrop",
            "Residential",
            "River",
            "SeaLake",
        )
    )


@register_dataset(NAME)
class EuroSAT(Dataset):
    """EuroSAT Dataset.
    homepage="https://github.com/phelber/eurosat",
    """

    def __init__(self, root: Union[str, pathlib.Path], *, skip_integrity_check: bool = False) -> None:
        self._categories = _info()["categories"]
        super().__init__(root, skip_integrity_check=skip_integrity_check)

    def _resources(self) -> List[OnlineResource]:
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
            label=Label.from_category(category, categories=self._categories),
            path=path,
            image=EncodedImage.from_file(buffer),
        )

    def _datapipe(self, resource_dps: List[IterDataPipe]) -> IterDataPipe[Dict[str, Any]]:
        dp = resource_dps[0]
        dp = hint_shuffling(dp)
        dp = hint_sharding(dp)
        return Mapper(dp, self._prepare_sample)

    def __len__(self) -> int:
        return 27_000
