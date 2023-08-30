import pathlib
from typing import Any, Dict, List, Tuple, Union

from torchdata.datapipes.iter import Filter, IterDataPipe, Mapper
from torchvision.prototype.datasets.utils import Dataset, EncodedImage, HttpResource, OnlineResource
from torchvision.prototype.datasets.utils._internal import (
    hint_sharding,
    hint_shuffling,
    path_comparator,
    read_categories_file,
)
from torchvision.prototype.tv_tensors import Label

from .._api import register_dataset, register_info

NAME = "country211"


@register_info(NAME)
def _info() -> Dict[str, Any]:
    return dict(categories=read_categories_file(NAME))


@register_dataset(NAME)
class Country211(Dataset):
    """
    - **homepage**: https://github.com/openai/CLIP/blob/main/data/country211.md
    """

    def __init__(
        self,
        root: Union[str, pathlib.Path],
        *,
        split: str = "train",
        skip_integrity_check: bool = False,
    ) -> None:
        self._split = self._verify_str_arg(split, "split", ("train", "val", "test"))
        self._split_folder_name = "valid" if split == "val" else split

        self._categories = _info()["categories"]

        super().__init__(root, skip_integrity_check=skip_integrity_check)

    def _resources(self) -> List[OnlineResource]:
        return [
            HttpResource(
                "https://openaipublic.azureedge.net/clip/data/country211.tgz",
                sha256="c011343cdc1296a8c31ff1d7129cf0b5e5b8605462cffd24f89266d6e6f4da3c",
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

    def _filter_split(self, data: Tuple[str, Any], *, split: str) -> bool:
        return pathlib.Path(data[0]).parent.parent.name == split

    def _datapipe(self, resource_dps: List[IterDataPipe]) -> IterDataPipe[Dict[str, Any]]:
        dp = resource_dps[0]
        dp = Filter(dp, path_comparator("parent.parent.name", self._split_folder_name))
        dp = hint_shuffling(dp)
        dp = hint_sharding(dp)
        return Mapper(dp, self._prepare_sample)

    def __len__(self) -> int:
        return {
            "train": 31_650,
            "val": 10_550,
            "test": 21_100,
        }[self._split]

    def _generate_categories(self) -> List[str]:
        resources = self._resources()
        dp = resources[0].load(self._root)
        return sorted({pathlib.Path(path).parent.name for path, _ in dp})
