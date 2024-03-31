import pathlib
from typing import Any, Dict, List, Union

import torch
from torchdata.datapipes.iter import CSVDictParser, IterDataPipe, Mapper
from torchvision.prototype.datasets.utils import Dataset, KaggleDownloadResource, OnlineResource
from torchvision.prototype.datasets.utils._internal import hint_sharding, hint_shuffling
from torchvision.prototype.tv_tensors import Label
from torchvision.tv_tensors import Image

from .._api import register_dataset, register_info

NAME = "fer2013"


@register_info(NAME)
def _info() -> Dict[str, Any]:
    return dict(categories=("angry", "disgust", "fear", "happy", "sad", "surprise", "neutral"))


@register_dataset(NAME)
class FER2013(Dataset):
    """FER 2013 Dataset
    homepage="https://www.kaggle.com/c/challenges-in-representation-learning-facial-expression-recognition-challenge"
    """

    def __init__(
        self, root: Union[str, pathlib.Path], *, split: str = "train", skip_integrity_check: bool = False
    ) -> None:
        self._split = self._verify_str_arg(split, "split", {"train", "test"})
        self._categories = _info()["categories"]

        super().__init__(root, skip_integrity_check=skip_integrity_check)

    _CHECKSUMS = {
        "train": "a2b7c9360cc0b38d21187e5eece01c2799fce5426cdeecf746889cc96cda2d10",
        "test": "dec8dfe8021e30cd6704b85ec813042b4a5d99d81cb55e023291a94104f575c3",
    }

    def _resources(self) -> List[OnlineResource]:
        archive = KaggleDownloadResource(
            "https://www.kaggle.com/c/challenges-in-representation-learning-facial-expression-recognition-challenge",
            file_name=f"{self._split}.csv.zip",
            sha256=self._CHECKSUMS[self._split],
        )
        return [archive]

    def _prepare_sample(self, data: Dict[str, Any]) -> Dict[str, Any]:
        label_id = data.get("emotion")

        return dict(
            image=Image(torch.tensor([int(idx) for idx in data["pixels"].split()], dtype=torch.uint8).reshape(48, 48)),
            label=Label(int(label_id), categories=self._categories) if label_id is not None else None,
        )

    def _datapipe(self, resource_dps: List[IterDataPipe]) -> IterDataPipe[Dict[str, Any]]:
        dp = resource_dps[0]
        dp = CSVDictParser(dp)
        dp = hint_shuffling(dp)
        dp = hint_sharding(dp)
        return Mapper(dp, self._prepare_sample)

    def __len__(self) -> int:
        return 28_709 if self._split == "train" else 3_589
