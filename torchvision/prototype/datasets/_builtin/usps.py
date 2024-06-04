import pathlib
from typing import Any, Dict, List, Union

import torch
from torchdata.datapipes.iter import Decompressor, IterDataPipe, LineReader, Mapper
from torchvision.prototype.datasets.utils import Dataset, HttpResource, OnlineResource
from torchvision.prototype.datasets.utils._internal import hint_sharding, hint_shuffling
from torchvision.prototype.tv_tensors import Label
from torchvision.tv_tensors import Image

from .._api import register_dataset, register_info

NAME = "usps"


@register_info(NAME)
def _info() -> Dict[str, Any]:
    return dict(categories=[str(c) for c in range(10)])


@register_dataset(NAME)
class USPS(Dataset):
    """USPS Dataset
    homepage="https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/multiclass.html#usps",
    """

    def __init__(
        self,
        root: Union[str, pathlib.Path],
        *,
        split: str = "train",
        skip_integrity_check: bool = False,
    ) -> None:
        self._split = self._verify_str_arg(split, "split", {"train", "test"})

        self._categories = _info()["categories"]
        super().__init__(root, skip_integrity_check=skip_integrity_check)

    _URL = "https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/multiclass"

    _RESOURCES = {
        "train": HttpResource(
            f"{_URL}/usps.bz2", sha256="3771e9dd6ba685185f89867b6e249233dd74652389f263963b3b741e994b034f"
        ),
        "test": HttpResource(
            f"{_URL}/usps.t.bz2", sha256="a9c0164e797d60142a50604917f0baa604f326e9a689698763793fa5d12ffc4e"
        ),
    }

    def _resources(self) -> List[OnlineResource]:
        return [USPS._RESOURCES[self._split]]

    def _prepare_sample(self, line: str) -> Dict[str, Any]:
        label, *values = line.strip().split(" ")
        values = [float(value.split(":")[1]) for value in values]
        pixels = torch.tensor(values).add_(1).div_(2)
        return dict(
            image=Image(pixels.reshape(16, 16)),
            label=Label(int(label) - 1, categories=self._categories),
        )

    def _datapipe(self, resource_dps: List[IterDataPipe]) -> IterDataPipe[Dict[str, Any]]:
        dp = Decompressor(resource_dps[0])
        dp = LineReader(dp, decode=True, return_path=False)
        dp = hint_shuffling(dp)
        dp = hint_sharding(dp)
        return Mapper(dp, self._prepare_sample)

    def __len__(self) -> int:
        return 7_291 if self._split == "train" else 2_007
