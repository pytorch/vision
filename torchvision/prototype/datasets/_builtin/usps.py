from typing import Any, Dict, List

import torch
from torchdata.datapipes.iter import IterDataPipe, LineReader, Mapper, Decompressor
from torchvision.prototype.datasets.utils import Dataset, DatasetInfo, DatasetConfig, OnlineResource, HttpResource
from torchvision.prototype.datasets.utils._internal import hint_sharding, hint_shuffling
from torchvision.prototype.features import Image, Label


class USPS(Dataset):
    def _make_info(self) -> DatasetInfo:
        return DatasetInfo(
            "usps",
            homepage="https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/multiclass.html#usps",
            valid_options=dict(
                split=("train", "test"),
            ),
            categories=10,
        )

    _URL = "https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/multiclass"

    _RESOURCES = {
        "train": HttpResource(
            f"{_URL}/usps.bz2", sha256="3771e9dd6ba685185f89867b6e249233dd74652389f263963b3b741e994b034f"
        ),
        "test": HttpResource(
            f"{_URL}/usps.t.bz2", sha256="a9c0164e797d60142a50604917f0baa604f326e9a689698763793fa5d12ffc4e"
        ),
    }

    def resources(self, config: DatasetConfig) -> List[OnlineResource]:
        return [USPS._RESOURCES[config.split]]

    def _prepare_sample(self, line: str) -> Dict[str, Any]:
        label, *values = line.strip().split(" ")
        values = [float(value.split(":")[1]) for value in values]
        pixels = torch.tensor(values).add_(1).div_(2)
        return dict(
            image=Image(pixels.reshape(16, 16)),
            label=Label(int(label) - 1, categories=self.categories),
        )

    def _make_datapipe(
        self,
        resource_dps: List[IterDataPipe],
        *,
        config: DatasetConfig,
    ) -> IterDataPipe[Dict[str, Any]]:
        dp = Decompressor(resource_dps[0])
        dp = LineReader(dp, decode=True, return_path=False)
        dp = hint_shuffling(dp)
        dp = hint_sharding(dp)
        return Mapper(dp, self._prepare_sample)
