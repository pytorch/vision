from typing import Any, Dict, List, Tuple

import numpy as np
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

    def _prepare_sample(self, data: Tuple[torch.Tensor, torch.Tensor]) -> Dict[str, Any]:
        _filename, line = data

        raw_data = line.split()
        tmp_list = [x.split(":")[-1] for x in raw_data[1:]]
        img = np.asarray(tmp_list, dtype=np.float32).reshape((-1, 16, 16))
        img = ((img + 1) / 2 * 255).astype(dtype=np.uint8)
        img = torch.from_numpy(img)
        target = int(raw_data[0]) - 1

        return dict(
            image=Image(img),
            label=Label(target, dtype=torch.int64, categories=self.categories),
        )

    def _make_datapipe(
        self,
        resource_dps: List[IterDataPipe],
        *,
        config: DatasetConfig,
    ) -> IterDataPipe[Dict[str, Any]]:
        dp = Decompressor(resource_dps[0])
        dp = LineReader(dp, decode=True)
        dp = hint_sharding(dp)
        dp = hint_shuffling(dp)
        return Mapper(dp, self._prepare_sample)
