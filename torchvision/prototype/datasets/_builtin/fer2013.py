from typing import Any, Dict, List, cast

import torch
from torchdata.datapipes.iter import IterDataPipe, Mapper, CSVDictParser
from torchvision.prototype.datasets.utils import (
    Dataset,
    DatasetConfig,
    DatasetInfo,
    OnlineResource,
    KaggleDownloadResource,
)
from torchvision.prototype.datasets.utils._internal import (
    hint_sharding,
    hint_shuffling,
)
from torchvision.prototype.features import Label, Image


class FER2013(Dataset):
    def _make_info(self) -> DatasetInfo:
        return DatasetInfo(
            "fer2013",
            homepage="https://www.kaggle.com/c/challenges-in-representation-learning-facial-expression-recognition-challenge",
            categories=("angry", "disgust", "fear", "happy", "sad", "surprise", "neutral"),
            valid_options=dict(split=("train", "test")),
        )

    _CHECKSUMS = {
        "train": "a2b7c9360cc0b38d21187e5eece01c2799fce5426cdeecf746889cc96cda2d10",
        "test": "dec8dfe8021e30cd6704b85ec813042b4a5d99d81cb55e023291a94104f575c3",
    }

    def resources(self, config: DatasetConfig) -> List[OnlineResource]:
        archive = KaggleDownloadResource(
            cast(str, self.info.homepage),
            file_name=f"{config.split}.csv.zip",
            sha256=self._CHECKSUMS[config.split],
        )
        return [archive]

    def _prepare_sample(self, data: Dict[str, Any]) -> Dict[str, Any]:
        label_id = data.get("emotion")

        return dict(
            image=Image(torch.tensor([int(idx) for idx in data["pixels"].split()], dtype=torch.uint8).reshape(48, 48)),
            label=Label(int(label_id), categories=self.categories) if label_id is not None else None,
        )

    def _make_datapipe(
        self,
        resource_dps: List[IterDataPipe],
        *,
        config: DatasetConfig,
    ) -> IterDataPipe[Dict[str, Any]]:
        dp = resource_dps[0]
        dp = CSVDictParser(dp)
        dp = hint_sharding(dp)
        dp = hint_shuffling(dp)
        return Mapper(dp, self._prepare_sample)
