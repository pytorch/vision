import functools
import io
from typing import Any, Callable, Dict, List, Optional, Union, cast

import torch
from torchdata.datapipes.iter import IterDataPipe, Mapper, CSVDictParser
from torchvision.prototype.datasets.decoder import raw
from torchvision.prototype.datasets.utils import (
    Dataset,
    DatasetConfig,
    DatasetInfo,
    OnlineResource,
    DatasetType,
    KaggleDownloadResource,
)
from torchvision.prototype.datasets.utils._internal import (
    hint_sharding,
    hint_shuffling,
    image_buffer_from_array,
)
from torchvision.prototype.features import Label, Image


class FER2013(Dataset):
    def _make_info(self) -> DatasetInfo:
        return DatasetInfo(
            "fer2013",
            type=DatasetType.RAW,
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

    def _collate_and_decode_sample(
        self,
        data: Dict[str, Any],
        *,
        decoder: Optional[Callable[[io.IOBase], torch.Tensor]],
    ) -> Dict[str, Any]:
        raw_image = torch.tensor([int(idx) for idx in data["pixels"].split()], dtype=torch.uint8).reshape(48, 48)
        label_id = data.get("emotion")
        label_idx = int(label_id) if label_id is not None else None

        image: Union[Image, io.BytesIO]
        if decoder is raw:
            image = Image(raw_image)
        else:
            image_buffer = image_buffer_from_array(raw_image.numpy())
            image = decoder(image_buffer) if decoder else image_buffer  # type: ignore[assignment]

        return dict(
            image=image,
            label=Label(label_idx, category=self.info.categories[label_idx]) if label_idx is not None else None,
        )

    def _make_datapipe(
        self,
        resource_dps: List[IterDataPipe],
        *,
        config: DatasetConfig,
        decoder: Optional[Callable[[io.IOBase], torch.Tensor]],
    ) -> IterDataPipe[Dict[str, Any]]:
        dp = resource_dps[0]
        dp = CSVDictParser(dp)
        dp = hint_sharding(dp)
        dp = hint_shuffling(dp)
        return Mapper(dp, functools.partial(self._collate_and_decode_sample, decoder=decoder))
