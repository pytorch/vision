import io
import pathlib
from typing import Any, Callable, Dict, List, Optional, Tuple

import torch
from torch.utils.data import IterDataPipe
from torch.utils.data.datapipes.iter import (
    Mapper,
    TarArchiveReader,
    FileLoader,
)

from torchvision.prototype.datasets.utils import (
    Dataset,
    DatasetConfig,
    DatasetInfo,
    HttpResource,
)


class Caltech256(Dataset):
    @property
    def info(self) -> DatasetInfo:
        return DatasetInfo(
            "caltech256",
            homepage="http://www.vision.caltech.edu/Image_Datasets/Caltech256",
        )

    def resources(self, config: DatasetConfig) -> List[HttpResource]:
        return [
            HttpResource(
                "http://www.vision.caltech.edu/Image_Datasets/Caltech256/256_ObjectCategories.tar",
                sha256="08ff01b03c65566014ae88eb0490dbe4419fc7ac4de726ee1163e39fd809543e",
            )
        ]

    def _collate_and_decode_sample(
        self,
        data: Tuple[str, io.BufferedIOBase],
        *,
        decoder: Optional[Callable[[str, io.BufferedIOBase], torch.Tensor]],
    ) -> Dict[str, Any]:
        path, image = data

        dir_name = pathlib.Path(path).parent.name
        label_str, category = dir_name.split(".")
        label = torch.tensor(int(label_str))

        if decoder:
            image = decoder(path, image)

        return dict(label=label, category=category, image=image)

    def _make_datapipe(
        self,
        resource_dps: List[IterDataPipe],
        *,
        config: DatasetConfig,
        decoder: Optional[Callable[[str, io.BufferedIOBase], torch.Tensor]],
    ) -> IterDataPipe[Dict[str, Any]]:
        dp = resource_dps[0]
        dp = TarArchiveReader(dp)
        return Mapper(
            dp, self._collate_and_decode_sample, fn_kwargs=dict(decoder=decoder)
        )
