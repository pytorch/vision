from typing import Any, Dict, List, Tuple

import torch
from torchdata.datapipes.iter import (
    IterDataPipe,
    Mapper,
    CSVParser,
)
from torchvision.prototype.datasets.utils import (
    Dataset,
    DatasetConfig,
    DatasetInfo,
    HttpResource,
    OnlineResource,
)
from torchvision.prototype.datasets.utils._internal import hint_sharding, hint_shuffling
from torchvision.prototype.features import Image, OneHotLabel


class SEMEION(Dataset):
    def _make_info(self) -> DatasetInfo:
        return DatasetInfo(
            "semeion",
            categories=10,
            homepage="https://archive.ics.uci.edu/ml/datasets/Semeion+Handwritten+Digit",
        )

    def resources(self, config: DatasetConfig) -> List[OnlineResource]:
        data = HttpResource(
            "http://archive.ics.uci.edu/ml/machine-learning-databases/semeion/semeion.data",
            sha256="f43228ae3da5ea6a3c95069d53450b86166770e3b719dcc333182128fe08d4b1",
        )
        return [data]

    def _prepare_sample(self, data: Tuple[str, ...]) -> Dict[str, Any]:
        image_data, label_data = data[:256], data[256:-1]

        return dict(
            image=Image(torch.tensor([float(pixel) for pixel in image_data], dtype=torch.uint8).reshape(16, 16)),
            label=OneHotLabel([int(label) for label in label_data], categories=self.categories),
        )

    def _make_datapipe(
        self,
        resource_dps: List[IterDataPipe],
        *,
        config: DatasetConfig,
    ) -> IterDataPipe[Dict[str, Any]]:
        dp = resource_dps[0]
        dp = CSVParser(dp, delimiter=" ")
        dp = hint_sharding(dp)
        dp = hint_shuffling(dp)
        return Mapper(dp, self._prepare_sample)
