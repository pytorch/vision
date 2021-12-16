from typing import Any, Dict, List, Tuple

import torch
from torchdata.datapipes.iter import (
    IterDataPipe,
    Mapper,
    Shuffler,
    CSVParser,
)
from torchvision.prototype.datasets.utils import (
    Dataset,
    DatasetConfig,
    DatasetInfo,
    HttpResource,
    OnlineResource,
)
from torchvision.prototype.datasets.utils._internal import INFINITE_BUFFER_SIZE
from torchvision.prototype.features import Image, Label


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
        image_data = torch.tensor([float(pixel) for pixel in data[:256]], dtype=torch.uint8).reshape(16, 16)
        label_data = [int(label) for label in data[256:] if label]

        label_idx = next((idx for idx, one_hot_label in enumerate(label_data) if one_hot_label))
        return dict(
            image=Image(image_data.unsqueeze(0)), label=Label(label_idx, category=self.info.categories[label_idx])
        )

    def _make_datapipe(
        self,
        resource_dps: List[IterDataPipe],
        *,
        config: DatasetConfig,
    ) -> IterDataPipe[Dict[str, Any]]:
        dp = resource_dps[0]
        dp = CSVParser(dp, delimiter=" ")
        dp = Shuffler(dp, buffer_size=INFINITE_BUFFER_SIZE)
        dp = Mapper(dp, self._prepare_sample)
        return dp
