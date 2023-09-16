import pathlib
from typing import Any, Dict, List, Tuple, Union

import torch
from torchdata.datapipes.iter import CSVParser, IterDataPipe, Mapper
from torchvision.prototype.datasets.utils import Dataset, HttpResource, OnlineResource
from torchvision.prototype.datasets.utils._internal import hint_sharding, hint_shuffling
from torchvision.prototype.tv_tensors import OneHotLabel
from torchvision.tv_tensors import Image

from .._api import register_dataset, register_info

NAME = "semeion"


@register_info(NAME)
def _info() -> Dict[str, Any]:
    return dict(categories=[str(i) for i in range(10)])


@register_dataset(NAME)
class SEMEION(Dataset):
    """Semeion dataset
    homepage="https://archive.ics.uci.edu/ml/datasets/Semeion+Handwritten+Digit",
    """

    def __init__(self, root: Union[str, pathlib.Path], *, skip_integrity_check: bool = False) -> None:

        self._categories = _info()["categories"]
        super().__init__(root, skip_integrity_check=skip_integrity_check)

    def _resources(self) -> List[OnlineResource]:
        data = HttpResource(
            "http://archive.ics.uci.edu/ml/machine-learning-databases/semeion/semeion.data",
            sha256="f43228ae3da5ea6a3c95069d53450b86166770e3b719dcc333182128fe08d4b1",
        )
        return [data]

    def _prepare_sample(self, data: Tuple[str, ...]) -> Dict[str, Any]:
        image_data, label_data = data[:256], data[256:-1]

        return dict(
            image=Image(torch.tensor([float(pixel) for pixel in image_data], dtype=torch.float).reshape(16, 16)),
            label=OneHotLabel([int(label) for label in label_data], categories=self._categories),
        )

    def _datapipe(self, resource_dps: List[IterDataPipe]) -> IterDataPipe[Dict[str, Any]]:
        dp = resource_dps[0]
        dp = CSVParser(dp, delimiter=" ")
        dp = hint_shuffling(dp)
        dp = hint_sharding(dp)
        return Mapper(dp, self._prepare_sample)

    def __len__(self) -> int:
        return 1_593
