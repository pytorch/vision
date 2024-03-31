import pathlib
from typing import Any, BinaryIO, Dict, List, Tuple, Union

import numpy as np
from torchdata.datapipes.iter import IterDataPipe, Mapper, UnBatcher
from torchvision.prototype.datasets.utils import Dataset, HttpResource, OnlineResource
from torchvision.prototype.datasets.utils._internal import hint_sharding, hint_shuffling, read_mat
from torchvision.prototype.tv_tensors import Label
from torchvision.tv_tensors import Image

from .._api import register_dataset, register_info

NAME = "svhn"


@register_info(NAME)
def _info() -> Dict[str, Any]:
    return dict(categories=[str(c) for c in range(10)])


@register_dataset(NAME)
class SVHN(Dataset):
    """SVHN Dataset.
    homepage="http://ufldl.stanford.edu/housenumbers/",
    dependencies = scipy
    """

    def __init__(
        self,
        root: Union[str, pathlib.Path],
        *,
        split: str = "train",
        skip_integrity_check: bool = False,
    ) -> None:
        self._split = self._verify_str_arg(split, "split", {"train", "test", "extra"})
        self._categories = _info()["categories"]
        super().__init__(root, skip_integrity_check=skip_integrity_check, dependencies=("scipy",))

    _CHECKSUMS = {
        "train": "435e94d69a87fde4fd4d7f3dd208dfc32cb6ae8af2240d066de1df7508d083b8",
        "test": "cdce80dfb2a2c4c6160906d0bd7c68ec5a99d7ca4831afa54f09182025b6a75b",
        "extra": "a133a4beb38a00fcdda90c9489e0c04f900b660ce8a316a5e854838379a71eb3",
    }

    def _resources(self) -> List[OnlineResource]:
        data = HttpResource(
            f"http://ufldl.stanford.edu/housenumbers/{self._split}_32x32.mat",
            sha256=self._CHECKSUMS[self._split],
        )

        return [data]

    def _read_images_and_labels(self, data: Tuple[str, BinaryIO]) -> List[Tuple[np.ndarray, np.ndarray]]:
        _, buffer = data
        content = read_mat(buffer)
        return list(
            zip(
                content["X"].transpose((3, 0, 1, 2)),
                content["y"].squeeze(),
            )
        )

    def _prepare_sample(self, data: Tuple[np.ndarray, np.ndarray]) -> Dict[str, Any]:
        image_array, label_array = data

        return dict(
            image=Image(image_array.transpose((2, 0, 1))),
            label=Label(int(label_array) % 10, categories=self._categories),
        )

    def _datapipe(self, resource_dps: List[IterDataPipe]) -> IterDataPipe[Dict[str, Any]]:
        dp = resource_dps[0]
        dp = Mapper(dp, self._read_images_and_labels)
        dp = UnBatcher(dp)
        dp = hint_shuffling(dp)
        dp = hint_sharding(dp)
        return Mapper(dp, self._prepare_sample)

    def __len__(self) -> int:
        return {
            "train": 73_257,
            "test": 26_032,
            "extra": 531_131,
        }[self._split]
