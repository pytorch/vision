from typing import Any, Dict, List, Tuple, BinaryIO

import numpy as np
from torchdata.datapipes.iter import (
    IterDataPipe,
    Mapper,
    UnBatcher,
)
from torchvision.prototype.datasets.utils import (
    Dataset,
    DatasetConfig,
    DatasetInfo,
    HttpResource,
    OnlineResource,
)
from torchvision.prototype.datasets.utils._internal import (
    read_mat,
    hint_sharding,
    hint_shuffling,
)
from torchvision.prototype.features import Label, Image


class SVHN(Dataset):
    def _make_info(self) -> DatasetInfo:
        return DatasetInfo(
            "svhn",
            dependencies=("scipy",),
            categories=10,
            homepage="http://ufldl.stanford.edu/housenumbers/",
            valid_options=dict(split=("train", "test", "extra")),
        )

    _CHECKSUMS = {
        "train": "435e94d69a87fde4fd4d7f3dd208dfc32cb6ae8af2240d066de1df7508d083b8",
        "test": "cdce80dfb2a2c4c6160906d0bd7c68ec5a99d7ca4831afa54f09182025b6a75b",
        "extra": "a133a4beb38a00fcdda90c9489e0c04f900b660ce8a316a5e854838379a71eb3",
    }

    def resources(self, config: DatasetConfig) -> List[OnlineResource]:
        data = HttpResource(
            f"http://ufldl.stanford.edu/housenumbers/{config.split}_32x32.mat",
            sha256=self._CHECKSUMS[config.split],
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
            label=Label(int(label_array) % 10, categories=self.categories),
        )

    def _make_datapipe(
        self,
        resource_dps: List[IterDataPipe],
        *,
        config: DatasetConfig,
    ) -> IterDataPipe[Dict[str, Any]]:
        dp = resource_dps[0]
        dp = Mapper(dp, self._read_images_and_labels)
        dp = UnBatcher(dp)
        dp = hint_sharding(dp)
        dp = hint_shuffling(dp)
        return Mapper(dp, self._prepare_sample)
