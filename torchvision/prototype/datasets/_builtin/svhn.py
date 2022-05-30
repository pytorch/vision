import functools
import io
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
import torch
from torchdata.datapipes.iter import (
    IterDataPipe,
    Mapper,
    UnBatcher,
)
from torchvision.prototype.datasets.decoder import raw
from torchvision.prototype.datasets.utils import (
    Dataset,
    DatasetConfig,
    DatasetInfo,
    HttpResource,
    OnlineResource,
    DatasetType,
)
from torchvision.prototype.datasets.utils._internal import (
    read_mat,
    hint_sharding,
    hint_shuffling,
    image_buffer_from_array,
)
from torchvision.prototype.features import Label, Image


class SVHN(Dataset):
    def _make_info(self) -> DatasetInfo:
        return DatasetInfo(
            "svhn",
            type=DatasetType.RAW,
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

    def _read_images_and_labels(self, data: Tuple[str, io.IOBase]) -> List[Tuple[np.ndarray, np.ndarray]]:
        _, buffer = data
        content = read_mat(buffer)
        return list(
            zip(
                content["X"].transpose((3, 0, 1, 2)),
                content["y"].squeeze(),
            )
        )

    def _collate_and_decode_sample(
        self,
        data: Tuple[np.ndarray, np.ndarray],
        *,
        decoder: Optional[Callable[[io.IOBase], torch.Tensor]],
    ) -> Dict[str, Any]:
        image_array, label_array = data

        if decoder is raw:
            image = Image(image_array.transpose((2, 0, 1)))
        else:
            image_buffer = image_buffer_from_array(image_array)
            image = decoder(image_buffer) if decoder else image_buffer  # type: ignore[assignment]

        return dict(
            image=image,
            label=Label(int(label_array) % 10),
        )

    def _make_datapipe(
        self,
        resource_dps: List[IterDataPipe],
        *,
        config: DatasetConfig,
        decoder: Optional[Callable[[io.IOBase], torch.Tensor]],
    ) -> IterDataPipe[Dict[str, Any]]:
        dp = resource_dps[0]
        dp = Mapper(dp, self._read_images_and_labels)
        dp = UnBatcher(dp)
        dp = hint_sharding(dp)
        dp = hint_shuffling(dp)
        return Mapper(dp, functools.partial(self._collate_and_decode_sample, decoder=decoder))
