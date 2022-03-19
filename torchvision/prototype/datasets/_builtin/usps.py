import bz2
import functools
from typing import Any, Dict, List, Tuple, BinaryIO, Iterator

import numpy as np
import torch
from torchdata.datapipes.iter import IterDataPipe, IterableWrapper, LineReader, Mapper
from torchvision.prototype.datasets.utils import Dataset, DatasetInfo, DatasetConfig, OnlineResource, HttpResource
from torchvision.prototype.datasets.utils._internal import hint_sharding, hint_shuffling
from torchvision.prototype.features import Image, Label


class USPSFileReader(IterDataPipe[torch.Tensor]):
    def __init__(self, datapipe: IterDataPipe[Tuple[Any, BinaryIO]]) -> None:
        self.datapipe = datapipe

    def __iter__(self) -> Iterator[Tuple[torch.Tensor, torch.Tensor]]:
        for path, _ in self.datapipe:
            with bz2.open(path) as fp:
                datapipe = IterableWrapper([(path, fp)])
                line_reader = LineReader(datapipe, decode=True)
                for _, line in line_reader:
                    raw_data = line.split()
                    tmp_list = [x.split(":")[-1] for x in raw_data[1:]]
                    img = np.asarray(tmp_list, dtype=np.float32).reshape((-1, 16, 16))
                    img = ((img + 1) / 2 * 255).astype(dtype=np.uint8)
                    target = int(raw_data[0]) - 1
                    yield torch.from_numpy(img), torch.tensor(target)


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

    def _prepare_sample(self, data: Tuple[torch.Tensor, torch.Tensor], *, config: DatasetConfig) -> Dict[str, Any]:
        image, label = data
        return dict(
            image=Image(image),
            label=Label(label, dtype=torch.int64, categories=self.categories),
        )

    def _make_datapipe(
        self,
        resource_dps: List[IterDataPipe],
        *,
        config: DatasetConfig,
    ) -> IterDataPipe[Dict[str, Any]]:
        dp = USPSFileReader(resource_dps[0])
        dp = hint_sharding(dp)
        dp = hint_shuffling(dp)
        return Mapper(dp, functools.partial(self._prepare_sample, config=config))
