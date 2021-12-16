import io
import pathlib
from typing import Any, Callable, Dict, List, Optional, Tuple

import torch
from torch.utils.data import IterDataPipe
from torch.utils.data.datapipes.iter import (
    Filter,
    Mapper,
    Shuffler,
)
from torchdata.datapipes.iter import CSVParser, IterKeyZipper
from torchvision.prototype.datasets.utils import (
    Dataset,
    DatasetConfig,
    DatasetInfo,
    HttpResource,
    OnlineResource,
    DatasetType,
)
from torchvision.prototype.datasets.utils._internal import (
    INFINITE_BUFFER_SIZE,
    path_accessor,
    path_comparator,
    hint_sharding,
)
from torchvision.prototype.features import Label


class UCF101(Dataset):
    """
    `UCF101 <https://www.crcv.ucf.edu/data/UCF101.php>`_ dataset.

    UCF101 is an action recognition video dataset, containing 101 classes
    of various human actions.
    """

    def _make_info(self) -> DatasetInfo:
        return DatasetInfo(
            "ucf101",
            type=DatasetType.VIDEO,
            dependencies=("rarfile",),
            valid_options=dict(
                split=("train", "test"),
                fold=("1", "2", "3"),
            ),
            homepage="https://www.crcv.ucf.edu/data/UCF101.php",
        )

    def resources(self, config: DatasetConfig) -> List[OnlineResource]:
        return [
            HttpResource(
                "https://www.crcv.ucf.edu/data/UCF101/UCF101TrainTestSplits-RecognitionTask.zip",
                sha256="5c0d1a53b8ed364a2ac830a73f405e51bece7d98ce1254fd19ed4a36b224bd27",
            ),
            HttpResource(
                "https://www.crcv.ucf.edu/data/UCF101/UCF101.rar",
                sha256="ca8dfadb4c891cb11316f94d52b6b0ac2a11994e67a0cae227180cd160bd8e55",
                extract=True,
            ),
        ]

    def _collate_and_decode(
        self,
        data: Tuple[Tuple[str, int], Tuple[str, io.IOBase]],
        *,
        decoder: Optional[Callable[[io.IOBase], Dict[str, Any]]] = None,
    ) -> Dict[str, Any]:
        split_data, image_data = data
        _, label_idx = split_data
        path, buffer = image_data
        return dict(
            label=Label(label_idx, category=self.categories[label_idx]),
            path=path,
            video=decoder(buffer) if decoder else buffer,
        )

    def _make_datapipe(
        self,
        resource_dps: List[IterDataPipe],
        *,
        config: DatasetConfig,
        decoder: Optional[Callable[[io.IOBase], torch.Tensor]],
    ) -> IterDataPipe[Dict[str, Any]]:
        splits_dp, images_dp = resource_dps

        splits_dp = Filter(splits_dp, path_comparator("name", f"{config.split}list0{config.fold}.txt"))
        splits_dp = CSVParser(splits_dp, delimiter=" ")
        splits_dp = hint_sharding(splits_dp)
        splits_dp = Shuffler(splits_dp, buffer_size=INFINITE_BUFFER_SIZE)

        dp = IterKeyZipper(splits_dp, images_dp, path_accessor("name"))
        return Mapper(dp, self._collate_and_decode, fn_kwargs=dict(decoder=decoder))

    def _generate_categories(self, root: pathlib.Path) -> List[str]:
        dp = self.resources(self.default_config)[1].load(pathlib.Path(root) / self.name)
        dir_names = {pathlib.Path(path).parent.name for path, _ in dp}
        return [name.split(".")[1] for name in sorted(dir_names)]
