import io
import pathlib
from typing import Any, Callable, Dict, List, Optional, Tuple

from torch.utils.data import IterDataPipe
from torch.utils.data.datapipes.iter import (
    Filter,
    Mapper,
    ZipArchiveReader,
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
from torchvision.prototype.datasets.utils._internal import RarArchiveReader, INFINITE_BUFFER_SIZE
from torchvision.prototype.datasets.utils._internal import path_accessor, path_comparator


class UCF101(Dataset):
    """
    `UCF101 <https://www.crcv.ucf.edu/data/UCF101.php>`_ dataset.

    UCF101 is an action recognition video dataset, containing 101 classes
    of various human actions.
    """

    def _make_info(self) -> DatasetInfo:
        return DatasetInfo(
            "UCF101",
            type=DatasetType.VIDEO,
            valid_options={"split": ["train", "test"], "fold": ["1", "2", "3"]},
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
            ),
        ]

    def _generate_categories(self, root: pathlib.Path) -> List[str]:
        dp = self.resources(self.default_config)[1].to_datapipe(pathlib.Path(root) / self.name)
        dp = RarArchiveReader(dp)
        dir_names = {pathlib.Path(path).parent.name for path, _ in dp}
        return [name.split(".")[1] for name in sorted(dir_names)]

    def _collate_and_decode(
        self,
        data: Tuple[Tuple[str, int], Tuple[str, io.IOBase]],
        *,
        decoder: Optional[Callable[[io.IOBase], Dict[str, Any]]] = None,
    ) -> Dict[str, Any]:
        annotations_d, file_d = data
        label = annotations_d[1]
        _path, file_handle = file_d
        file = decoder(file_handle) if decoder else file_handle
        return {"path": _path, "file": file, "target": label}

    def _make_datapipe(
        self,
        resource_dps: List[IterDataPipe],
        *,
        config: DatasetConfig,
        decoder: Optional[Callable[[io.IOBase], Dict[str, Any]]],
    ) -> IterDataPipe[Dict[str, Any]]:

        annotations = resource_dps[0]
        files = resource_dps[1]

        annotations_dp = ZipArchiveReader(annotations)
        annotations_dp = Filter(annotations_dp, path_comparator("name", f"{config.split}list0{config.fold}.txt"))
        annotations_dp = CSVParser(annotations_dp, delimiter=" ")

        annotations_dp = Shuffler(annotations_dp, buffer_size=INFINITE_BUFFER_SIZE)

        files_dp = RarArchiveReader(files)
        dp = IterKeyZipper(annotations_dp, files_dp, path_accessor("name"))
        return Mapper(dp, self._collate_and_decode, fn_kwargs=dict(decoder=decoder))
