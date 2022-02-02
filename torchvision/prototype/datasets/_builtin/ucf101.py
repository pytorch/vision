import csv
import pathlib
from typing import Any, Dict, List, Tuple, cast, BinaryIO

from torch.utils.data import IterDataPipe
from torch.utils.data.datapipes.iter import Filter, Mapper
from torchdata.datapipes.iter import CSVParser, IterKeyZipper
from torchvision.prototype.datasets.utils import (
    Dataset,
    DatasetConfig,
    DatasetInfo,
    HttpResource,
    OnlineResource,
)
from torchvision.prototype.datasets.utils._internal import (
    path_accessor,
    path_comparator,
    hint_sharding,
    hint_shuffling,
    INFINITE_BUFFER_SIZE,
)
from torchvision.prototype.features import Label, EncodedVideo

csv.register_dialect("ucf101", delimiter=" ")


class UCF101(Dataset):
    def _make_info(self) -> DatasetInfo:
        return DatasetInfo(
            "ucf101",
            dependencies=("rarfile",),
            valid_options=dict(
                split=("train", "test"),
                fold=("1", "2", "3"),
            ),
            homepage="https://www.crcv.ucf.edu/data/UCF101.php",
        )

    def _extract_videos_archive(self, path: pathlib.Path) -> pathlib.Path:
        folder = OnlineResource._extract(path)
        for rar_file in folder.glob("*.rar"):
            OnlineResource._extract(rar_file)
            rar_file.unlink()
        return folder

    def resources(self, config: DatasetConfig) -> List[OnlineResource]:
        url_root = "https://www.crcv.ucf.edu/data/UCF101/"

        splits = HttpResource(
            f"{url_root}/UCF101TrainTestSplits-RecognitionTask.zip",
            sha256="5c0d1a53b8ed364a2ac830a73f405e51bece7d98ce1254fd19ed4a36b224bd27",
        )

        videos = HttpResource(
            f"{url_root}/UCF101.rar",
            sha256="ca8dfadb4c891cb11316f94d52b6b0ac2a11994e67a0cae227180cd160bd8e55",
            extract=True,
        )
        videos._preprocess = self._extract_videos_archive

        return [splits, videos]

    def _prepare_sample(self, data: Tuple[Tuple[str, str], Tuple[str, BinaryIO]]) -> Dict[str, Any]:
        _, (path, buffer) = data
        path = pathlib.Path(path)
        return dict(
            label=Label.from_category(path.parent.name, categories=self.categories),
            video=EncodedVideo.from_file(buffer, path=path),
        )

    def _make_datapipe(
        self, resource_dps: List[IterDataPipe], *, config: DatasetConfig
    ) -> IterDataPipe[Dict[str, Any]]:
        splits_dp, images_dp = resource_dps

        splits_dp: IterDataPipe[Tuple[str, BinaryIO]] = Filter(
            splits_dp, path_comparator("name", f"{config.split}list0{config.fold}.txt")
        )
        splits_dp = CSVParser(splits_dp, dialect="ucf101")
        splits_dp = hint_sharding(splits_dp)
        splits_dp = hint_shuffling(splits_dp)

        dp = IterKeyZipper(splits_dp, images_dp, path_accessor("name"), buffer_size=INFINITE_BUFFER_SIZE)
        return Mapper(dp, self._prepare_sample)

    def _generate_categories(self, root: pathlib.Path) -> Tuple[str, ...]:
        config = self.default_config
        resources = self.resources(config)

        dp = resources[0].load(root)
        dp: IterDataPipe[Tuple[str, BinaryIO]] = Filter(dp, path_comparator("name", "classInd.txt"))
        dp = CSVParser(dp, dialect="ucf101")
        _, categories = zip(*dp)
        return cast(Tuple[str, ...], categories)
