import functools
import pathlib
import re
from typing import Any, Dict, List, Tuple, BinaryIO

from torchdata.datapipes.iter import IterDataPipe, Mapper, Filter, CSVDictParser, IterKeyZipper
from torchvision.prototype.datasets.utils import (
    Dataset,
    DatasetConfig,
    DatasetInfo,
    HttpResource,
    OnlineResource,
)
from torchvision.prototype.datasets.utils._internal import (
    INFINITE_BUFFER_SIZE,
    getitem,
    path_accessor,
    hint_sharding,
    hint_shuffling,
)
from torchvision.prototype.features import EncodedVideo, Label


class HMDB51(Dataset):
    def _make_info(self) -> DatasetInfo:
        return DatasetInfo(
            "hmdb51",
            homepage="https://serre-lab.clps.brown.edu/resource/hmdb-a-large-human-motion-database/",
            dependencies=("rarfile",),
            valid_options=dict(
                split=("train", "test"),
                fold=("1", "2", "3"),
            ),
        )

    def _extract_videos_archive(self, path: pathlib.Path) -> pathlib.Path:
        folder = OnlineResource._extract(path)
        for rar_file in folder.glob("*.rar"):
            OnlineResource._extract(rar_file)
            rar_file.unlink()
        return folder

    def resources(self, config: DatasetConfig) -> List[OnlineResource]:
        url_root = "https://serre-lab.clps.brown.edu/wp-content/uploads/2013/10"

        splits = HttpResource(
            f"{url_root}/test_train_splits.rar",
            sha256="229c94f845720d01eb3946d39f39292ea962d50a18136484aa47c1eba251d2b7",
        )
        videos = HttpResource(
            f"{url_root}/hmdb51_org.rar",
            sha256="9e714a0d8b76104d76e932764a7ca636f929fff66279cda3f2e326fa912a328e",
        )
        videos._preprocess = self._extract_videos_archive
        return [splits, videos]

    _SPLIT_FILE_PATTERN = re.compile(r"(?P<category>\w+?)_test_split(?P<fold>[1-3])[.]txt")

    def _is_fold(self, data: Tuple[str, Any], *, fold: str) -> bool:
        path = pathlib.Path(data[0])
        return self._SPLIT_FILE_PATTERN.match(path.name)["fold"] == fold  # type: ignore[index]

    _SPLIT_ID_TO_NAME = {
        "1": "train",
        "2": "test",
    }

    def _is_split(self, data: Dict[str, Any], *, split: str) -> bool:
        split_id = data["split_id"]

        # In addition to split id 1 and 2 corresponding to the train and test splits, some videos are annotated with
        # split id 0, which indicates that the video is not included in either split
        if split_id not in self._SPLIT_ID_TO_NAME:
            return False

        return self._SPLIT_ID_TO_NAME[split_id] == split

    def _prepare_sample(self, data: Tuple[List[str], Tuple[str, BinaryIO]]) -> Dict[str, Any]:
        _, (path, buffer) = data
        path = pathlib.Path(path)
        return dict(
            label=Label.from_category(path.parent.name, categories=self.categories),
            video=EncodedVideo.from_file(buffer, path=path),
        )

    def _make_datapipe(
        self,
        resource_dps: List[IterDataPipe],
        *,
        config: DatasetConfig,
    ) -> IterDataPipe[Dict[str, Any]]:
        splits_dp, videos_dp = resource_dps

        splits_dp = Filter(splits_dp, functools.partial(self._is_fold, fold=config.fold))
        splits_dp = CSVDictParser(splits_dp, fieldnames=("filename", "split_id"), delimiter=" ")
        splits_dp = Filter(splits_dp, functools.partial(self._is_split, split=config.split))
        splits_dp = hint_sharding(splits_dp)
        splits_dp = hint_shuffling(splits_dp)

        dp = IterKeyZipper(
            splits_dp,
            videos_dp,
            key_fn=getitem("filename"),
            ref_key_fn=path_accessor("name"),
            buffer_size=INFINITE_BUFFER_SIZE,
        )
        return Mapper(dp, self._prepare_sample)

    def _generate_categories(self, root: pathlib.Path) -> List[str]:
        config = self.default_config
        resources = self.resources(config)

        dp = resources[0].load(root)
        categories = {
            self._SPLIT_FILE_PATTERN.match(pathlib.Path(path).name)["category"] for path, _ in dp  # type: ignore[index]
        }
        return sorted(categories)
