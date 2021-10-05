import io
import pathlib
import re
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import torch
from torch.utils.data import IterDataPipe
from torch.utils.data.datapipes.iter import (
    Mapper,
    Shuffler,
    Filter,
)
from torchdata.datapipes.iter import KeyZipper, CSVParser
from torchvision.prototype.datasets.utils import (
    Dataset,
    DatasetConfig,
    DatasetInfo,
    HttpResource,
    OnlineResource,
)
from torchvision.prototype.datasets.utils._internal import (
    create_categories_file,
    INFINITE_BUFFER_SIZE,
    RarArchiveReader,
)

HERE = pathlib.Path(__file__).parent


class HMDB51(Dataset):
    @property
    def info(self) -> DatasetInfo:
        return DatasetInfo(
            "hmdb51",
            type="video",
            categories=HERE / "hmdb51.categories",
            homepage="https://serre-lab.clps.brown.edu/resource/hmdb-a-large-human-motion-database/",
            valid_options=dict(
                split=("train", "test"),
                split_number=("1", "2", "3"),
            ),
        )

    def resources(self, config: DatasetConfig) -> List[OnlineResource]:
        splits = HttpResource(
            "http://serre-lab.clps.brown.edu/wp-content/uploads/2013/10/test_train_splits.rar",
            sha256="229c94f845720d01eb3946d39f39292ea962d50a18136484aa47c1eba251d2b7",
        )
        videos = HttpResource(
            "http://serre-lab.clps.brown.edu/wp-content/uploads/2013/10/hmdb51_org.rar",
            sha256="9e714a0d8b76104d76e932764a7ca636f929fff66279cda3f2e326fa912a328e",
        )
        return [splits, videos]

    _SPLIT_FILE_PATTERN = re.compile(r"(?P<category>\w+?)_test_split(?P<split_number>[1-3])[.]txt")

    def _is_split_number(self, data: Tuple[str, Any], *, config: DatasetConfig) -> bool:
        path = pathlib.Path(data[0])
        split_number = self._SPLIT_FILE_PATTERN.match(path.name).group("split_number")  # type: ignore[union-attr]
        return split_number == config.split_number

    _SPLIT_ID_TO_NAME = {
        "1": "train",
        "2": "test",
    }

    def _is_split(self, data: List[str], *, config=DatasetConfig) -> bool:
        split_id = data[1]
        if split_id not in self._SPLIT_ID_TO_NAME:
            return False
        return self._SPLIT_ID_TO_NAME[split_id] == config.split

    def _splits_key(self, data: List[str]) -> str:
        return data[0]

    def _videos_key(self, data: Tuple[str, Any]) -> str:
        path = pathlib.Path(data[0])
        return path.name

    def _collate_and_decode_sample(
        self, data: Tuple[List[str], Tuple[str, io.IOBase]], *, decoder: Optional[Callable[[io.IOBase], Dict[str, Any]]]
    ) -> Dict[str, Any]:
        _, video_data = data
        path, buffer = video_data

        category = pathlib.Path(path).parent.name
        label = torch.tensor(self.info.categories.index(category))

        sample = dict(
            path=path,
            category=category,
            label=label,
        )

        sample.update(decoder(buffer) if decoder else dict(video=buffer))
        return sample

    def _make_datapipe(
        self,
        resource_dps: List[IterDataPipe],
        *,
        config: DatasetConfig,
        decoder: Optional[Callable[[io.IOBase], Dict[str, Any]]],
    ) -> IterDataPipe[Dict[str, Any]]:
        splits_dp, videos_dp = resource_dps

        splits_dp = RarArchiveReader(splits_dp)
        splits_dp = Filter(splits_dp, self._is_split_number, fn_kwargs=dict(config=config))
        splits_dp = CSVParser(splits_dp, delimiter=" ")
        splits_dp = Filter(splits_dp, self._is_split, fn_kwargs=dict(config=config))
        splits_dp = Shuffler(splits_dp, buffer_size=INFINITE_BUFFER_SIZE)

        videos_dp = RarArchiveReader(videos_dp)
        videos_dp = RarArchiveReader(videos_dp)

        dp = KeyZipper(
            splits_dp,
            videos_dp,
            key_fn=self._splits_key,
            ref_key_fn=self._videos_key,
            buffer_size=INFINITE_BUFFER_SIZE,
        )
        return Mapper(dp, self._collate_and_decode_sample, fn_kwargs=dict(decoder=decoder))

    def generate_categories_file(self, root: Union[str, pathlib.Path]) -> None:
        splits_archive = self.resources(self.default_config)[0]
        dp = splits_archive.to_datapipe(pathlib.Path(root) / self.name)
        dp = RarArchiveReader(dp)

        categories = {
            self._SPLIT_FILE_PATTERN.match(pathlib.Path(path).name).group("category")  # type: ignore[union-attr]
            for path, _ in dp
        }
        create_categories_file(HERE, self.name, sorted(categories))


if __name__ == "__main__":
    from torchvision.prototype.datasets import home

    root = home()
    HMDB51().generate_categories_file(root)
