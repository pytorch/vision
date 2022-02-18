import itertools
import pathlib
from typing import Any, Dict, List, Tuple, BinaryIO, Iterator

from torchdata.datapipes.iter import IterDataPipe, Mapper, IterKeyZipper, LineReader
from torchvision.prototype.datasets.utils import (
    Dataset,
    DatasetConfig,
    DatasetInfo,
    HttpResource,
    OnlineResource,
)
from torchvision.prototype.datasets.utils._internal import (
    INFINITE_BUFFER_SIZE,
    hint_sharding,
    hint_shuffling,
    path_accessor,
    getitem,
)
from torchvision.prototype.features import Label, EncodedImage


class LFWSplitFileParser(IterDataPipe[Tuple[str, str]]):
    def __init__(self, datapipe: IterDataPipe[str], *, split: str) -> None:
        self.datapipe = datapipe
        self.split = split

    def __iter__(self) -> Iterator[Tuple[str, str]]:
        lines = iter(self.datapipe)

        if self.split not in {"train", "test"}:
            next(lines)
            for part in range(1, int(self.split)):
                lines = itertools.islice(lines, int(next(lines)), None)

        for line in itertools.islice(lines, int(next(lines))):
            category, num_images = line.split("\t")
            for idx in range(int(num_images)):
                yield category, f"{category}_{idx + 1:04d}.jpg"


class LFWPeople(Dataset):
    def _make_info(self) -> DatasetInfo:
        return DatasetInfo(
            "lfw-people",
            homepage="http://vis-www.cs.umass.edu/lfw/",
            valid_options=dict(
                split=("train", "test", *[str(fold) for fold in range(1, 11)]),
                image_set=("original", "funneled", "deep_funneled"),
            ),
        )

    _CHECKSUMS = {
        "lfw.tgz": "055f7d9c632d7370e6fb4afc7468d40f970c34a80d4c6f50ffec63f5a8d536c0",
        "lfw-funneled.tgz": "b47c8422c8cded889dc5a13418c4bc2abbda121092b3533a83306f90d900100a",
        "lfw-deepfunneled.tgz": "08575363d69edaed9a7c3ccb9e2eebd76c91bc1011b2baaf725f8d1284bf3a2f",
        "peopleDevTrain.txt": "2e58df13ba4cc143189336e001cc2c195f1dfa5b29a6475c02857be353c2a2fa",
        "peopleDevTest.txt": "c3ed1f90499473401542101a56c14c1d82396bb8218c67a356d7460601047555",
        "people.txt": "bc8fa2b7f25c895cf7efb6de4f4fc27ca8b8d1cfa63350ad8b96eae2b198d365",
    }

    def resources(self, config: DatasetConfig) -> List[OnlineResource]:
        url_root = "http://vis-www.cs.umass.edu/lfw"

        file_name = {
            "original": "lfw.tgz",
            "funneled": "lfw-funneled.tgz",
            "deep_funneled": "lfw-deepfunneled.tgz",
        }[config.image_set]
        images = HttpResource(
            f"{url_root}/{file_name}",
            sha256=self._CHECKSUMS[file_name],
            decompress=True,
        )

        file_name = {
            "train": "peopleDevTrain.txt",
            "test": "peopleDevTest.txt",
        }.get(config.split, "people.txt")
        split = HttpResource(f"{url_root}/{file_name}", sha256=self._CHECKSUMS[file_name])

        return [images, split]

    def _prepare_sample(self, data: Tuple[Tuple[str, str], Tuple[str, BinaryIO]]) -> Dict[str, Any]:
        (category, _), (path, buffer) = data

        return dict(
            label=Label.from_category(category, categories=self.categories),
            path=path,
            image=EncodedImage.from_file(buffer),
        )

    def _make_datapipe(
        self, resource_dps: List[IterDataPipe], *, config: DatasetConfig
    ) -> IterDataPipe[Dict[str, Any]]:
        images_dp, split_dp = resource_dps

        split_dp = LineReader(split_dp, decode=True, return_path=False)
        split_dp = LFWSplitFileParser(split_dp, split=config.split)
        split_dp = hint_sharding(split_dp)
        split_dp = hint_shuffling(split_dp)

        dp = IterKeyZipper(
            split_dp,
            images_dp,
            key_fn=getitem(1),
            ref_key_fn=path_accessor("name"),
            buffer_size=INFINITE_BUFFER_SIZE,
        )
        return Mapper(dp, self._prepare_sample)

    def _generate_categories(self, root: pathlib.Path) -> List[str]:
        config = self.default_config
        resources = self.resources(config)

        dp = resources[0].load(root)
        dp = Mapper(dp, path_accessor("parent.name"))

        return sorted(set(dp))
