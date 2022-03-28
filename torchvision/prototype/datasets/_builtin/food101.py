from pathlib import Path
from typing import Any, Tuple, List, Dict, Optional, BinaryIO

from torchdata.datapipes.iter import (
    IterDataPipe,
    Filter,
    Mapper,
    LineReader,
    Demultiplexer,
    IterKeyZipper,
)
from torchvision.prototype.datasets.utils import Dataset, DatasetInfo, DatasetConfig, HttpResource, OnlineResource
from torchvision.prototype.datasets.utils._internal import (
    hint_shuffling,
    hint_sharding,
    path_comparator,
    getitem,
    INFINITE_BUFFER_SIZE,
)
from torchvision.prototype.features import Label, EncodedImage


class Food101(Dataset):
    def _make_info(self) -> DatasetInfo:
        return DatasetInfo(
            "food101",
            homepage="https://data.vision.ee.ethz.ch/cvl/datasets_extra/food-101",
            valid_options=dict(split=("train", "test")),
        )

    def resources(self, config: DatasetConfig) -> List[OnlineResource]:
        return [
            HttpResource(
                url="http://data.vision.ee.ethz.ch/cvl/food-101.tar.gz",
                sha256="d97d15e438b7f4498f96086a4f7e2fa42a32f2712e87d3295441b2b6314053a4",
                preprocess="decompress",
            )
        ]

    def _classify_archive(self, data: Tuple[str, Any]) -> Optional[int]:
        path = Path(data[0])
        if path.parents[1].name == "images":
            return 0
        elif path.parents[0].name == "meta":
            return 1
        else:
            return None

    def _prepare_sample(self, data: Tuple[str, Tuple[str, BinaryIO]]) -> Dict[str, Any]:
        id, (path, buffer) = data
        return dict(
            label=Label.from_category(id.split("/", 1)[0], categories=self.categories),
            path=path,
            image=EncodedImage.from_file(buffer),
        )

    def _image_key(self, data: Tuple[str, Any]) -> str:
        path = Path(data[0])
        return path.relative_to(path.parents[1]).with_suffix("").as_posix()

    def _make_datapipe(
        self,
        resource_dps: List[IterDataPipe],
        *,
        config: DatasetConfig,
    ) -> IterDataPipe[Dict[str, Any]]:
        archive_dp = resource_dps[0]
        images_dp, split_dp = Demultiplexer(
            archive_dp, 2, self._classify_archive, drop_none=True, buffer_size=INFINITE_BUFFER_SIZE
        )
        split_dp = Filter(split_dp, path_comparator("name", f"{config.split}.txt"))
        split_dp = LineReader(split_dp, decode=True, return_path=False)
        split_dp = hint_sharding(split_dp)
        split_dp = hint_shuffling(split_dp)

        dp = IterKeyZipper(
            split_dp,
            images_dp,
            key_fn=getitem(),
            ref_key_fn=self._image_key,
            buffer_size=INFINITE_BUFFER_SIZE,
        )

        return Mapper(dp, self._prepare_sample)

    def _generate_categories(self, root: Path) -> List[str]:
        resources = self.resources(self.default_config)
        dp = resources[0].load(root)
        dp = Filter(dp, path_comparator("name", "classes.txt"))
        dp = LineReader(dp, decode=True, return_path=False)
        return list(dp)
