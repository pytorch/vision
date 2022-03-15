import functools
from pathlib import Path
from typing import Any, Tuple, List, Dict, cast, Optional, BinaryIO

from torchdata.datapipes.iter import IterDataPipe, Filter, Mapper, CSVParser, JsonParser, Demultiplexer
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
            # Only train and test splits, no validation.
            valid_options=dict(split=("train", "test")),
        )

    def resources(self, config: DatasetConfig) -> List[OnlineResource]:
        return [
            HttpResource(
                url="http://data.vision.ee.ethz.ch/cvl/food-101.tar.gz",
                sha256="d97d15e438b7f4498f96086a4f7e2fa42a32f2712e87d3295441b2b6314053a4",
                decompress=True,
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

    def _prepare_sample(self, data: Tuple[str, BinaryIO]) -> Dict[str, Any]:
        image_path, image_buffer = data
        category = Path(image_path).parent.name
        return dict(
            image=EncodedImage.from_file(image_buffer),
            label=Label.from_category(category, categories=self.categories),
        )

    def _img_in_metadata(self, data: Tuple[str, Any], *, metadata: Dict[str, List[str]]) -> bool:
        # Check whether or not the file is contained in the list of the splitted metdata.
        path = Path(data[0])
        category = Path(path).parent.name
        if category in metadata:
            img_ids = {e.split("/")[-1] for e in metadata[category]}
            return path.stem in img_ids
        else:
            return False

    def _make_datapipe(
        self,
        resource_dps: List[IterDataPipe],
        *,
        config: DatasetConfig,
    ) -> IterDataPipe[Dict[str, Any]]:
        archive_dp = resource_dps[0]
        images_dp, metadata_dp = Demultiplexer(
            archive_dp, 2, self._classify_archive, drop_none=True, buffer_size=INFINITE_BUFFER_SIZE
        )
        metadata_dp = Filter(metadata_dp, path_comparator("name", f"{config.split}.json"))
        metadata_dp = JsonParser(metadata_dp)
        metadata = next(iter(Mapper(metadata_dp, getitem(1))))
        images_dp = Filter(images_dp, functools.partial(self._img_in_metadata, metadata=metadata))
        images_dp = hint_sharding(images_dp)
        images_dp = hint_shuffling(images_dp)
        return Mapper(images_dp, self._prepare_sample)

    def _generate_categories(self, root: Path) -> List[str]:
        resources = self.resources(self.default_config)
        dp = resources[0].load(root)
        dp = Filter(dp, path_comparator("name", "classes.txt"))
        dp = CSVParser(dp, delimiter=" ")
        categories = [sample[0] for sample in dp]
        return cast(List[str], categories)
