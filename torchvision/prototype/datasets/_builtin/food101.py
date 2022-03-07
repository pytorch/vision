import enum
import functools
import pathlib
from pathlib import Path
from typing import Any, Tuple, List, Dict, cast, Optional

import numpy as np
from torchdata.datapipes.iter import IterDataPipe, Filter, Mapper, CSVParser
from torchvision.prototype.datasets.utils import Dataset, DatasetInfo, DatasetConfig, HttpResource, OnlineResource
from torchvision.prototype.datasets.utils._internal import (
    hint_shuffling,
    hint_sharding,
    path_comparator,
)
from torchvision.prototype.features import Label, EncodedImage


class Food101Demux(enum.IntEnum):
    METADATA = 0
    IMAGES = 1


class Food101(Dataset):
    """`The Food-101 Data Set <https://data.vision.ee.ethz.ch/cvl/datasets_extra/food-101/>`_.

    The Food-101 is a challenging data set of 101 food categories, with 101'000 images.
    For each class, 250 manually reviewed test images are provided as well as 750 training images.
    On purpose, the training images were not cleaned, and thus still contain some amount of noise.
    This comes mostly in the form of intense colors and sometimes wrong labels. All images were
    rescaled to have a maximum side length of 512 pixels.

    TODO: Add more details?

    """

    _URL = "http://data.vision.ee.ethz.ch/cvl/food-101.tar.gz"
    _SHA256 = "d97d15e438b7f4498f96086a4f7e2fa42a32f2712e87d3295441b2b6314053a4"
    _META_FILE_NAME = "classes.txt"

    # TODO: Inspired from dtd.py. Is this useful?
    def _classify_archive(self, data: Tuple[str, Any]) -> Optional[int]:
        path = pathlib.Path(data[0])
        print(path)
        if path.name == "images":
            return Food101Demux.IMAGES
        else:
            return Food101Demux.METADATA

    def _make_info(self) -> DatasetInfo:
        return DatasetInfo(
            "food-101",
            homepage="https://data.vision.ee.ethz.ch/cvl/datasets_extra/food-101",
            # Only train and test splits.
            valid_options=dict(split=("train", "test")),
        )

    def resources(self, config: DatasetConfig) -> List[OnlineResource]:
        return [
            HttpResource(
                url=self._URL,
                sha256=self._SHA256,
                decompress=True,
            )
        ]

    def _prepare_sample(self, data: Tuple[np.ndarray, int]) -> Dict[str, Any]:
        image_path, image_buffer = data
        # Extract the category for the image path, it is the folder before the image name.
        category = Path(image_path).parent.name
        # TODO: Extract the list of all categories. Where to do this?
        return dict(
            image=EncodedImage.from_file(image_buffer),
            # TODO: Should the categories be prepared as well? Most likely, yes.
            # For now hardcoded list.
            label=Label.from_category(category, categories=self.categories),
        )

    # TODO: Doesn't seem correct, investigate...
    def _is_data_file(self, data: Tuple[str, Any], *, split: str) -> bool:
        # We need to use the train.json and test.json to get the correct
        # mapping.
        path = pathlib.Path(data[0])
        print(path)
        # Need to compare to the actual list of train and test splits.
        # return path.name == split
        return True

    def _generate_categories(self, root: pathlib.Path) -> List[str]:
        resources = self.resources(self.default_config)
        dp = resources[0].load(root)
        dp = Filter(dp, path_comparator("name", self._META_FILE_NAME))
        # TODO: Probably not the best way to parse, but it works for now. Improve.
        dp = CSVParser(dp, delimiter=" ")
        categories = []
        for sample in dp:
            categories.append(sample[0])
        return cast(List[str], categories)

    def _make_datapipe(
        self,
        resource_dps: List[IterDataPipe],
        *,
        config: DatasetConfig,
    ) -> IterDataPipe[Dict[str, Any]]:
        dp = resource_dps[0]

        # TODO: Could this be useful? Maybe for easily extracting metadata information for the
        # train and test splits.
        # metadata_dp, images_dp = Demultiplexer(
        #     archive_dp, 2, self._classify_archive, drop_none=True, buffer_size=INFINITE_BUFFER_SIZE
        # )

        # Train/Test filtering based on the meta file.
        # TODO: Finish this.
        dp = Filter(dp, functools.partial(self._is_data_file, split=config.split))
        dp = hint_sharding(dp)
        dp = hint_shuffling(dp)
        return Mapper(dp, self._prepare_sample)
