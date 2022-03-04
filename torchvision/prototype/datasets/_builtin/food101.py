import pathlib
from pathlib import Path
from typing import Any, Tuple, List, Dict, cast

import numpy as np
from torchdata.datapipes.iter import IterDataPipe, Filter, Mapper
from torchvision.prototype.datasets.utils import Dataset, DatasetInfo, DatasetConfig, HttpResource, OnlineResource
from torchvision.prototype.datasets.utils._internal import hint_shuffling, hint_sharding, path_comparator
from torchvision.prototype.features import Label, EncodedImage


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
    # TODO: The sha has changed, is this correct?
    # _SHA256 = "85eeb15f3717b99a5da872d97d918f87"
    _SHA256 = "d97d15e438b7f4498f96086a4f7e2fa42a32f2712e87d3295441b2b6314053a4"
    _META_FILE_NAME = "meta"

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
            )
        ]

    def _prepare_sample(self, data: Tuple[np.ndarray, int]) -> Dict[str, Any]:
        image_path, image_buffer = data
        # Extract the category for the image path, it is the folder before the image name.
        category = Path(image_path).parent.name
        categories = [category]
        return dict(
            image=EncodedImage.from_file(image_buffer),
            # TODO: Should the categories be prepared as well? Most likely, yes.
            # For now hardcoded list.
            label=Label.from_category(category, categories=categories),
        )

    # TODO: Doesn't seem correct, investigate...
    def _is_data_file(self, data: Tuple[str, Any], *, split: str) -> bool:
        path = pathlib.Path(data[0])
        return path.name == split

    # TODO: Doesn't yet work, fix.
    def _generate_categories(self, root: pathlib.Path) -> List[str]:
        resources = self.resources(self.default_config)
        dp = resources[0].load(root)
        dp = Filter(dp, path_comparator("name", self._META_FILE_NAME))
        return cast(List[str], next(iter(dp)))

    def _make_datapipe(
        self,
        resource_dps: List[IterDataPipe],
        *,
        config: DatasetConfig,
    ) -> IterDataPipe[Dict[str, Any]]:
        # TODO: Finish the datapipe extraction.
        # Since there is only one resource in the resources method.
        dp = resource_dps[0]
        # dp = TarArchiveReader(dp)
        # TODO: What about train/val/test splits?
        # dp = Filter(dp, functools.partial(self._is_data_file, split=config.split))
        dp = hint_sharding(dp)
        dp = hint_shuffling(dp)
        return Mapper(dp, self._prepare_sample)


# TODO: Translate this into the dp correspondant part.
"""
with open(self._meta_folder / f"{split}.json") as f:
    metadata = json.loads(f.read())

self.classes = sorted(metadata.keys())
self.class_to_idx = dict(zip(self.classes, range(len(self.classes))))

for class_label, im_rel_paths in metadata.items():
    self._labels += [self.class_to_idx[class_label]] * len(im_rel_paths)
    self._image_files += [
        self._images_folder.joinpath(*f"{im_rel_path}.jpg".split("/")) for im_rel_path in im_rel_paths
    ]
"""
