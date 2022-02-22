import collections
import io
import pathlib
from typing import Any, Dict, List, Tuple

import numpy as np
from PIL import Image
from torchdata.datapipes.iter import (
    IterDataPipe,
    Mapper,
)
from torchvision.prototype.datasets.utils import (
    Dataset,
    DatasetConfig,
    DatasetInfo,
    HttpResource,
    OnlineResource,
    DatasetType,
)
from torchvision.prototype.datasets.utils._internal import hint_sharding, hint_shuffling
from torchvision.prototype.features import Label


class Omniglot(Dataset):
    def _make_info(self) -> DatasetInfo:
        return DatasetInfo(
            "omniglot", type=DatasetType.RAW, categories=None, homepage="https://github.com/brendenlake/omniglot"
        )

    _CHECKSUMS = {
        "images_background": "ad41ab679c8b5d90b271ef46be6987cc81211774a695c29dcc5367b2b26ee640",
        "images_evaluation": "1f61a8f3366785b057fc117d9228e78a16e3d976c8953b2a10fcc74cf0609cee",
    }

    def resources(self, config: DatasetConfig) -> List[OnlineResource]:
        data = HttpResource(
            f"https://raw.githubusercontent.com/brendenlake/omniglot/master/python/{config.split}.zip",
            sha256=self._CHECKSUMS[config.split],
        )

        return [data]

    def _get_alphabets_and_characters(self, dp: IterDataPipe):
        categories = collections.OrderedDict()
        for path, _ in dp:
            character = pathlib.Path(path).parents[0].as_posix().split("/")[-1]
            alphabet = pathlib.Path(path).parents[1].as_posix().split("/")[-1]
            if alphabet not in categories.keys():
                categories[alphabet] = [character]
            else:
                categories[alphabet].append(character)
        self._alphabets = list(categories.keys())
        self._characters = [categories[alphabet] for alphabet in self._alphabets]
        return self._alphabets, self._characters

    def _read_images_and_labels(self, data: Tuple[str, io.IOBase]) -> Tuple[Image.Image, int]:
        image_path, image_file = data
        alphabet_class = pathlib.Path(image_path).parents[1].as_posix().split("/")[-1]
        image = Image.open(image_file, mode="r").convert("L")
        idx = self._alphabets.index(alphabet_class)
        return image, idx

    def _collate_and_decode(self, data: Tuple[np.ndarray, int]) -> Dict[str, Any]:
        image, image_label = data

        label = Label(image_label, category=self._alphabets[image_label])
        return dict(image=image, label=label)

    def _make_datapipe(
        self, resource_dps: List[IterDataPipe], *, config: DatasetConfig
    ) -> IterDataPipe[Dict[str, Any]]:
        dp = resource_dps[0]
        self._get_alphabets_and_characters(dp)

        dp = Mapper(dp, self._read_images_and_labels)
        dp = hint_sharding(dp)
        dp = hint_shuffling(dp)

        return Mapper(dp, self._collate_and_decode)

    def _generate_categories(self, root: pathlib.Path) -> List[str]:
        return self._alphabets
