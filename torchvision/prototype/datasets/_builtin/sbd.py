import pathlib
import re
from typing import Any, BinaryIO, cast, Dict, List, Optional, Tuple, Union

import numpy as np
import torch
from torchdata.datapipes.iter import Demultiplexer, Filter, IterDataPipe, IterKeyZipper, LineReader, Mapper
from torchvision.prototype.datasets.utils import Dataset, EncodedImage, HttpResource, OnlineResource
from torchvision.prototype.datasets.utils._internal import (
    getitem,
    hint_sharding,
    hint_shuffling,
    INFINITE_BUFFER_SIZE,
    path_accessor,
    path_comparator,
    read_categories_file,
    read_mat,
)

from .._api import register_dataset, register_info

NAME = "sbd"


@register_info(NAME)
def _info() -> Dict[str, Any]:
    return dict(categories=read_categories_file(NAME))


@register_dataset(NAME)
class SBD(Dataset):
    """
    - **homepage**: http://home.bharathh.info/pubs/codes/SBD/download.html
    - **dependencies**:
        - <scipy `https://scipy.org`>_
    """

    def __init__(
        self,
        root: Union[str, pathlib.Path],
        *,
        split: str = "train",
        skip_integrity_check: bool = False,
    ) -> None:
        self._split = self._verify_str_arg(split, "split", ("train", "val", "train_noval"))

        self._categories = _info()["categories"]

        super().__init__(root, dependencies=("scipy",), skip_integrity_check=skip_integrity_check)

    def _resources(self) -> List[OnlineResource]:
        resources = [
            HttpResource(
                "https://www2.eecs.berkeley.edu/Research/Projects/CS/vision/grouping/semantic_contours/benchmark.tgz",
                sha256="6a5a2918d5c73ce032fdeba876574d150d9d04113ab87540a1304cbcc715be53",
            )
        ]
        if self._split == "train_noval":
            resources.append(
                HttpResource(
                    "http://home.bharathh.info/pubs/codes/SBD/train_noval.txt",
                    sha256="0b2068f7a359d2907431803e1cd63bf6162da37d7d503b589d3b08c6fd0c2432",
                )
            )
        return resources  # type: ignore[return-value]

    def _classify_archive(self, data: Tuple[str, Any]) -> Optional[int]:
        path = pathlib.Path(data[0])
        parent, grandparent, *_ = path.parents

        if grandparent.name == "dataset":
            if parent.name == "img":
                return 0
            elif parent.name == "cls":
                return 1

        if parent.name == "dataset" and self._split != "train_noval":
            return 2

        return None

    def _prepare_sample(self, data: Tuple[Tuple[Any, Tuple[str, BinaryIO]], Tuple[str, BinaryIO]]) -> Dict[str, Any]:
        split_and_image_data, ann_data = data
        _, image_data = split_and_image_data
        image_path, image_buffer = image_data
        ann_path, ann_buffer = ann_data

        anns = read_mat(ann_buffer, squeeze_me=True)["GTcls"]

        return dict(
            image_path=image_path,
            image=EncodedImage.from_file(image_buffer),
            ann_path=ann_path,
            # the boundaries are stored in sparse CSC format, which is not supported by PyTorch
            boundaries=torch.as_tensor(
                np.stack([raw_boundary.toarray() for raw_boundary in anns["Boundaries"].item()])
            ),
            segmentation=torch.as_tensor(anns["Segmentation"].item()),
        )

    def _datapipe(self, resource_dps: List[IterDataPipe]) -> IterDataPipe[Dict[str, Any]]:
        if self._split == "train_noval":
            archive_dp, split_dp = resource_dps
            images_dp, anns_dp = Demultiplexer(
                archive_dp,
                2,
                self._classify_archive,
                buffer_size=INFINITE_BUFFER_SIZE,
                drop_none=True,
            )
        else:
            archive_dp = resource_dps[0]
            images_dp, anns_dp, split_dp = Demultiplexer(
                archive_dp,
                3,
                self._classify_archive,
                buffer_size=INFINITE_BUFFER_SIZE,
                drop_none=True,
            )

        split_dp = Filter(split_dp, path_comparator("name", f"{self._split}.txt"))
        split_dp = LineReader(split_dp, decode=True)
        split_dp = hint_shuffling(split_dp)
        split_dp = hint_sharding(split_dp)

        dp = split_dp
        for level, data_dp in enumerate((images_dp, anns_dp)):
            dp = IterKeyZipper(
                dp,
                data_dp,
                key_fn=getitem(*[0] * level, 1),
                ref_key_fn=path_accessor("stem"),
                buffer_size=INFINITE_BUFFER_SIZE,
            )
        return Mapper(dp, self._prepare_sample)

    def __len__(self) -> int:
        return {
            "train": 8_498,
            "val": 2_857,
            "train_noval": 5_623,
        }[self._split]

    def _generate_categories(self) -> Tuple[str, ...]:
        resources = self._resources()

        dp = resources[0].load(self._root)
        dp = Filter(dp, path_comparator("name", "category_names.m"))
        dp = LineReader(dp)
        dp = Mapper(dp, bytes.decode, input_col=1)
        lines = tuple(zip(*iter(dp)))[1]

        pattern = re.compile(r"\s*'(?P<category>\w+)';\s*%(?P<label>\d+)")
        categories_and_labels = cast(
            List[Tuple[str, ...]],
            [
                pattern.match(line).groups()  # type: ignore[union-attr]
                # the first and last line contain no information
                for line in lines[1:-1]
            ],
        )
        categories_and_labels.sort(key=lambda category_and_label: int(category_and_label[1]))
        categories, _ = zip(*categories_and_labels)

        return categories
