import pathlib
import re
from typing import Any, Dict, List, Optional, Tuple, cast, BinaryIO

import numpy as np
from torchdata.datapipes.iter import (
    IterDataPipe,
    Mapper,
    Demultiplexer,
    Filter,
    IterKeyZipper,
    LineReader,
)
from torchvision.prototype.datasets.utils import (
    Dataset,
    DatasetConfig,
    DatasetInfo,
    HttpResource,
    OnlineResource,
)
from torchvision.prototype.datasets.utils._internal import (
    INFINITE_BUFFER_SIZE,
    read_mat,
    getitem,
    path_accessor,
    path_comparator,
    hint_sharding,
    hint_shuffling,
)
from torchvision.prototype.features import _Feature, EncodedImage


class SBD(Dataset):
    def _make_info(self) -> DatasetInfo:
        return DatasetInfo(
            "sbd",
            dependencies=("scipy",),
            homepage="http://home.bharathh.info/pubs/codes/SBD/download.html",
            valid_options=dict(
                split=("train", "val", "train_noval"),
            ),
        )

    def resources(self, config: DatasetConfig) -> List[OnlineResource]:
        archive = HttpResource(
            "https://www2.eecs.berkeley.edu/Research/Projects/CS/vision/grouping/semantic_contours/benchmark.tgz",
            sha256="6a5a2918d5c73ce032fdeba876574d150d9d04113ab87540a1304cbcc715be53",
        )
        extra_split = HttpResource(
            "http://home.bharathh.info/pubs/codes/SBD/train_noval.txt",
            sha256="0b2068f7a359d2907431803e1cd63bf6162da37d7d503b589d3b08c6fd0c2432",
        )
        return [archive, extra_split]

    def _classify_archive(self, data: Tuple[str, Any]) -> Optional[int]:
        path = pathlib.Path(data[0])
        parent, grandparent, *_ = path.parents

        if parent.name == "dataset":
            return 0
        elif grandparent.name == "dataset":
            if parent.name == "img":
                return 1
            elif parent.name == "cls":
                return 2
            else:
                return None
        else:
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
            boundaries=_Feature(np.stack([raw_boundary.toarray() for raw_boundary in anns["Boundaries"].item()])),
            segmentation=_Feature(anns["Segmentation"].item()),
        )

    def _make_datapipe(
        self,
        resource_dps: List[IterDataPipe],
        *,
        config: DatasetConfig,
    ) -> IterDataPipe[Dict[str, Any]]:
        archive_dp, extra_split_dp = resource_dps

        archive_dp = resource_dps[0]
        split_dp, images_dp, anns_dp = Demultiplexer(
            archive_dp,
            3,
            self._classify_archive,
            buffer_size=INFINITE_BUFFER_SIZE,
            drop_none=True,
        )
        if config.split == "train_noval":
            split_dp = extra_split_dp

        split_dp = Filter(split_dp, path_comparator("name", f"{config.split}.txt"))
        split_dp = LineReader(split_dp, decode=True)
        split_dp = hint_sharding(split_dp)
        split_dp = hint_shuffling(split_dp)

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

    def _generate_categories(self, root: pathlib.Path) -> Tuple[str, ...]:
        resources = self.resources(self.default_config)

        dp = resources[0].load(root)
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
