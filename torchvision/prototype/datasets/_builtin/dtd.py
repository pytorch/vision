import enum
import functools
import io
import pathlib
from typing import Any, Callable, Dict, List, Optional, Tuple

import torch
from torchdata.datapipes.iter import (
    IterDataPipe,
    Mapper,
    Shuffler,
    Filter,
    IterKeyZipper,
    Demultiplexer,
    LineReader,
    CSVParser,
)
from torchvision.prototype.datasets.utils import (
    Dataset,
    DatasetConfig,
    DatasetInfo,
    HttpResource,
    OnlineResource,
    DatasetType,
)
from torchvision.prototype.datasets.utils._internal import (
    INFINITE_BUFFER_SIZE,
    hint_sharding,
    path_comparator,
    getitem,
)
from torchvision.prototype.features import Label


class DTDDemux(enum.IntEnum):
    SPLIT = 0
    JOINT_CATEGORIES = 1
    IMAGES = 2


class DTD(Dataset):
    def _make_info(self) -> DatasetInfo:
        return DatasetInfo(
            "dtd",
            type=DatasetType.IMAGE,
            homepage="https://www.robots.ox.ac.uk/~vgg/data/dtd/",
            valid_options=dict(
                split=("train", "test", "val"),
                fold=tuple(str(fold) for fold in range(1, 11)),
            ),
        )

    def resources(self, config: DatasetConfig) -> List[OnlineResource]:
        archive = HttpResource(
            "https://www.robots.ox.ac.uk/~vgg/data/dtd/download/dtd-r1.0.1.tar.gz",
            sha256="e42855a52a4950a3b59612834602aa253914755c95b0cff9ead6d07395f8e205",
            decompress=True,
        )
        return [archive]

    def _classify_archive(self, data: Tuple[str, Any]) -> Optional[int]:
        path = pathlib.Path(data[0])
        if path.parent.name == "labels":
            if path.name == "labels_joint_anno.txt":
                return DTDDemux.JOINT_CATEGORIES

            return DTDDemux.SPLIT
        elif path.parents[1].name == "images":
            return DTDDemux.IMAGES
        else:
            return None

    def _image_key_fn(self, data: Tuple[str, Any]) -> str:
        path = pathlib.Path(data[0])
        return str(path.relative_to(path.parents[1]))

    def _collate_and_decode_sample(
        self,
        data: Tuple[Tuple[str, List[str]], Tuple[str, io.IOBase]],
        *,
        decoder: Optional[Callable[[io.IOBase], torch.Tensor]],
    ) -> Dict[str, Any]:
        (_, joint_categories_data), image_data = data
        _, *joint_categories = joint_categories_data
        path, buffer = image_data

        category = pathlib.Path(path).parent.name

        return dict(
            joint_categories={category for category in joint_categories if category},
            label=Label(self.info.categories.index(category), category=category),
            path=path,
            image=decoder(buffer) if decoder else buffer,
        )

    def _make_datapipe(
        self,
        resource_dps: List[IterDataPipe],
        *,
        config: DatasetConfig,
        decoder: Optional[Callable[[io.IOBase], torch.Tensor]],
    ) -> IterDataPipe[Dict[str, Any]]:
        archive_dp = resource_dps[0]

        splits_dp, joint_categories_dp, images_dp = Demultiplexer(
            archive_dp, 3, self._classify_archive, drop_none=True, buffer_size=INFINITE_BUFFER_SIZE
        )

        splits_dp = Filter(splits_dp, path_comparator("name", f"{config.split}{config.fold}.txt"))
        splits_dp = LineReader(splits_dp, decode=True, return_path=False)
        splits_dp = Shuffler(splits_dp, buffer_size=INFINITE_BUFFER_SIZE)
        splits_dp = hint_sharding(splits_dp)

        joint_categories_dp = CSVParser(joint_categories_dp, delimiter=" ")

        dp = IterKeyZipper(
            splits_dp,
            joint_categories_dp,
            key_fn=getitem(),
            ref_key_fn=getitem(0),
            buffer_size=INFINITE_BUFFER_SIZE,
        )
        dp = IterKeyZipper(
            dp,
            images_dp,
            key_fn=getitem(0),
            ref_key_fn=self._image_key_fn,
            buffer_size=INFINITE_BUFFER_SIZE,
        )
        return Mapper(dp, functools.partial(self._collate_and_decode_sample, decoder=decoder))

    def _filter_images(self, data: Tuple[str, Any]) -> bool:
        return self._classify_archive(data) == DTDDemux.IMAGES

    def _generate_categories(self, root: pathlib.Path) -> List[str]:
        resources = self.resources(self.default_config)

        dp = resources[0].load(root)
        dp = Filter(dp, self._filter_images)

        return sorted({pathlib.Path(path).parent.name for path, _ in dp})
