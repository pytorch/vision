import functools
import io
import pathlib
from typing import Any, Callable, Dict, List, Optional, Tuple
from xml.etree import ElementTree

import torch
from torchdata.datapipes.iter import (
    IterDataPipe,
    Mapper,
    Filter,
    Demultiplexer,
    IterKeyZipper,
    LineReader,
)
from torchvision.datasets import VOCDetection
from torchvision.prototype.datasets.utils import (
    Dataset,
    DatasetConfig,
    DatasetInfo,
    HttpResource,
    OnlineResource,
    DatasetType,
)
from torchvision.prototype.datasets.utils._internal import (
    path_accessor,
    getitem,
    INFINITE_BUFFER_SIZE,
    path_comparator,
    hint_sharding,
    hint_shuffling,
)
from torchvision.prototype.features import BoundingBox


class VOCDatasetInfo(DatasetInfo):
    def __init__(self, *args: Any, **kwargs: Any):
        super().__init__(*args, **kwargs)
        self._configs = tuple(config for config in self._configs if config.split != "test" or config.year == "2007")

    def make_config(self, **options: Any) -> DatasetConfig:
        config = super().make_config(**options)
        if config.split == "test" and config.year != "2007":
            raise ValueError("`split='test'` is only available for `year='2007'`")

        return config


class VOC(Dataset):
    def _make_info(self) -> DatasetInfo:
        return VOCDatasetInfo(
            "voc",
            type=DatasetType.IMAGE,
            homepage="http://host.robots.ox.ac.uk/pascal/VOC/",
            valid_options=dict(
                split=("train", "val", "trainval", "test"),
                year=("2012", "2007", "2008", "2009", "2010", "2011"),
                task=("detection", "segmentation"),
            ),
        )

    _TRAIN_VAL_ARCHIVES = {
        "2007": ("VOCtrainval_06-Nov-2007.tar", "7d8cd951101b0957ddfd7a530bdc8a94f06121cfc1e511bb5937e973020c7508"),
        "2008": ("VOCtrainval_14-Jul-2008.tar", "7f0ca53c1b5a838fbe946965fc106c6e86832183240af5c88e3f6c306318d42e"),
        "2009": ("VOCtrainval_11-May-2009.tar", "11cbe1741fb5bdadbbca3c08e9ec62cd95c14884845527d50847bc2cf57e7fd6"),
        "2010": ("VOCtrainval_03-May-2010.tar", "1af4189cbe44323ab212bff7afbc7d0f55a267cc191eb3aac911037887e5c7d4"),
        "2011": ("VOCtrainval_25-May-2011.tar", "0a7f5f5d154f7290ec65ec3f78b72ef72c6d93ff6d79acd40dc222a9ee5248ba"),
        "2012": ("VOCtrainval_11-May-2012.tar", "e14f763270cf193d0b5f74b169f44157a4b0c6efa708f4dd0ff78ee691763bcb"),
    }
    _TEST_ARCHIVES = {
        "2007": ("VOCtest_06-Nov-2007.tar", "6836888e2e01dca84577a849d339fa4f73e1e4f135d312430c4856b5609b4892")
    }

    def resources(self, config: DatasetConfig) -> List[OnlineResource]:
        file_name, sha256 = (self._TEST_ARCHIVES if config.split == "test" else self._TRAIN_VAL_ARCHIVES)[config.year]
        archive = HttpResource(f"http://host.robots.ox.ac.uk/pascal/VOC/voc{config.year}/{file_name}", sha256=sha256)
        return [archive]

    _ANNS_FOLDER = dict(
        detection="Annotations",
        segmentation="SegmentationClass",
    )
    _SPLIT_FOLDER = dict(
        detection="Main",
        segmentation="Segmentation",
    )

    def _is_in_folder(self, data: Tuple[str, Any], *, name: str, depth: int = 1) -> bool:
        path = pathlib.Path(data[0])
        return name in path.parent.parts[-depth:]

    def _classify_archive(self, data: Tuple[str, Any], *, config: DatasetConfig) -> Optional[int]:
        if self._is_in_folder(data, name="ImageSets", depth=2):
            return 0
        elif self._is_in_folder(data, name="JPEGImages"):
            return 1
        elif self._is_in_folder(data, name=self._ANNS_FOLDER[config.task]):
            return 2
        else:
            return None

    def _decode_detection_ann(self, buffer: io.IOBase) -> torch.Tensor:
        result = VOCDetection.parse_voc_xml(ElementTree.parse(buffer).getroot())  # type: ignore[arg-type]
        objects = result["annotation"]["object"]
        bboxes = [obj["bndbox"] for obj in objects]
        bboxes = [[int(bbox[part]) for part in ("xmin", "ymin", "xmax", "ymax")] for bbox in bboxes]
        return BoundingBox(bboxes)

    def _collate_and_decode_sample(
        self,
        data: Tuple[Tuple[Tuple[str, str], Tuple[str, io.IOBase]], Tuple[str, io.IOBase]],
        *,
        config: DatasetConfig,
        decoder: Optional[Callable[[io.IOBase], torch.Tensor]],
    ) -> Dict[str, Any]:
        split_and_image_data, ann_data = data
        _, image_data = split_and_image_data
        image_path, image_buffer = image_data
        ann_path, ann_buffer = ann_data

        image = decoder(image_buffer) if decoder else image_buffer

        if config.task == "detection":
            ann = self._decode_detection_ann(ann_buffer)
        else:  # config.task == "segmentation":
            ann = decoder(ann_buffer) if decoder else ann_buffer  # type: ignore[assignment]

        return dict(image_path=image_path, image=image, ann_path=ann_path, ann=ann)

    def _make_datapipe(
        self,
        resource_dps: List[IterDataPipe],
        *,
        config: DatasetConfig,
        decoder: Optional[Callable[[io.IOBase], torch.Tensor]],
    ) -> IterDataPipe[Dict[str, Any]]:
        archive_dp = resource_dps[0]
        split_dp, images_dp, anns_dp = Demultiplexer(
            archive_dp,
            3,
            functools.partial(self._classify_archive, config=config),
            drop_none=True,
            buffer_size=INFINITE_BUFFER_SIZE,
        )

        split_dp = Filter(split_dp, functools.partial(self._is_in_folder, name=self._SPLIT_FOLDER[config.task]))
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
        return Mapper(dp, functools.partial(self._collate_and_decode_sample, config=config, decoder=decoder))
