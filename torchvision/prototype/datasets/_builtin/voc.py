import functools
import io
import pathlib
from typing import Any, Callable, Dict, List, Optional, Tuple
from xml.etree import ElementTree

import torch
from torchdata.datapipes.iter import (
    IterDataPipe,
    Mapper,
    Shuffler,
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
)

HERE = pathlib.Path(__file__).parent


class VOC(Dataset):
    def _make_info(self) -> DatasetInfo:
        return DatasetInfo(
            "voc",
            type=DatasetType.IMAGE,
            homepage="http://host.robots.ox.ac.uk/pascal/VOC/",
            valid_options=dict(
                split=("train", "val", "test"),
                year=("2012",),
                task=("detection", "segmentation"),
            ),
        )

    def resources(self, config: DatasetConfig) -> List[OnlineResource]:
        if config.year == "2012":
            if config.split == "train":
                archive = HttpResource(
                    "http://host.robots.ox.ac.uk/pascal/VOC/voc2012/VOCtrainval_11-May-2012.tar",
                    sha256="e14f763270cf193d0b5f74b169f44157a4b0c6efa708f4dd0ff78ee691763bcb",
                )
            else:
                raise RuntimeError("FIXME")
        else:
            raise RuntimeError("FIXME")
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
        return torch.tensor(bboxes)

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

        split_dp = Filter(split_dp, self._is_in_folder, fn_kwargs=dict(name=self._SPLIT_FOLDER[config.task]))
        split_dp = Filter(split_dp, path_comparator("name", f"{config.split}.txt"))
        split_dp = LineReader(split_dp, decode=True)
        split_dp = Shuffler(split_dp, buffer_size=INFINITE_BUFFER_SIZE)

        dp = split_dp
        for level, data_dp in enumerate((images_dp, anns_dp)):
            dp = IterKeyZipper(
                dp,
                data_dp,
                key_fn=getitem(*[0] * level, 1),
                ref_key_fn=path_accessor("stem"),
                buffer_size=INFINITE_BUFFER_SIZE,
            )
        return Mapper(dp, self._collate_and_decode_sample, fn_kwargs=dict(config=config, decoder=decoder))
