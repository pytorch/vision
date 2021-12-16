import functools
import pathlib
from typing import Any, Dict, List, Optional, Tuple, BinaryIO, cast
from xml.etree import ElementTree

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
    DecodeableImageStreamWrapper,
    DecodeableStreamWrapper,
)
from torchvision.prototype.datasets.utils._internal import (
    path_accessor,
    getitem,
    INFINITE_BUFFER_SIZE,
    path_comparator,
)
from torchvision.prototype.features import BoundingBox, Label


class VOC(Dataset):
    def _make_info(self) -> DatasetInfo:
        return DatasetInfo(
            "voc",
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

    def _parse_detection_ann(self, buffer: BinaryIO) -> Dict[str, Any]:
        return cast(Dict[str, Any], VOCDetection.parse_voc_xml(ElementTree.parse(buffer).getroot())["annotation"])

    def _decode_detection_ann(self, buffer: BinaryIO) -> Dict[str, Any]:
        anns = self._parse_detection_ann(buffer)
        instances = anns["object"]
        return dict(
            bounding_boxes=BoundingBox(
                [
                    [int(instance["bndbox"][part]) for part in ("xmin", "ymin", "xmax", "ymax")]
                    for instance in instances
                ],
                format="xyxy",
                image_size=tuple(int(anns["size"][dim]) for dim in ("height", "width")),
            ),
            labels=[
                Label(self.info.categories.index(instance["name"]), category=instance["name"]) for instance in instances
            ],
        )

    def _prepare_sample(
        self,
        data: Tuple[Tuple[Tuple[str, str], Tuple[str, BinaryIO]], Tuple[str, BinaryIO]],
        *,
        config: DatasetConfig,
    ) -> Dict[str, Any]:
        split_and_image_data, ann_data = data
        _, image_data = split_and_image_data
        image_path, image_buffer = image_data
        ann_path, ann_buffer = ann_data

        return dict(
            image_path=image_path,
            image=DecodeableImageStreamWrapper(image_buffer),
            ann_path=ann_path,
            ann=DecodeableStreamWrapper(ann_buffer, self._decode_detection_ann)
            if config.task == "detection"
            else DecodeableImageStreamWrapper(ann_buffer),
        )

    def _make_datapipe(
        self,
        resource_dps: List[IterDataPipe],
        *,
        config: DatasetConfig,
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
        return Mapper(dp, self._prepare_sample, fn_kwargs=dict(config=config))

    def _filter_detection_anns(self, data: Tuple[str, Any], *, config: DatasetConfig) -> bool:
        return self._classify_archive(data, config=config) == 2

    def _generate_categories(self, root: pathlib.Path) -> List[str]:
        config = self.info.make_config(task="detection")

        resource = self.resources(config)[0]
        dp = resource.load(pathlib.Path(root) / self.name)
        dp = Filter(dp, self._filter_detection_anns, fn_kwargs=dict(config=config))
        dp = Mapper(dp, self._parse_detection_ann, input_col=1)

        return sorted({instance["name"] for _, anns in dp for instance in anns["object"]})
