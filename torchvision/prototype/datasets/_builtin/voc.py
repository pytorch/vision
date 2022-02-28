import functools
import pathlib
from typing import Any, Dict, List, Optional, Tuple, BinaryIO, cast, Callable
from xml.etree import ElementTree

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
)
from torchvision.prototype.datasets.utils._internal import (
    path_accessor,
    getitem,
    INFINITE_BUFFER_SIZE,
    path_comparator,
    hint_sharding,
    hint_shuffling,
)
from torchvision.prototype.features import BoundingBox, Label, EncodedImage


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

    def _parse_detection_ann(self, buffer: BinaryIO) -> Dict[str, Any]:
        return cast(Dict[str, Any], VOCDetection.parse_voc_xml(ElementTree.parse(buffer).getroot())["annotation"])

    def _prepare_detection_ann(self, buffer: BinaryIO) -> Dict[str, Any]:
        anns = self._parse_detection_ann(buffer)
        instances = anns["object"]
        return dict(
            bounding_boxes=BoundingBox(
                [
                    [int(instance["bndbox"][part]) for part in ("xmin", "ymin", "xmax", "ymax")]
                    for instance in instances
                ],
                format="xyxy",
                image_size=cast(Tuple[int, int], tuple(int(anns["size"][dim]) for dim in ("height", "width"))),
            ),
            labels=Label(
                [self.categories.index(instance["name"]) for instance in instances], categories=self.categories
            ),
        )

    def _prepare_segmentation_ann(self, buffer: BinaryIO) -> Dict[str, Any]:
        return dict(segmentation=EncodedImage.from_file(buffer))

    def _prepare_sample(
        self,
        data: Tuple[Tuple[Tuple[str, str], Tuple[str, BinaryIO]], Tuple[str, BinaryIO]],
        *,
        prepare_ann_fn: Callable[[BinaryIO], Dict[str, Any]],
    ) -> Dict[str, Any]:
        split_and_image_data, ann_data = data
        _, image_data = split_and_image_data
        image_path, image_buffer = image_data
        ann_path, ann_buffer = ann_data

        return dict(
            prepare_ann_fn(ann_buffer),
            image_path=image_path,
            image=EncodedImage.from_file(image_buffer),
            ann_path=ann_path,
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
        return Mapper(
            dp,
            functools.partial(
                self._prepare_sample,
                prepare_ann_fn=self._prepare_detection_ann
                if config.task == "detection"
                else self._prepare_segmentation_ann,
            ),
        )

    def _filter_detection_anns(self, data: Tuple[str, Any], *, config: DatasetConfig) -> bool:
        return self._classify_archive(data, config=config) == 2

    def _generate_categories(self, root: pathlib.Path) -> List[str]:
        config = self.info.make_config(task="detection")

        resource = self.resources(config)[0]
        dp = resource.load(pathlib.Path(root) / self.name)
        dp = Filter(dp, self._filter_detection_anns, fn_kwargs=dict(config=config))
        dp = Mapper(dp, self._parse_detection_ann, input_col=1)

        return sorted({instance["name"] for _, anns in dp for instance in anns["object"]})
