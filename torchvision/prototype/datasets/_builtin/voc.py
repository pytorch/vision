import enum
import functools
import pathlib
from typing import Any, BinaryIO, cast, Dict, List, Optional, Tuple, Union
from xml.etree import ElementTree

from torchdata.datapipes.iter import Demultiplexer, Filter, IterDataPipe, IterKeyZipper, LineReader, Mapper
from torchvision.datasets import VOCDetection
from torchvision.prototype.datasets.utils import Dataset, EncodedImage, HttpResource, OnlineResource
from torchvision.prototype.datasets.utils._internal import (
    getitem,
    hint_sharding,
    hint_shuffling,
    INFINITE_BUFFER_SIZE,
    path_accessor,
    path_comparator,
    read_categories_file,
)
from torchvision.prototype.tv_tensors import Label
from torchvision.tv_tensors import BoundingBoxes

from .._api import register_dataset, register_info

NAME = "voc"


@register_info(NAME)
def _info() -> Dict[str, Any]:
    return dict(categories=read_categories_file(NAME))


@register_dataset(NAME)
class VOC(Dataset):
    """
    - **homepage**: http://host.robots.ox.ac.uk/pascal/VOC/
    """

    def __init__(
        self,
        root: Union[str, pathlib.Path],
        *,
        split: str = "train",
        year: str = "2012",
        task: str = "detection",
        skip_integrity_check: bool = False,
    ) -> None:
        self._year = self._verify_str_arg(year, "year", ("2007", "2008", "2009", "2010", "2011", "2012"))
        if split == "test" and year != "2007":
            raise ValueError("`split='test'` is only available for `year='2007'`")
        else:
            self._split = self._verify_str_arg(split, "split", ("train", "val", "trainval", "test"))
        self._task = self._verify_str_arg(task, "task", ("detection", "segmentation"))

        self._anns_folder = "Annotations" if task == "detection" else "SegmentationClass"
        self._split_folder = "Main" if task == "detection" else "Segmentation"

        self._categories = _info()["categories"]

        super().__init__(root, skip_integrity_check=skip_integrity_check)

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

    def _resources(self) -> List[OnlineResource]:
        file_name, sha256 = (self._TEST_ARCHIVES if self._split == "test" else self._TRAIN_VAL_ARCHIVES)[self._year]
        archive = HttpResource(f"http://host.robots.ox.ac.uk/pascal/VOC/voc{self._year}/{file_name}", sha256=sha256)
        return [archive]

    def _is_in_folder(self, data: Tuple[str, Any], *, name: str, depth: int = 1) -> bool:
        path = pathlib.Path(data[0])
        return name in path.parent.parts[-depth:]

    class _Demux(enum.IntEnum):
        SPLIT = 0
        IMAGES = 1
        ANNS = 2

    def _classify_archive(self, data: Tuple[str, Any]) -> Optional[int]:
        if self._is_in_folder(data, name="ImageSets", depth=2):
            return self._Demux.SPLIT
        elif self._is_in_folder(data, name="JPEGImages"):
            return self._Demux.IMAGES
        elif self._is_in_folder(data, name=self._anns_folder):
            return self._Demux.ANNS
        else:
            return None

    def _parse_detection_ann(self, buffer: BinaryIO) -> Dict[str, Any]:
        ann = cast(Dict[str, Any], VOCDetection.parse_voc_xml(ElementTree.parse(buffer).getroot())["annotation"])
        buffer.close()
        return ann

    def _prepare_detection_ann(self, buffer: BinaryIO) -> Dict[str, Any]:
        anns = self._parse_detection_ann(buffer)
        instances = anns["object"]
        return dict(
            bounding_boxes=BoundingBoxes(
                [
                    [int(instance["bndbox"][part]) for part in ("xmin", "ymin", "xmax", "ymax")]
                    for instance in instances
                ],
                format="xyxy",
                spatial_size=cast(Tuple[int, int], tuple(int(anns["size"][dim]) for dim in ("height", "width"))),
            ),
            labels=Label(
                [self._categories.index(instance["name"]) for instance in instances], categories=self._categories
            ),
        )

    def _prepare_segmentation_ann(self, buffer: BinaryIO) -> Dict[str, Any]:
        return dict(segmentation=EncodedImage.from_file(buffer))

    def _prepare_sample(
        self,
        data: Tuple[Tuple[Tuple[str, str], Tuple[str, BinaryIO]], Tuple[str, BinaryIO]],
    ) -> Dict[str, Any]:
        split_and_image_data, ann_data = data
        _, image_data = split_and_image_data
        image_path, image_buffer = image_data
        ann_path, ann_buffer = ann_data

        return dict(
            (self._prepare_detection_ann if self._task == "detection" else self._prepare_segmentation_ann)(ann_buffer),
            image_path=image_path,
            image=EncodedImage.from_file(image_buffer),
            ann_path=ann_path,
        )

    def _datapipe(self, resource_dps: List[IterDataPipe]) -> IterDataPipe[Dict[str, Any]]:
        archive_dp = resource_dps[0]
        split_dp, images_dp, anns_dp = Demultiplexer(
            archive_dp,
            3,
            self._classify_archive,
            drop_none=True,
            buffer_size=INFINITE_BUFFER_SIZE,
        )

        split_dp = Filter(split_dp, functools.partial(self._is_in_folder, name=self._split_folder))
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
            ("train", "2007", "detection"): 2_501,
            ("train", "2007", "segmentation"): 209,
            ("train", "2008", "detection"): 2_111,
            ("train", "2008", "segmentation"): 511,
            ("train", "2009", "detection"): 3_473,
            ("train", "2009", "segmentation"): 749,
            ("train", "2010", "detection"): 4_998,
            ("train", "2010", "segmentation"): 964,
            ("train", "2011", "detection"): 5_717,
            ("train", "2011", "segmentation"): 1_112,
            ("train", "2012", "detection"): 5_717,
            ("train", "2012", "segmentation"): 1_464,
            ("val", "2007", "detection"): 2_510,
            ("val", "2007", "segmentation"): 213,
            ("val", "2008", "detection"): 2_221,
            ("val", "2008", "segmentation"): 512,
            ("val", "2009", "detection"): 3_581,
            ("val", "2009", "segmentation"): 750,
            ("val", "2010", "detection"): 5_105,
            ("val", "2010", "segmentation"): 964,
            ("val", "2011", "detection"): 5_823,
            ("val", "2011", "segmentation"): 1_111,
            ("val", "2012", "detection"): 5_823,
            ("val", "2012", "segmentation"): 1_449,
            ("trainval", "2007", "detection"): 5_011,
            ("trainval", "2007", "segmentation"): 422,
            ("trainval", "2008", "detection"): 4_332,
            ("trainval", "2008", "segmentation"): 1_023,
            ("trainval", "2009", "detection"): 7_054,
            ("trainval", "2009", "segmentation"): 1_499,
            ("trainval", "2010", "detection"): 10_103,
            ("trainval", "2010", "segmentation"): 1_928,
            ("trainval", "2011", "detection"): 11_540,
            ("trainval", "2011", "segmentation"): 2_223,
            ("trainval", "2012", "detection"): 11_540,
            ("trainval", "2012", "segmentation"): 2_913,
            ("test", "2007", "detection"): 4_952,
            ("test", "2007", "segmentation"): 210,
        }[(self._split, self._year, self._task)]

    def _filter_anns(self, data: Tuple[str, Any]) -> bool:
        return self._classify_archive(data) == self._Demux.ANNS

    def _generate_categories(self) -> List[str]:
        self._task = "detection"
        resources = self._resources()

        archive_dp = resources[0].load(self._root)
        dp = Filter(archive_dp, self._filter_anns)
        dp = Mapper(dp, self._parse_detection_ann, input_col=1)

        categories = sorted({instance["name"] for _, anns in dp for instance in anns["object"]})
        # We add a background category to be used during segmentation
        categories.insert(0, "__background__")

        return categories
