import csv
import functools
import pathlib
from typing import Any, BinaryIO, Callable, Dict, List, Optional, Tuple, Union

import torch
from torchdata.datapipes.iter import (
    CSVDictParser,
    CSVParser,
    Demultiplexer,
    Filter,
    IterDataPipe,
    IterKeyZipper,
    LineReader,
    Mapper,
)
from torchdata.datapipes.map import IterToMapConverter
from torchvision.prototype.datasets.utils import Dataset, EncodedImage, GDriveResource, OnlineResource
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
from torchvision.prototype.tv_tensors import Label
from torchvision.tv_tensors import BoundingBoxes

from .._api import register_dataset, register_info

csv.register_dialect("cub200", delimiter=" ")


NAME = "cub200"


@register_info(NAME)
def _info() -> Dict[str, Any]:
    return dict(categories=read_categories_file(NAME))


@register_dataset(NAME)
class CUB200(Dataset):
    """
    - **homepage**: http://www.vision.caltech.edu/visipedia/CUB-200.html
    """

    def __init__(
        self,
        root: Union[str, pathlib.Path],
        *,
        split: str = "train",
        year: str = "2011",
        skip_integrity_check: bool = False,
    ) -> None:
        self._split = self._verify_str_arg(split, "split", ("train", "test"))
        self._year = self._verify_str_arg(year, "year", ("2010", "2011"))

        self._categories = _info()["categories"]

        super().__init__(
            root,
            # TODO: this will only be available after https://github.com/pytorch/vision/pull/5473
            # dependencies=("scipy",),
            skip_integrity_check=skip_integrity_check,
        )

    def _resources(self) -> List[OnlineResource]:
        if self._year == "2011":
            archive = GDriveResource(
                "1hbzc_P1FuxMkcabkgn9ZKinBwW683j45",
                file_name="CUB_200_2011.tgz",
                sha256="0c685df5597a8b24909f6a7c9db6d11e008733779a671760afef78feb49bf081",
                preprocess="decompress",
            )
            segmentations = GDriveResource(
                "1EamOKGLoTuZdtcVYbHMWNpkn3iAVj8TP",
                file_name="segmentations.tgz",
                sha256="dc77f6cffea0cbe2e41d4201115c8f29a6320ecb04fffd2444f51b8066e4b84f",
                preprocess="decompress",
            )
            return [archive, segmentations]
        else:  # self._year == "2010"
            split = GDriveResource(
                "1vZuZPqha0JjmwkdaS_XtYryE3Jf5Q1AC",
                file_name="lists.tgz",
                sha256="aeacbd5e3539ae84ea726e8a266a9a119c18f055cd80f3836d5eb4500b005428",
                preprocess="decompress",
            )
            images = GDriveResource(
                "1GDr1OkoXdhaXWGA8S3MAq3a522Tak-nx",
                file_name="images.tgz",
                sha256="2a6d2246bbb9778ca03aa94e2e683ccb4f8821a36b7f235c0822e659d60a803e",
                preprocess="decompress",
            )
            anns = GDriveResource(
                "16NsbTpMs5L6hT4hUJAmpW2u7wH326WTR",
                file_name="annotations.tgz",
                sha256="c17b7841c21a66aa44ba8fe92369cc95dfc998946081828b1d7b8a4b716805c1",
                preprocess="decompress",
            )
            return [split, images, anns]

    def _2011_classify_archive(self, data: Tuple[str, Any]) -> Optional[int]:
        path = pathlib.Path(data[0])
        if path.parents[1].name == "images":
            return 0
        elif path.name == "train_test_split.txt":
            return 1
        elif path.name == "images.txt":
            return 2
        elif path.name == "bounding_boxes.txt":
            return 3
        else:
            return None

    def _2011_extract_file_name(self, rel_posix_path: str) -> str:
        return rel_posix_path.rsplit("/", maxsplit=1)[1]

    def _2011_filter_split(self, row: List[str]) -> bool:
        _, split_id = row
        return {
            "0": "test",
            "1": "train",
        }[split_id] == self._split

    def _2011_segmentation_key(self, data: Tuple[str, Any]) -> str:
        path = pathlib.Path(data[0])
        return path.with_suffix(".jpg").name

    def _2011_prepare_ann(
        self, data: Tuple[str, Tuple[List[str], Tuple[str, BinaryIO]]], spatial_size: Tuple[int, int]
    ) -> Dict[str, Any]:
        _, (bounding_boxes_data, segmentation_data) = data
        segmentation_path, segmentation_buffer = segmentation_data
        return dict(
            bounding_boxes=BoundingBoxes(
                [float(part) for part in bounding_boxes_data[1:]], format="xywh", spatial_size=spatial_size
            ),
            segmentation_path=segmentation_path,
            segmentation=EncodedImage.from_file(segmentation_buffer),
        )

    def _2010_split_key(self, data: str) -> str:
        return data.rsplit("/", maxsplit=1)[1]

    def _2010_anns_key(self, data: Tuple[str, BinaryIO]) -> Tuple[str, Tuple[str, BinaryIO]]:
        path = pathlib.Path(data[0])
        return path.with_suffix(".jpg").name, data

    def _2010_prepare_ann(
        self, data: Tuple[str, Tuple[str, BinaryIO]], spatial_size: Tuple[int, int]
    ) -> Dict[str, Any]:
        _, (path, buffer) = data
        content = read_mat(buffer)
        return dict(
            ann_path=path,
            bounding_boxes=BoundingBoxes(
                [int(content["bbox"][coord]) for coord in ("left", "bottom", "right", "top")],
                format="xyxy",
                spatial_size=spatial_size,
            ),
            segmentation=torch.as_tensor(content["seg"]),
        )

    def _prepare_sample(
        self,
        data: Tuple[Tuple[str, Tuple[str, BinaryIO]], Any],
        *,
        prepare_ann_fn: Callable[[Any, Tuple[int, int]], Dict[str, Any]],
    ) -> Dict[str, Any]:
        data, anns_data = data
        _, image_data = data
        path, buffer = image_data

        image = EncodedImage.from_file(buffer)

        return dict(
            prepare_ann_fn(anns_data, image.spatial_size),
            image=image,
            label=Label(
                int(pathlib.Path(path).parent.name.rsplit(".", 1)[0]) - 1,
                categories=self._categories,
            ),
        )

    def _datapipe(self, resource_dps: List[IterDataPipe]) -> IterDataPipe[Dict[str, Any]]:
        prepare_ann_fn: Callable
        if self._year == "2011":
            archive_dp, segmentations_dp = resource_dps
            images_dp, split_dp, image_files_dp, bounding_boxes_dp = Demultiplexer(
                archive_dp, 4, self._2011_classify_archive, drop_none=True, buffer_size=INFINITE_BUFFER_SIZE
            )

            image_files_dp = CSVParser(image_files_dp, dialect="cub200")
            image_files_dp = Mapper(image_files_dp, self._2011_extract_file_name, input_col=1)
            image_files_map = IterToMapConverter(image_files_dp)

            split_dp = CSVParser(split_dp, dialect="cub200")
            split_dp = Filter(split_dp, self._2011_filter_split)
            split_dp = Mapper(split_dp, getitem(0))
            split_dp = Mapper(split_dp, image_files_map.__getitem__)

            bounding_boxes_dp = CSVParser(bounding_boxes_dp, dialect="cub200")
            bounding_boxes_dp = Mapper(bounding_boxes_dp, image_files_map.__getitem__, input_col=0)

            anns_dp = IterKeyZipper(
                bounding_boxes_dp,
                segmentations_dp,
                key_fn=getitem(0),
                ref_key_fn=self._2011_segmentation_key,
                keep_key=True,
                buffer_size=INFINITE_BUFFER_SIZE,
            )

            prepare_ann_fn = self._2011_prepare_ann
        else:  # self._year == "2010"
            split_dp, images_dp, anns_dp = resource_dps

            split_dp = Filter(split_dp, path_comparator("name", f"{self._split}.txt"))
            split_dp = LineReader(split_dp, decode=True, return_path=False)
            split_dp = Mapper(split_dp, self._2010_split_key)

            anns_dp = Mapper(anns_dp, self._2010_anns_key)

            prepare_ann_fn = self._2010_prepare_ann

        split_dp = hint_shuffling(split_dp)
        split_dp = hint_sharding(split_dp)

        dp = IterKeyZipper(
            split_dp,
            images_dp,
            getitem(),
            path_accessor("name"),
            buffer_size=INFINITE_BUFFER_SIZE,
        )
        dp = IterKeyZipper(
            dp,
            anns_dp,
            getitem(0),
            buffer_size=INFINITE_BUFFER_SIZE,
        )
        return Mapper(dp, functools.partial(self._prepare_sample, prepare_ann_fn=prepare_ann_fn))

    def __len__(self) -> int:
        return {
            ("train", "2010"): 3_000,
            ("test", "2010"): 3_033,
            ("train", "2011"): 5_994,
            ("test", "2011"): 5_794,
        }[(self._split, self._year)]

    def _generate_categories(self) -> List[str]:
        self._year = "2011"
        resources = self._resources()

        dp = resources[0].load(self._root)
        dp = Filter(dp, path_comparator("name", "classes.txt"))
        dp = CSVDictParser(dp, fieldnames=("label", "category"), dialect="cub200")

        return [row["category"].split(".")[1] for row in dp]
