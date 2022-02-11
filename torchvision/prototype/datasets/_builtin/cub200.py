import csv
import functools
import pathlib
from typing import Any, Dict, List, Optional, Tuple, BinaryIO, Callable

from torchdata.datapipes.iter import (
    IterDataPipe,
    Mapper,
    Filter,
    IterKeyZipper,
    Demultiplexer,
    LineReader,
    CSVParser,
    CSVDictParser,
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
    hint_sharding,
    hint_shuffling,
    getitem,
    path_comparator,
    path_accessor,
)
from torchvision.prototype.features import Label, BoundingBox, _Feature, EncodedImage

csv.register_dialect("cub200", delimiter=" ")


class CUB200(Dataset):
    def _make_info(self) -> DatasetInfo:
        return DatasetInfo(
            "cub200",
            homepage="http://www.vision.caltech.edu/visipedia/CUB-200-2011.html",
            dependencies=("scipy",),
            valid_options=dict(
                split=("train", "test"),
                year=("2011", "2010"),
            ),
        )

    def resources(self, config: DatasetConfig) -> List[OnlineResource]:
        if config.year == "2011":
            archive = HttpResource(
                "http://www.vision.caltech.edu/visipedia-data/CUB-200-2011/CUB_200_2011.tgz",
                sha256="0c685df5597a8b24909f6a7c9db6d11e008733779a671760afef78feb49bf081",
                decompress=True,
            )
            segmentations = HttpResource(
                "http://www.vision.caltech.edu/visipedia-data/CUB-200-2011/segmentations.tgz",
                sha256="dc77f6cffea0cbe2e41d4201115c8f29a6320ecb04fffd2444f51b8066e4b84f",
                decompress=True,
            )
            return [archive, segmentations]
        else:  # config.year == "2010"
            split = HttpResource(
                "http://www.vision.caltech.edu/visipedia-data/CUB-200/lists.tgz",
                sha256="aeacbd5e3539ae84ea726e8a266a9a119c18f055cd80f3836d5eb4500b005428",
                decompress=True,
            )
            images = HttpResource(
                "http://www.vision.caltech.edu/visipedia-data/CUB-200/images.tgz",
                sha256="2a6d2246bbb9778ca03aa94e2e683ccb4f8821a36b7f235c0822e659d60a803e",
                decompress=True,
            )
            anns = HttpResource(
                "http://www.vision.caltech.edu/visipedia-data/CUB-200/annotations.tgz",
                sha256="c17b7841c21a66aa44ba8fe92369cc95dfc998946081828b1d7b8a4b716805c1",
                decompress=True,
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

    def _2011_filter_split(self, row: List[str], *, split: str) -> bool:
        _, split_id = row
        return {
            "0": "test",
            "1": "train",
        }[split_id] == split

    def _2011_segmentation_key(self, data: Tuple[str, Any]) -> str:
        path = pathlib.Path(data[0])
        return path.with_suffix(".jpg").name

    def _2011_prepare_ann(
        self, data: Tuple[str, Tuple[List[str], Tuple[str, BinaryIO]]], image_size: Tuple[int, int]
    ) -> Dict[str, Any]:
        _, (bounding_box_data, segmentation_data) = data
        segmentation_path, segmentation_buffer = segmentation_data
        return dict(
            bounding_box=BoundingBox(
                [float(part) for part in bounding_box_data[1:]], format="xywh", image_size=image_size
            ),
            segmentation_path=segmentation_path,
            segmentation=EncodedImage.from_file(segmentation_buffer),
        )

    def _2010_split_key(self, data: str) -> str:
        return data.rsplit("/", maxsplit=1)[1]

    def _2010_anns_key(self, data: Tuple[str, BinaryIO]) -> Tuple[str, Tuple[str, BinaryIO]]:
        path = pathlib.Path(data[0])
        return path.with_suffix(".jpg").name, data

    def _2010_prepare_ann(self, data: Tuple[str, Tuple[str, BinaryIO]], image_size: Tuple[int, int]) -> Dict[str, Any]:
        _, (path, buffer) = data
        content = read_mat(buffer)
        return dict(
            ann_path=path,
            bounding_box=BoundingBox(
                [int(content["bbox"][coord]) for coord in ("left", "bottom", "right", "top")],
                format="xyxy",
                image_size=image_size,
            ),
            segmentation=_Feature(content["seg"]),
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
            prepare_ann_fn(anns_data, image.image_size),
            image=image,
            label=Label(int(pathlib.Path(path).parent.name.rsplit(".", 1)[0]), categories=self.categories),
        )

    def _make_datapipe(
        self,
        resource_dps: List[IterDataPipe],
        *,
        config: DatasetConfig,
    ) -> IterDataPipe[Dict[str, Any]]:
        prepare_ann_fn: Callable
        if config.year == "2011":
            archive_dp, segmentations_dp = resource_dps
            images_dp, split_dp, image_files_dp, bounding_boxes_dp = Demultiplexer(
                archive_dp, 4, self._2011_classify_archive, drop_none=True, buffer_size=INFINITE_BUFFER_SIZE
            )

            image_files_dp = CSVParser(image_files_dp, dialect="cub200")
            image_files_map = dict(
                (image_id, rel_posix_path.rsplit("/", maxsplit=1)[1]) for image_id, rel_posix_path in image_files_dp
            )

            split_dp = CSVParser(split_dp, dialect="cub200")
            split_dp = Filter(split_dp, functools.partial(self._2011_filter_split, split=config.split))
            split_dp = Mapper(split_dp, getitem(0))
            split_dp = Mapper(split_dp, image_files_map.get)

            bounding_boxes_dp = CSVParser(bounding_boxes_dp, dialect="cub200")
            bounding_boxes_dp = Mapper(bounding_boxes_dp, image_files_map.get, input_col=0)

            anns_dp = IterKeyZipper(
                bounding_boxes_dp,
                segmentations_dp,
                key_fn=getitem(0),
                ref_key_fn=self._2011_segmentation_key,
                keep_key=True,
                buffer_size=INFINITE_BUFFER_SIZE,
            )

            prepare_ann_fn = self._2011_prepare_ann
        else:  # config.year == "2010"
            split_dp, images_dp, anns_dp = resource_dps

            split_dp = Filter(split_dp, path_comparator("name", f"{config.split}.txt"))
            split_dp = LineReader(split_dp, decode=True, return_path=False)
            split_dp = Mapper(split_dp, self._2010_split_key)

            anns_dp = Mapper(anns_dp, self._2010_anns_key)

            prepare_ann_fn = self._2010_prepare_ann

        split_dp = hint_sharding(split_dp)
        split_dp = hint_shuffling(split_dp)

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

    def _generate_categories(self, root: pathlib.Path) -> List[str]:
        config = self.info.make_config(year="2011")
        resources = self.resources(config)

        dp = resources[0].load(root)
        dp = Filter(dp, path_comparator("name", "classes.txt"))
        dp = CSVDictParser(dp, fieldnames=("label", "category"), dialect="cub200")

        return [row["category"].split(".")[1] for row in dp]
