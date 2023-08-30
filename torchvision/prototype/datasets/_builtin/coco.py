import pathlib
import re
from collections import defaultdict, OrderedDict
from typing import Any, BinaryIO, cast, Dict, List, Optional, Tuple, Union

import torch
from torchdata.datapipes.iter import (
    Demultiplexer,
    Filter,
    Grouper,
    IterDataPipe,
    IterKeyZipper,
    JsonParser,
    Mapper,
    UnBatcher,
)
from torchvision.prototype.datasets.utils import Dataset, EncodedImage, HttpResource, OnlineResource
from torchvision.prototype.datasets.utils._internal import (
    getitem,
    hint_sharding,
    hint_shuffling,
    INFINITE_BUFFER_SIZE,
    MappingIterator,
    path_accessor,
    read_categories_file,
)
from torchvision.prototype.tv_tensors import Label
from torchvision.tv_tensors import BoundingBoxes, Mask

from .._api import register_dataset, register_info


NAME = "coco"


@register_info(NAME)
def _info() -> Dict[str, Any]:
    categories, super_categories = zip(*read_categories_file(NAME))
    return dict(categories=categories, super_categories=super_categories)


@register_dataset(NAME)
class Coco(Dataset):
    """
    - **homepage**: https://cocodataset.org/
    - **dependencies**:
        - <pycocotools `https://github.com/cocodataset/cocoapi`>_
    """

    def __init__(
        self,
        root: Union[str, pathlib.Path],
        *,
        split: str = "train",
        year: str = "2017",
        annotations: Optional[str] = "instances",
        skip_integrity_check: bool = False,
    ) -> None:
        self._split = self._verify_str_arg(split, "split", {"train", "val"})
        self._year = self._verify_str_arg(year, "year", {"2017", "2014"})
        self._annotations = (
            self._verify_str_arg(annotations, "annotations", self._ANN_DECODERS.keys())
            if annotations is not None
            else None
        )

        info = _info()
        categories, super_categories = info["categories"], info["super_categories"]
        self._categories = categories
        self._category_to_super_category = dict(zip(categories, super_categories))

        super().__init__(root, dependencies=("pycocotools",), skip_integrity_check=skip_integrity_check)

    _IMAGE_URL_BASE = "http://images.cocodataset.org/zips"

    _IMAGES_CHECKSUMS = {
        ("2014", "train"): "ede4087e640bddba550e090eae701092534b554b42b05ac33f0300b984b31775",
        ("2014", "val"): "fe9be816052049c34717e077d9e34aa60814a55679f804cd043e3cbee3b9fde0",
        ("2017", "train"): "69a8bb58ea5f8f99d24875f21416de2e9ded3178e903f1f7603e283b9e06d929",
        ("2017", "val"): "4f7e2ccb2866ec5041993c9cf2a952bbed69647b115d0f74da7ce8f4bef82f05",
    }

    _META_URL_BASE = "http://images.cocodataset.org/annotations"

    _META_CHECKSUMS = {
        "2014": "031296bbc80c45a1d1f76bf9a90ead27e94e99ec629208449507a4917a3bf009",
        "2017": "113a836d90195ee1f884e704da6304dfaaecff1f023f49b6ca93c4aaae470268",
    }

    def _resources(self) -> List[OnlineResource]:
        images = HttpResource(
            f"{self._IMAGE_URL_BASE}/{self._split}{self._year}.zip",
            sha256=self._IMAGES_CHECKSUMS[(self._year, self._split)],
        )
        meta = HttpResource(
            f"{self._META_URL_BASE}/annotations_trainval{self._year}.zip",
            sha256=self._META_CHECKSUMS[self._year],
        )
        return [images, meta]

    def _segmentation_to_mask(
        self, segmentation: Any, *, is_crowd: bool, spatial_size: Tuple[int, int]
    ) -> torch.Tensor:
        from pycocotools import mask

        if is_crowd:
            segmentation = mask.frPyObjects(segmentation, *spatial_size)
        else:
            segmentation = mask.merge(mask.frPyObjects(segmentation, *spatial_size))

        return torch.from_numpy(mask.decode(segmentation)).to(torch.bool)

    def _decode_instances_anns(self, anns: List[Dict[str, Any]], image_meta: Dict[str, Any]) -> Dict[str, Any]:
        spatial_size = (image_meta["height"], image_meta["width"])
        labels = [ann["category_id"] for ann in anns]
        return dict(
            segmentations=Mask(
                torch.stack(
                    [
                        self._segmentation_to_mask(
                            ann["segmentation"], is_crowd=ann["iscrowd"], spatial_size=spatial_size
                        )
                        for ann in anns
                    ]
                )
            ),
            areas=torch.as_tensor([ann["area"] for ann in anns]),
            crowds=torch.as_tensor([ann["iscrowd"] for ann in anns], dtype=torch.bool),
            bounding_boxes=BoundingBoxes(
                [ann["bbox"] for ann in anns],
                format="xywh",
                spatial_size=spatial_size,
            ),
            labels=Label(labels, categories=self._categories),
            super_categories=[self._category_to_super_category[self._categories[label]] for label in labels],
            ann_ids=[ann["id"] for ann in anns],
        )

    def _decode_captions_ann(self, anns: List[Dict[str, Any]], image_meta: Dict[str, Any]) -> Dict[str, Any]:
        return dict(
            captions=[ann["caption"] for ann in anns],
            ann_ids=[ann["id"] for ann in anns],
        )

    _ANN_DECODERS = OrderedDict(
        [
            ("instances", _decode_instances_anns),
            ("captions", _decode_captions_ann),
        ]
    )

    _META_FILE_PATTERN = re.compile(
        rf"(?P<annotations>({'|'.join(_ANN_DECODERS.keys())}))_(?P<split>[a-zA-Z]+)(?P<year>\d+)[.]json"
    )

    def _filter_meta_files(self, data: Tuple[str, Any]) -> bool:
        match = self._META_FILE_PATTERN.match(pathlib.Path(data[0]).name)
        return bool(
            match
            and match["split"] == self._split
            and match["year"] == self._year
            and match["annotations"] == self._annotations
        )

    def _classify_meta(self, data: Tuple[str, Any]) -> Optional[int]:
        key, _ = data
        if key == "images":
            return 0
        elif key == "annotations":
            return 1
        else:
            return None

    def _prepare_image(self, data: Tuple[str, BinaryIO]) -> Dict[str, Any]:
        path, buffer = data
        return dict(
            path=path,
            image=EncodedImage.from_file(buffer),
        )

    def _prepare_sample(
        self,
        data: Tuple[Tuple[List[Dict[str, Any]], Dict[str, Any]], Tuple[str, BinaryIO]],
    ) -> Dict[str, Any]:
        ann_data, image_data = data
        anns, image_meta = ann_data

        sample = self._prepare_image(image_data)
        # this method is only called if we have annotations
        annotations = cast(str, self._annotations)
        sample.update(self._ANN_DECODERS[annotations](self, anns, image_meta))
        return sample

    def _datapipe(self, resource_dps: List[IterDataPipe]) -> IterDataPipe[Dict[str, Any]]:
        images_dp, meta_dp = resource_dps

        if self._annotations is None:
            dp = hint_shuffling(images_dp)
            dp = hint_sharding(dp)
            dp = hint_shuffling(dp)
            return Mapper(dp, self._prepare_image)

        meta_dp = Filter(meta_dp, self._filter_meta_files)
        meta_dp = JsonParser(meta_dp)
        meta_dp = Mapper(meta_dp, getitem(1))
        meta_dp: IterDataPipe[Dict[str, Dict[str, Any]]] = MappingIterator(meta_dp)
        images_meta_dp, anns_meta_dp = Demultiplexer(
            meta_dp,
            2,
            self._classify_meta,
            drop_none=True,
            buffer_size=INFINITE_BUFFER_SIZE,
        )

        images_meta_dp = Mapper(images_meta_dp, getitem(1))
        images_meta_dp = UnBatcher(images_meta_dp)

        anns_meta_dp = Mapper(anns_meta_dp, getitem(1))
        anns_meta_dp = UnBatcher(anns_meta_dp)
        anns_meta_dp = Grouper(anns_meta_dp, group_key_fn=getitem("image_id"), buffer_size=INFINITE_BUFFER_SIZE)
        anns_meta_dp = hint_shuffling(anns_meta_dp)
        anns_meta_dp = hint_sharding(anns_meta_dp)

        anns_dp = IterKeyZipper(
            anns_meta_dp,
            images_meta_dp,
            key_fn=getitem(0, "image_id"),
            ref_key_fn=getitem("id"),
            buffer_size=INFINITE_BUFFER_SIZE,
        )
        dp = IterKeyZipper(
            anns_dp,
            images_dp,
            key_fn=getitem(1, "file_name"),
            ref_key_fn=path_accessor("name"),
            buffer_size=INFINITE_BUFFER_SIZE,
        )
        return Mapper(dp, self._prepare_sample)

    def __len__(self) -> int:
        return {
            ("train", "2017"): defaultdict(lambda: 118_287, instances=117_266),
            ("train", "2014"): defaultdict(lambda: 82_783, instances=82_081),
            ("val", "2017"): defaultdict(lambda: 5_000, instances=4_952),
            ("val", "2014"): defaultdict(lambda: 40_504, instances=40_137),
        }[(self._split, self._year)][
            self._annotations  # type: ignore[index]
        ]

    def _generate_categories(self) -> Tuple[Tuple[str, str]]:
        self._annotations = "instances"
        resources = self._resources()

        dp = resources[1].load(self._root)
        dp = Filter(dp, self._filter_meta_files)
        dp = JsonParser(dp)

        _, meta = next(iter(dp))
        # List[Tuple[super_category, id, category]]
        label_data = [cast(Tuple[str, int, str], tuple(info.values())) for info in meta["categories"]]

        # COCO actually defines 91 categories, but only 80 of them have instances. Still, the category_id refers to the
        # full set. To keep the labels dense, we fill the gaps with N/A. Note that there are only 10 gaps, so the total
        # number of categories is 90 rather than 91.
        _, ids, _ = zip(*label_data)
        missing_ids = set(range(1, max(ids) + 1)) - set(ids)
        label_data.extend([("N/A", id, "N/A") for id in missing_ids])

        # We also add a background category to be used during segmentation.
        label_data.append(("N/A", 0, "__background__"))

        super_categories, _, categories = zip(*sorted(label_data, key=lambda info: info[1]))

        return cast(Tuple[Tuple[str, str]], tuple(zip(categories, super_categories)))
