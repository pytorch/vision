import io
import pathlib
import re
from collections import OrderedDict
from typing import Any, Callable, Dict, List, Optional, Tuple, cast

import torch
from torchdata.datapipes.iter import (
    IterDataPipe,
    Mapper,
    Shuffler,
    Filter,
    Demultiplexer,
    Grouper,
    IterKeyZipper,
    JsonParser,
    UnBatcher,
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
    MappingIterator,
    INFINITE_BUFFER_SIZE,
    BUILTIN_DIR,
    getitem,
    path_accessor,
)
from torchvision.prototype.features import BoundingBox, Label, Feature
from torchvision.prototype.utils._internal import FrozenMapping


class Coco(Dataset):
    def _make_info(self) -> DatasetInfo:
        name = "coco"
        categories, super_categories = zip(*DatasetInfo.read_categories_file(BUILTIN_DIR / f"{name}.categories"))

        return DatasetInfo(
            name,
            type=DatasetType.IMAGE,
            dependencies=("pycocotools",),
            categories=categories,
            homepage="https://cocodataset.org/",
            valid_options=dict(
                split=("train", "val"),
                year=("2017", "2014"),
                annotations=(*self._ANN_DECODERS.keys(), None),
            ),
            extra=dict(category_to_super_category=FrozenMapping(zip(categories, super_categories))),
        )

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

    def resources(self, config: DatasetConfig) -> List[OnlineResource]:
        images = HttpResource(
            f"{self._IMAGE_URL_BASE}/{config.split}{config.year}.zip",
            sha256=self._IMAGES_CHECKSUMS[(config.year, config.split)],
        )
        meta = HttpResource(
            f"{self._META_URL_BASE}/annotations_trainval{config.year}.zip",
            sha256=self._META_CHECKSUMS[config.year],
        )
        return [images, meta]

    def _segmentation_to_mask(self, segmentation: Any, *, is_crowd: bool, image_size: Tuple[int, int]) -> torch.Tensor:
        from pycocotools import mask

        if is_crowd:
            segmentation = mask.frPyObjects(segmentation, *image_size)
        else:
            segmentation = mask.merge(mask.frPyObjects(segmentation, *image_size))

        return torch.from_numpy(mask.decode(segmentation)).to(torch.bool)

    def _decode_instances_anns(self, anns: List[Dict[str, Any]], image_meta: Dict[str, Any]) -> Dict[str, Any]:
        image_size = (image_meta["height"], image_meta["width"])
        labels = [ann["category_id"] for ann in anns]
        categories = [self.info.categories[label] for label in labels]
        return dict(
            # TODO: create a segmentation feature
            segmentations=Feature(
                torch.stack(
                    [
                        self._segmentation_to_mask(ann["segmentation"], is_crowd=ann["iscrowd"], image_size=image_size)
                        for ann in anns
                    ]
                )
            ),
            areas=Feature([ann["area"] for ann in anns]),
            crowds=Feature([ann["iscrowd"] for ann in anns], dtype=torch.bool),
            bounding_boxes=BoundingBox(
                [ann["bbox"] for ann in anns],
                format="xywh",
                image_size=image_size,
            ),
            labels=Label(labels),
            categories=categories,
            super_categories=[self.info.extra.category_to_super_category[category] for category in categories],
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
        fr"(?P<annotations>({'|'.join(_ANN_DECODERS.keys())}))_(?P<split>[a-zA-Z]+)(?P<year>\d+)[.]json"
    )

    def _filter_meta_files(self, data: Tuple[str, Any], *, split: str, year: str, annotations: str) -> bool:
        match = self._META_FILE_PATTERN.match(pathlib.Path(data[0]).name)
        return bool(match and match["split"] == split and match["year"] == year and match["annotations"] == annotations)

    def _classify_meta(self, data: Tuple[str, Any]) -> Optional[int]:
        key, _ = data
        if key == "images":
            return 0
        elif key == "annotations":
            return 1
        else:
            return None

    def _collate_and_decode_image(
        self, data: Tuple[str, io.IOBase], *, decoder: Optional[Callable[[io.IOBase], torch.Tensor]]
    ) -> Dict[str, Any]:
        path, buffer = data
        return dict(path=path, image=decoder(buffer) if decoder else buffer)

    def _collate_and_decode_sample(
        self,
        data: Tuple[Tuple[List[Dict[str, Any]], Dict[str, Any]], Tuple[str, io.IOBase]],
        *,
        annotations: Optional[str],
        decoder: Optional[Callable[[io.IOBase], torch.Tensor]],
    ) -> Dict[str, Any]:
        ann_data, image_data = data
        anns, image_meta = ann_data

        sample = self._collate_and_decode_image(image_data, decoder=decoder)
        if annotations:
            sample.update(self._ANN_DECODERS[annotations](self, anns, image_meta))

        return sample

    def _make_datapipe(
        self,
        resource_dps: List[IterDataPipe],
        *,
        config: DatasetConfig,
        decoder: Optional[Callable[[io.IOBase], torch.Tensor]],
    ) -> IterDataPipe[Dict[str, Any]]:
        images_dp, meta_dp = resource_dps

        if config.annotations is None:
            dp = Shuffler(images_dp)
            return Mapper(dp, self._collate_and_decode_image, fn_kwargs=dict(decoder=decoder))

        meta_dp = Filter(
            meta_dp,
            self._filter_meta_files,
            fn_kwargs=dict(split=config.split, year=config.year, annotations=config.annotations),
        )
        meta_dp = JsonParser(meta_dp)
        meta_dp = Mapper(meta_dp, getitem(1))
        meta_dp = MappingIterator(meta_dp)
        images_meta_dp, anns_meta_dp = Demultiplexer(
            meta_dp,
            2,
            self._classify_meta,
            drop_none=True,
            buffer_size=INFINITE_BUFFER_SIZE,
        )

        images_meta_dp = Mapper(images_meta_dp, getitem(1))
        images_meta_dp = UnBatcher(images_meta_dp)
        images_meta_dp = Shuffler(images_meta_dp)

        anns_meta_dp = Mapper(anns_meta_dp, getitem(1))
        anns_meta_dp = UnBatcher(anns_meta_dp)
        anns_meta_dp = Grouper(anns_meta_dp, group_key_fn=getitem("image_id"), buffer_size=INFINITE_BUFFER_SIZE)

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
        return Mapper(
            dp, self._collate_and_decode_sample, fn_kwargs=dict(annotations=config.annotations, decoder=decoder)
        )

    def _generate_categories(self, root: pathlib.Path) -> Tuple[Tuple[str, str]]:
        config = self.default_config
        resources = self.resources(config)

        dp = resources[1].load(pathlib.Path(root) / self.name)
        dp = Filter(
            dp, self._filter_meta_files, fn_kwargs=dict(split=config.split, year=config.year, annotations="instances")
        )
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
