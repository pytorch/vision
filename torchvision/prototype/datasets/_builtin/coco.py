import functools
import io
import pathlib
import re
from typing import Any, Callable, Dict, List, Optional, Tuple, cast

import torch
from torchdata.datapipes.iter import (
    IterDataPipe,
    Mapper,
    Shuffler,
    Filter,
    Demultiplexer,
    ZipArchiveReader,
    Grouper,
    IterKeyZipper,
    JsonParser,
    UnBatcher,
    Concater,
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
    path_comparator,
)
from torchvision.prototype.features import BoundingBox, Label
from torchvision.prototype.features._feature import DEFAULT
from torchvision.prototype.utils._internal import FrozenMapping


class CocoLabel(Label):
    super_category: Optional[str]

    @classmethod
    def _parse_meta_data(
        cls,
        category: Optional[str] = DEFAULT,  # type: ignore[assignment]
        super_category: Optional[str] = DEFAULT,  # type: ignore[assignment]
    ) -> Dict[str, Tuple[Any, Any]]:
        return dict(category=(category, None), super_category=(super_category, None))


class Coco(Dataset):
    def _decode_instances_anns(self, anns: List[Dict[str, Any]]) -> Dict[str, Any]:
        labels = [ann["category_id"] for ann in anns]
        categories = [self.info.categories[label] for label in labels]
        return dict(
            areas=torch.tensor([ann["area"] for ann in anns]),
            iscrowd=torch.tensor([ann["iscrowd"] for ann in anns], dtype=torch.bool),
            # FIXME: set the proper image_size here
            bounding_boxes=BoundingBox([ann["bbox"] for ann in anns], format="xywh") if anns else [],
            labels=[
                CocoLabel(
                    label,
                    category=category,
                    super_category=self.info.extra.category_to_super_category[category],
                )
                for label, category in zip(labels, categories)
            ],
            instances_ann_ids=[ann["id"] for ann in anns],
        )

    def _decode_captions_ann(self, anns: List[Dict[str, Any]]) -> Dict[str, Any]:
        return dict(
            captions=[ann["caption"] for ann in anns],
            captions_ann_ids=[ann["id"] for ann in anns],
        )

    _ANN_TYPES, _ANN_TYPE_DEFAULTS, _ANN_DECODERS = zip(
        *(
            ("instances", True, _decode_instances_anns),
            ("captions", False, _decode_captions_ann),
        )
    )
    _ANN_TYPES: Tuple[str, ...]
    _ANN_TYPE_DEFAULTS: Tuple[bool, ...]
    _ANN_DECODERS: Tuple[Callable[["Coco", List[Dict[str, Any]]], Dict[str, Any]], ...]

    _ANN_TYPE_OPTIONS = dict(zip(_ANN_TYPES, [(default, not default) for default in _ANN_TYPE_DEFAULTS]))
    _ANN_DECODER_MAP = dict(zip(_ANN_TYPES, _ANN_DECODERS))

    def _make_info(self) -> DatasetInfo:
        name = "coco"
        categories, super_categories = zip(*DatasetInfo.read_categories_file(BUILTIN_DIR / f"{name}.categories"))

        return DatasetInfo(
            name,
            type=DatasetType.IMAGE,
            categories=categories,
            homepage="https://cocodataset.org/",
            valid_options=dict(
                self._ANN_TYPE_OPTIONS,
                split=("train", "val"),
                year=("2017", "2014"),
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

    _META_FILE_PATTERN = re.compile(fr"(?P<ann_type>({'|'.join(_ANN_TYPES)}))_(?P<split>[a-zA-Z]+)(?P<year>\d+)[.]json")

    def _classifiy_meta_files(
        self, data: Tuple[str, Any], *, split: str, year: str, ann_type_idcs: Dict[str, int]
    ) -> Optional[int]:
        match = self._META_FILE_PATTERN.match(pathlib.Path(data[0]).name)
        if not match or match["split"] != split or match["year"] != year:
            return None

        return ann_type_idcs.get(match["ann_type"])

    def _classify_meta(self, data: Tuple[str, Any]) -> Optional[int]:
        key, _ = data
        if key == "images":
            return 0
        elif key == "annotations":
            return 1
        else:
            return None

    def _make_partial_anns_dp(self, meta_dp: IterDataPipe[Tuple[str, io.IOBase]]) -> IterDataPipe:
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

        anns_meta_dp = Mapper(anns_meta_dp, getitem(1))
        anns_meta_dp = UnBatcher(anns_meta_dp)

        partial_anns_dp = Grouper(anns_meta_dp, group_key_fn=getitem("image_id"), buffer_size=INFINITE_BUFFER_SIZE)

        return IterKeyZipper(
            partial_anns_dp,
            images_meta_dp,
            key_fn=getitem(0, "image_id"),
            ref_key_fn=getitem("id"),
            buffer_size=INFINITE_BUFFER_SIZE,
        )

    def _precollate_anns(
        self, data: List[Tuple[List[Dict[str, Any]], Dict[str, Any]]], *, types: List[str]
    ) -> Tuple[str, Dict[str, Dict[str, Any]]]:
        ann_data, (image_data, *_) = zip(*data)
        anns = dict(zip(types, ann_data))
        for empty_ann_types in set(self._ANN_TYPES) - set(types):
            anns[empty_ann_types] = []
        return image_data["file_name"], anns

    def _make_anns_dp(
        self, meta_dp: IterDataPipe[Tuple[str, io.IOBase]], *, config: DatasetConfig
    ) -> IterDataPipe[Tuple[str, Dict[str, Dict[str, Any]]]]:
        meta_dp = ZipArchiveReader(meta_dp)

        ann_types = [type for type in self._ANN_TYPES if config[type]]
        partial_anns_dps = Demultiplexer(
            meta_dp,
            len(ann_types),
            functools.partial(
                self._classifiy_meta_files,
                split=config.split,
                year=config.year,
                ann_type_idcs=dict(zip(ann_types, range(len(ann_types)))),
            ),
            drop_none=True,
            buffer_size=INFINITE_BUFFER_SIZE,
        )
        partial_anns_dps = [self._make_partial_anns_dp(dp) for dp in partial_anns_dps]

        anns_dp = Concater(*partial_anns_dps)
        anns_dp = Grouper(anns_dp, group_key_fn=getitem(1, "id"), buffer_size=INFINITE_BUFFER_SIZE)
        return Mapper(anns_dp, self._precollate_anns, fn_kwargs=dict(types=ann_types))

    def _add_empty_anns(
        self, data: Tuple[str, io.IOBase]
    ) -> Tuple[Tuple[str, Dict[str, List[Dict[str, Any]]]], Tuple[str, io.IOBase]]:
        return (pathlib.Path(data[0]).name, {ann_type: [] for ann_type in self._ANN_TYPES}), data

    def _collate_and_decode_sample(
        self,
        data: Tuple[Tuple[str, Dict[str, List[Dict[str, Any]]]], Tuple[str, io.IOBase]],
        *,
        decoder: Optional[Callable[[io.IOBase], torch.Tensor]],
    ) -> Dict[str, Any]:
        ann_data, image_data = data
        _, anns = ann_data
        path, buffer = image_data

        sample = dict(path=path, image=decoder(buffer) if decoder else buffer)

        for type, partial_anns in anns.items():
            sample.update(self._ANN_DECODER_MAP[type](self, partial_anns))

        return sample

    def _make_datapipe(
        self,
        resource_dps: List[IterDataPipe],
        *,
        config: DatasetConfig,
        decoder: Optional[Callable[[io.IOBase], torch.Tensor]],
    ) -> IterDataPipe[Dict[str, Any]]:
        images_dp, meta_dp = resource_dps

        images_dp = ZipArchiveReader(images_dp)

        if any(config[ann_type] for ann_type in self._ANN_TYPES):
            anns_dp = self._make_anns_dp(meta_dp, config=config)
            anns_dp = Shuffler(anns_dp, buffer_size=INFINITE_BUFFER_SIZE)

            dp = IterKeyZipper(
                anns_dp,
                images_dp,
                key_fn=getitem(0),
                ref_key_fn=path_accessor("name"),
                buffer_size=INFINITE_BUFFER_SIZE,
            )
        else:
            images_dp = Shuffler(images_dp, buffer_size=INFINITE_BUFFER_SIZE)
            dp = Mapper(images_dp, self._add_empty_anns)
        return Mapper(dp, self._collate_and_decode_sample, fn_kwargs=dict(decoder=decoder))

    def _generate_categories(self, root: pathlib.Path) -> Tuple[Tuple[str, str]]:
        config = self.default_config
        resources = self.resources(config)

        dp = resources[1].to_datapipe(pathlib.Path(root) / self.name)
        dp = ZipArchiveReader(dp)
        dp = Filter(dp, path_comparator("name", f"instances_{config.split}{config.year}.json"))
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
