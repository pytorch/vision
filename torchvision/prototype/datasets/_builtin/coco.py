import io
import pathlib
from typing import Any, Callable, Dict, List, Optional, Tuple

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
    getitem,
    path_accessor,
    path_comparator,
)

HERE = pathlib.Path(__file__).parent


class Coco(Dataset):
    def _make_info(self) -> DatasetInfo:
        return DatasetInfo(
            "coco",
            type=DatasetType.IMAGE,
            homepage="https://cocodataset.org/",
            valid_options=dict(
                split=("train", "val"),
                year=("2017", "2014"),
            ),
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

    def _classify_meta(self, data: Tuple[str, Any]) -> Optional[int]:
        key, _ = data
        if key == "images":
            return 0
        elif key == "annotations":
            return 1
        else:
            return None

    def _decode_ann(self, ann: Dict[str, Any]) -> Dict[str, Any]:
        area = torch.tensor(ann["area"])
        iscrowd = bool(ann["iscrowd"])
        bbox = torch.tensor(ann["bbox"])
        id = ann["id"]
        return dict(area=area, iscrowd=iscrowd, bbox=bbox, id=id)

    def _collate_and_decode_sample(
        self,
        data: Tuple[Tuple[List[Dict[str, Any]], Dict[str, Any]], Tuple[str, io.IOBase]],
        *,
        decoder: Optional[Callable[[io.IOBase], torch.Tensor]],
    ) -> Dict[str, Any]:
        ann_data, image_data = data
        anns, image_meta = ann_data
        path, buffer = image_data

        anns = [self._decode_ann(ann) for ann in anns]

        image = decoder(buffer) if decoder else buffer

        return dict(anns=anns, id=image_meta["id"], path=path, image=image)

    def _make_datapipe(
        self,
        resource_dps: List[IterDataPipe],
        *,
        config: DatasetConfig,
        decoder: Optional[Callable[[io.IOBase], torch.Tensor]],
    ) -> IterDataPipe[Dict[str, Any]]:
        images_dp, meta_dp = resource_dps

        meta_dp = ZipArchiveReader(meta_dp)
        meta_dp = Filter(meta_dp, path_comparator("name", value=f"instances_{config.split}{config.year}.json"))
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

        anns_dp = Grouper(anns_meta_dp, group_key_fn=getitem("image_id"), buffer_size=INFINITE_BUFFER_SIZE)
        # drop images without annotations
        anns_dp = Filter(anns_dp, bool)
        anns_dp = Shuffler(anns_dp, buffer_size=INFINITE_BUFFER_SIZE)
        anns_dp = IterKeyZipper(
            anns_dp,
            images_meta_dp,
            key_fn=getitem(0, "image_id"),
            ref_key_fn=getitem("id"),
            buffer_size=INFINITE_BUFFER_SIZE,
        )

        images_dp = ZipArchiveReader(images_dp)

        dp = IterKeyZipper(
            anns_dp,
            images_dp,
            key_fn=getitem(1, "file_name"),
            ref_key_fn=path_accessor("name"),
            buffer_size=INFINITE_BUFFER_SIZE,
        )
        return Mapper(dp, self._collate_and_decode_sample, fn_kwargs=dict(decoder=decoder))
