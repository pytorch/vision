from functools import partial
from pathlib import Path
from typing import Any, Dict, List

from torchdata.datapipes.iter import IterDataPipe, Mapper, Filter, Demultiplexer, IterKeyZipper, JsonParser
from torchvision.prototype.datasets.utils import (
    Dataset,
    DatasetInfo,
    DatasetConfig,
    ManualDownloadResource,
    OnlineResource,
)
from torchvision.prototype.datasets.utils._internal import INFINITE_BUFFER_SIZE
from torchvision.prototype.features import EncodedImage


class CityscapesDatasetInfo(DatasetInfo):
    def __init__(self, *args: Any, **kwargs: Any):
        super().__init__(*args, **kwargs)
        self._configs = tuple(
            config
            for config in self._configs
            if not (
                (config.split == "test" and config.mode == "coarse")
                or (config.split == "train_extra" and config.mode == "fine")
            )
        )

    def make_config(self, **options: Any) -> DatasetConfig:
        config = super().make_config(**options)
        if config.split == "test" and config.mode == "coarse":
            raise ValueError("`split='test'` is only available for `mode='fine'`")
        if config.split == "train_extra" and config.mode == "fine":
            raise ValueError("`split='train_extra'` is only available for `mode='coarse'`")

        return config


class CityscapesResource(ManualDownloadResource):
    def __init__(self, **kwargs: Any) -> None:
        super().__init__(
            "Register on https://www.cityscapes-dataset.com/login/ and follow the instructions there.", **kwargs
        )


class Cityscapes(Dataset):
    def _make_info(self) -> DatasetInfo:
        name = "cityscapes"
        categories = None

        return CityscapesDatasetInfo(
            name,
            categories=categories,
            homepage="http://www.cityscapes-dataset.com/",
            valid_options=dict(
                split=("train", "val", "test", "train_extra"),
                mode=("fine", "coarse"),
                # target_type=("instance", "semantic", "polygon", "color")
            ),
        )

    _FILES_CHECKSUMS = {
        "gtCoarse.zip": "3555e09349ed49127053d940eaa66a87a79a175662b329c1a26a58d47e602b5b",
        "gtFine_trainvaltest.zip": "40461a50097844f400fef147ecaf58b18fd99e14e4917fb7c3bf9c0d87d95884",
        "leftImg8bit_trainextra.zip": "e41cc14c0c06aad051d52042465d9b8c22bacf6e4c93bb98de273ed7177b7133",
        "leftImg8bit_trainvaltest.zip": "3ccff9ac1fa1d80a6a064407e589d747ed0657aac7dc495a4403ae1235a37525",
    }

    def resources(self, config: DatasetConfig) -> List[OnlineResource]:
        if config.mode == "fine":
            resources = [
                CityscapesResource(
                    file_name="leftImg8bit_trainvaltest.zip",
                    sha256=self._FILES_CHECKSUMS["leftImg8bit_trainvaltest.zip"],
                ),
                CityscapesResource(
                    file_name="gtFine_trainvaltest.zip", sha256=self._FILES_CHECKSUMS["gtFine_trainvaltest.zip"]
                ),
            ]
        else:
            resources = [
                CityscapesResource(
                    file_name="leftImg8bit_trainextra.zip", sha256=self._FILES_CHECKSUMS["leftImg8bit_trainextra.zip"]
                ),
                CityscapesResource(file_name="gtCoarse.zip", sha256=self._FILES_CHECKSUMS["gtCoarse.zip"]),
            ]
        return resources

    def _filter_split_images(self, data, *, req_split: str):
        path = Path(data[0])
        split = path.parent.parts[-2]
        return split == req_split and ".png" == path.suffix

    def _filter_classify_targets(self, data, *, req_split: str):
        path = Path(data[0])
        name = path.name
        split = path.parent.parts[-2]
        if split != req_split:
            return None
        for i, target_type in enumerate(["instance", "label", "polygon", "color"]):
            ext = ".json" if target_type == "polygon" else ".png"
            if ext in path.suffix and target_type in name:
                return i
        return None

    def _prepare_sample(self, data):
        (img_path, img_data), target_data = data

        color_path, color_data = target_data[1]
        target_data = target_data[0]
        polygon_path, polygon_data = target_data[1]
        target_data = target_data[0]
        label_path, label_data = target_data[1]
        target_data = target_data[0]
        instance_path, instance_data = target_data

        return dict(
            image_path=img_path,
            image=EncodedImage.from_file(img_data),
            color_path=color_path,
            color=EncodedImage.from_file(color_data),
            polygon_path=polygon_path,
            polygon=polygon_data,
            segmentation_path=label_path,
            segmentation=EncodedImage.from_file(label_data),
            instances_path=color_path,
            instances=EncodedImage.from_file(instance_data),
        )

    def _make_datapipe(
        self,
        resource_dps: List[IterDataPipe],
        *,
        config: DatasetConfig,
    ) -> IterDataPipe[Dict[str, Any]]:
        archive_images, archive_targets = resource_dps

        images_dp = Filter(archive_images, filter_fn=partial(self._filter_split_images, req_split=config.split))

        targets_dps = Demultiplexer(
            archive_targets,
            4,
            classifier_fn=partial(self._filter_classify_targets, req_split=config.split),
            drop_none=True,
            buffer_size=INFINITE_BUFFER_SIZE,
        )

        # targets_dps[2] is for json polygon, we have to decode them
        targets_dps[2] = JsonParser(targets_dps[2])

        def img_key_fn(data):
            stem = Path(data[0]).stem
            stem = stem[: -len("_leftImg8bit")]
            return stem

        def target_key_fn(data, level=0):
            path = data[0]
            for _ in range(level):
                path = path[0]
            stem = Path(path).stem
            i = stem.rfind("_gt")
            stem = stem[:i]
            return stem

        zipped_targets_dp = targets_dps[0]
        for level, data_dp in enumerate(targets_dps[1:]):
            zipped_targets_dp = IterKeyZipper(
                zipped_targets_dp,
                data_dp,
                key_fn=partial(target_key_fn, level=level),
                ref_key_fn=target_key_fn,
                buffer_size=INFINITE_BUFFER_SIZE,
            )

        samples = IterKeyZipper(
            images_dp,
            zipped_targets_dp,
            key_fn=img_key_fn,
            ref_key_fn=partial(target_key_fn, level=len(targets_dps) - 1),
            buffer_size=INFINITE_BUFFER_SIZE,
        )
        return Mapper(samples, fn=self._prepare_sample)
