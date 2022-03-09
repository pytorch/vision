import functools
import pathlib
import re
from typing import Any, Dict, List, Optional, Tuple, BinaryIO, Match, cast

from torchdata.datapipes.iter import (
    IterDataPipe,
    LineReader,
    IterKeyZipper,
    Mapper,
    Filter,
    Demultiplexer,
    TarArchiveReader,
    Enumerator,
)
from torchvision.prototype.datasets.utils import (
    Dataset,
    DatasetConfig,
    DatasetInfo,
    OnlineResource,
    ManualDownloadResource,
)
from torchvision.prototype.datasets.utils._internal import (
    INFINITE_BUFFER_SIZE,
    BUILTIN_DIR,
    path_comparator,
    getitem,
    read_mat,
    hint_sharding,
    hint_shuffling,
)
from torchvision.prototype.features import Label, EncodedImage
from torchvision.prototype.utils._internal import FrozenMapping


class ImageNetResource(ManualDownloadResource):
    def __init__(self, **kwargs: Any) -> None:
        super().__init__("Register on https://image-net.org/ and follow the instructions there.", **kwargs)


class ImageNet(Dataset):
    def _make_info(self) -> DatasetInfo:
        name = "imagenet"
        categories, wnids = zip(*DatasetInfo.read_categories_file(BUILTIN_DIR / f"{name}.categories"))

        return DatasetInfo(
            name,
            dependencies=("scipy",),
            categories=categories,
            homepage="https://www.image-net.org/",
            valid_options=dict(split=("train", "val", "test")),
            extra=dict(
                wnid_to_category=FrozenMapping(zip(wnids, categories)),
                category_to_wnid=FrozenMapping(zip(categories, wnids)),
                sizes=FrozenMapping(
                    [
                        (DatasetConfig(split="train"), 1_281_167),
                        (DatasetConfig(split="val"), 50_000),
                        (DatasetConfig(split="test"), 100_000),
                    ]
                ),
            ),
        )

    def supports_sharded(self) -> bool:
        return True

    _IMAGES_CHECKSUMS = {
        "train": "b08200a27a8e34218a0e58fde36b0fe8f73bc377f4acea2d91602057c3ca45bb",
        "val": "c7e06a6c0baccf06d8dbeb6577d71efff84673a5dbdd50633ab44f8ea0456ae0",
        "test_v10102019": "9cf7f8249639510f17d3d8a0deb47cd22a435886ba8e29e2b3223e65a4079eb4",
    }

    def resources(self, config: DatasetConfig) -> List[OnlineResource]:
        name = "test_v10102019" if config.split == "test" else config.split
        images = ImageNetResource(
            file_name=f"ILSVRC2012_img_{name}.tar",
            sha256=self._IMAGES_CHECKSUMS[name],
        )
        resources: List[OnlineResource] = [images]

        if config.split == "val":
            devkit = ImageNetResource(
                file_name="ILSVRC2012_devkit_t12.tar.gz",
                sha256="b59243268c0d266621fd587d2018f69e906fb22875aca0e295b48cafaa927953",
            )
            resources.append(devkit)

        return resources

    def num_samples(self, config: DatasetConfig) -> int:
        return {
            "train": 1_281_167,
            "val": 50_000,
            "test": 100_000,
        }[config.split]

    _TRAIN_IMAGE_NAME_PATTERN = re.compile(r"(?P<wnid>n\d{8})_\d+[.]JPEG")

    def _prepare_train_data(self, data: Tuple[str, BinaryIO]) -> Tuple[Tuple[Label, str], Tuple[str, BinaryIO]]:
        path = pathlib.Path(data[0])
        wnid = cast(Match[str], self._TRAIN_IMAGE_NAME_PATTERN.match(path.name))["wnid"]
        label = Label.from_category(self.info.extra.wnid_to_category[wnid], categories=self.categories)
        return (label, wnid), data

    def _prepare_test_data(self, data: Tuple[str, BinaryIO]) -> Tuple[None, Tuple[str, BinaryIO]]:
        return None, data

    def _classifiy_devkit(self, data: Tuple[str, BinaryIO]) -> Optional[int]:
        return {
            "meta.mat": 0,
            "ILSVRC2012_validation_ground_truth.txt": 1,
        }.get(pathlib.Path(data[0]).name)

    def _extract_categories_and_wnids(self, data: Tuple[str, BinaryIO]) -> List[Tuple[str, str]]:
        synsets = read_mat(data[1], squeeze_me=True)["synsets"]
        return [
            (self._WNID_MAP.get(wnid, category.split(",", 1)[0]), wnid)
            for _, wnid, category, _, num_children, *_ in synsets
            # if num_children > 0, we are looking at a superclass that has no direct instance
            if num_children == 0
        ]

    def _imagenet_label_to_wnid(self, imagenet_label: str, *, wnids: List[str]) -> str:
        return wnids[int(imagenet_label) - 1]

    _VAL_TEST_IMAGE_NAME_PATTERN = re.compile(r"ILSVRC2012_(val|test)_(?P<id>\d{8})[.]JPEG")

    def _val_test_image_key(self, data: Tuple[str, Any]) -> int:
        path = pathlib.Path(data[0])
        return int(self._VAL_TEST_IMAGE_NAME_PATTERN.match(path.name).group("id"))  # type: ignore[union-attr]

    def _prepare_val_data(
        self, data: Tuple[Tuple[int, str], Tuple[str, BinaryIO]]
    ) -> Tuple[Tuple[Label, str], Tuple[str, BinaryIO]]:
        label_data, image_data = data
        _, wnid = label_data
        label = Label.from_category(self.info.extra.wnid_to_category[wnid], categories=self.categories)
        return (label, wnid), image_data

    def _prepare_sample(
        self,
        data: Tuple[Optional[Tuple[Label, str]], Tuple[str, BinaryIO]],
    ) -> Dict[str, Any]:
        label_data, (path, buffer) = data

        return dict(
            dict(zip(("label", "wnid"), label_data if label_data else (None, None))),
            path=path,
            image=EncodedImage.from_file(buffer),
        )

    def _make_datapipe(
        self, resource_dps: List[IterDataPipe], *, config: DatasetConfig
    ) -> IterDataPipe[Dict[str, Any]]:
        if config.split in {"train", "test"}:
            dp = resource_dps[0]

            # the train archive is a tar of tars
            if config.split == "train":
                dp = TarArchiveReader(dp)

            dp = hint_sharding(dp)
            dp = hint_shuffling(dp)
            dp = Mapper(dp, self._prepare_train_data if config.split == "train" else self._prepare_test_data)
        else:  # config.split == "val":
            images_dp, devkit_dp = resource_dps

            meta_dp, label_dp = Demultiplexer(
                devkit_dp, 2, self._classifiy_devkit, drop_none=True, buffer_size=INFINITE_BUFFER_SIZE
            )

            meta_dp = Mapper(meta_dp, self._extract_categories_and_wnids)
            _, wnids = zip(*next(iter(meta_dp)))

            label_dp = LineReader(label_dp, decode=True, return_path=False)
            label_dp = Mapper(label_dp, functools.partial(self._imagenet_label_to_wnid, wnids=wnids))
            label_dp: IterDataPipe[Tuple[int, str]] = Enumerator(label_dp, 1)
            label_dp = hint_sharding(label_dp)
            label_dp = hint_shuffling(label_dp)

            dp = IterKeyZipper(
                label_dp,
                images_dp,
                key_fn=getitem(0),
                ref_key_fn=self._val_test_image_key,
                buffer_size=INFINITE_BUFFER_SIZE,
            )
            dp = Mapper(dp, self._prepare_val_data)

        return Mapper(dp, self._prepare_sample)

    # Although the WordNet IDs (wnids) are unique, the corresponding categories are not. For example, both n02012849
    # and n03126707 are labeled 'crane' while the first means the bird and the latter means the construction equipment
    _WNID_MAP = {
        "n03126707": "construction crane",
        "n03710721": "tank suit",
    }

    def _generate_categories(self, root: pathlib.Path) -> List[Tuple[str, ...]]:
        config = self.info.make_config(split="val")
        resources = self.resources(config)

        devkit_dp = resources[1].load(root)
        meta_dp = Filter(devkit_dp, path_comparator("name", "meta.mat"))
        meta_dp = Mapper(meta_dp, self._extract_categories_and_wnids)

        categories_and_wnids = cast(List[Tuple[str, ...]], next(iter(meta_dp)))
        categories_and_wnids.sort(key=lambda category_and_wnid: category_and_wnid[1])
        return categories_and_wnids
