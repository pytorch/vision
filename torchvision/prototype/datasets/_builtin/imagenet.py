import io
import pathlib
import re
from typing import Any, Callable, Dict, List, Optional, Tuple, cast

import torch
from torchdata.datapipes.iter import IterDataPipe, LineReader, IterKeyZipper, Mapper, TarArchiveReader, Filter, Shuffler
from torchvision.prototype.datasets.utils import (
    Dataset,
    DatasetConfig,
    DatasetInfo,
    OnlineResource,
    ManualDownloadResource,
    DatasetType,
)
from torchvision.prototype.datasets.utils._internal import (
    INFINITE_BUFFER_SIZE,
    BUILTIN_DIR,
    path_comparator,
    Enumerator,
    getitem,
    read_mat,
)
from torchvision.prototype.features import Label
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
            type=DatasetType.IMAGE,
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

    @property
    def category_to_wnid(self) -> Dict[str, str]:
        return cast(Dict[str, str], self.info.extra.category_to_wnid)

    @property
    def wnid_to_category(self) -> Dict[str, str]:
        return cast(Dict[str, str], self.info.extra.wnid_to_category)

    _IMAGES_CHECKSUMS = {
        "train": "b08200a27a8e34218a0e58fde36b0fe8f73bc377f4acea2d91602057c3ca45bb",
        "val": "c7e06a6c0baccf06d8dbeb6577d71efff84673a5dbdd50633ab44f8ea0456ae0",
        "test_v10102019": "9cf7f8249639510f17d3d8a0deb47cd22a435886ba8e29e2b3223e65a4079eb4",
    }

    def resources(self, config: DatasetConfig) -> List[OnlineResource]:
        name = "test_v10102019" if config.split == "test" else config.split
        images = ImageNetResource(file_name=f"ILSVRC2012_img_{name}.tar", sha256=self._IMAGES_CHECKSUMS[name])

        devkit = ImageNetResource(
            file_name="ILSVRC2012_devkit_t12.tar.gz",
            sha256="b59243268c0d266621fd587d2018f69e906fb22875aca0e295b48cafaa927953",
        )

        return [images, devkit]

    _TRAIN_IMAGE_NAME_PATTERN = re.compile(r"(?P<wnid>n\d{8})_\d+[.]JPEG")

    def _collate_train_data(self, data: Tuple[str, io.IOBase]) -> Tuple[Tuple[Label, str, str], Tuple[str, io.IOBase]]:
        path = pathlib.Path(data[0])
        wnid = self._TRAIN_IMAGE_NAME_PATTERN.match(path.name).group("wnid")  # type: ignore[union-attr]
        category = self.wnid_to_category[wnid]
        label_data = (Label(self.categories.index(category)), category, wnid)
        return label_data, data

    _VAL_TEST_IMAGE_NAME_PATTERN = re.compile(r"ILSVRC2012_(val|test)_(?P<id>\d{8})[.]JPEG")

    def _val_test_image_key(self, data: Tuple[str, Any]) -> int:
        path = pathlib.Path(data[0])
        return int(self._VAL_TEST_IMAGE_NAME_PATTERN.match(path.name).group("id"))  # type: ignore[union-attr]

    def _collate_val_data(
        self, data: Tuple[Tuple[int, int], Tuple[str, io.IOBase]]
    ) -> Tuple[Tuple[Label, str, str], Tuple[str, io.IOBase]]:
        label_data, image_data = data
        _, label = label_data
        category = self.categories[label]
        wnid = self.category_to_wnid[category]
        return (Label(label), category, wnid), image_data

    def _collate_test_data(self, data: Tuple[str, io.IOBase]) -> Tuple[None, Tuple[str, io.IOBase]]:
        return None, data

    def _collate_and_decode_sample(
        self,
        data: Tuple[Optional[Tuple[Label, str, str]], Tuple[str, io.IOBase]],
        *,
        decoder: Optional[Callable[[io.IOBase], torch.Tensor]],
    ) -> Dict[str, Any]:
        label_data, (path, buffer) = data

        sample = dict(
            path=path,
            image=decoder(buffer) if decoder else buffer,
        )
        if label_data:
            sample.update(dict(zip(("label", "category", "wnid"), label_data)))

        return sample

    def _make_datapipe(
        self,
        resource_dps: List[IterDataPipe],
        *,
        config: DatasetConfig,
        decoder: Optional[Callable[[io.IOBase], torch.Tensor]],
    ) -> IterDataPipe[Dict[str, Any]]:
        images_dp, devkit_dp = resource_dps

        if config.split == "train":
            # the train archive is a tar of tars
            dp = TarArchiveReader(images_dp)
            dp = Shuffler(dp, buffer_size=INFINITE_BUFFER_SIZE)
            dp = Mapper(dp, self._collate_train_data)
        elif config.split == "val":
            devkit_dp = Filter(devkit_dp, path_comparator("name", "ILSVRC2012_validation_ground_truth.txt"))
            devkit_dp = LineReader(devkit_dp, return_path=False)
            devkit_dp = Mapper(devkit_dp, int)
            devkit_dp = Enumerator(devkit_dp, 1)
            devkit_dp = Shuffler(devkit_dp, buffer_size=INFINITE_BUFFER_SIZE)

            dp = IterKeyZipper(
                devkit_dp,
                images_dp,
                key_fn=getitem(0),
                ref_key_fn=self._val_test_image_key,
                buffer_size=INFINITE_BUFFER_SIZE,
            )
            dp = Mapper(dp, self._collate_val_data)
        else:  # config.split == "test"
            dp = Shuffler(images_dp, buffer_size=INFINITE_BUFFER_SIZE)
            dp = Mapper(dp, self._collate_test_data)

        return Mapper(dp, self._collate_and_decode_sample, fn_kwargs=dict(decoder=decoder))

    # Although the WordNet IDs (wnids) are unique, the corresponding categories are not. For example, both n02012849
    # and n03126707 are labeled 'crane' while the first means the bird and the latter means the construction equipment
    _WNID_MAP = {
        "n03126707": "construction crane",
        "n03710721": "tank suit",
    }

    def _generate_categories(self, root: pathlib.Path) -> List[Tuple[str, ...]]:
        resources = self.resources(self.default_config)
        devkit_dp = resources[1].load(root / self.name)
        devkit_dp = Filter(devkit_dp, path_comparator("name", "meta.mat"))

        meta = next(iter(devkit_dp))[1]
        synsets = read_mat(meta, squeeze_me=True)["synsets"]
        categories_and_wnids = cast(
            List[Tuple[str, ...]],
            [
                (self._WNID_MAP.get(wnid, category.split(",", 1)[0]), wnid)
                for _, wnid, category, _, num_children, *_ in synsets
                # if num_children > 0, we are looking at a superclass that has no direct instance
                if num_children == 0
            ],
        )
        categories_and_wnids.sort(key=lambda category_and_wnid: category_and_wnid[1])

        return categories_and_wnids
