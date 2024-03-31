import enum
import pathlib
import re

from typing import Any, BinaryIO, cast, Dict, Iterator, List, Match, Optional, Tuple, Union

from torchdata.datapipes.iter import (
    Demultiplexer,
    Enumerator,
    Filter,
    IterDataPipe,
    IterKeyZipper,
    LineReader,
    Mapper,
    TarArchiveLoader,
)
from torchdata.datapipes.map import IterToMapConverter
from torchvision.prototype.datasets.utils import Dataset, EncodedImage, ManualDownloadResource, OnlineResource
from torchvision.prototype.datasets.utils._internal import (
    getitem,
    hint_sharding,
    hint_shuffling,
    INFINITE_BUFFER_SIZE,
    path_accessor,
    read_categories_file,
    read_mat,
)
from torchvision.prototype.tv_tensors import Label

from .._api import register_dataset, register_info

NAME = "imagenet"


@register_info(NAME)
def _info() -> Dict[str, Any]:
    categories, wnids = zip(*read_categories_file(NAME))
    return dict(categories=categories, wnids=wnids)


class ImageNetResource(ManualDownloadResource):
    def __init__(self, **kwargs: Any) -> None:
        super().__init__("Register on https://image-net.org/ and follow the instructions there.", **kwargs)


class ImageNetDemux(enum.IntEnum):
    META = 0
    LABEL = 1


class CategoryAndWordNetIDExtractor(IterDataPipe):
    # Although the WordNet IDs (wnids) are unique, the corresponding categories are not. For example, both n02012849
    # and n03126707 are labeled 'crane' while the first means the bird and the latter means the construction equipment
    _WNID_MAP = {
        "n03126707": "construction crane",
        "n03710721": "tank suit",
    }

    def __init__(self, datapipe: IterDataPipe[Tuple[str, BinaryIO]]) -> None:
        self.datapipe = datapipe

    def __iter__(self) -> Iterator[Tuple[str, str]]:
        for _, stream in self.datapipe:
            synsets = read_mat(stream, squeeze_me=True)["synsets"]
            for _, wnid, category, _, num_children, *_ in synsets:
                if num_children > 0:
                    # we are looking at a superclass that has no direct instance
                    continue

                yield self._WNID_MAP.get(wnid, category.split(",", 1)[0]), wnid


@register_dataset(NAME)
class ImageNet(Dataset):
    """
    - **homepage**: https://www.image-net.org/
    """

    def __init__(
        self,
        root: Union[str, pathlib.Path],
        *,
        split: str = "train",
        skip_integrity_check: bool = False,
    ) -> None:
        self._split = self._verify_str_arg(split, "split", {"train", "val", "test"})

        info = _info()
        categories, wnids = info["categories"], info["wnids"]
        self._categories = categories
        self._wnids = wnids
        self._wnid_to_category = dict(zip(wnids, categories))

        super().__init__(root, skip_integrity_check=skip_integrity_check)

    _IMAGES_CHECKSUMS = {
        "train": "b08200a27a8e34218a0e58fde36b0fe8f73bc377f4acea2d91602057c3ca45bb",
        "val": "c7e06a6c0baccf06d8dbeb6577d71efff84673a5dbdd50633ab44f8ea0456ae0",
        "test_v10102019": "9cf7f8249639510f17d3d8a0deb47cd22a435886ba8e29e2b3223e65a4079eb4",
    }

    def _resources(self) -> List[OnlineResource]:
        name = "test_v10102019" if self._split == "test" else self._split
        images = ImageNetResource(
            file_name=f"ILSVRC2012_img_{name}.tar",
            sha256=self._IMAGES_CHECKSUMS[name],
        )
        resources: List[OnlineResource] = [images]

        if self._split == "val":
            devkit = ImageNetResource(
                file_name="ILSVRC2012_devkit_t12.tar.gz",
                sha256="b59243268c0d266621fd587d2018f69e906fb22875aca0e295b48cafaa927953",
            )
            resources.append(devkit)

        return resources

    _TRAIN_IMAGE_NAME_PATTERN = re.compile(r"(?P<wnid>n\d{8})_\d+[.]JPEG")

    def _prepare_train_data(self, data: Tuple[str, BinaryIO]) -> Tuple[Tuple[Label, str], Tuple[str, BinaryIO]]:
        path = pathlib.Path(data[0])
        wnid = cast(Match[str], self._TRAIN_IMAGE_NAME_PATTERN.match(path.name))["wnid"]
        label = Label.from_category(self._wnid_to_category[wnid], categories=self._categories)
        return (label, wnid), data

    def _prepare_test_data(self, data: Tuple[str, BinaryIO]) -> Tuple[None, Tuple[str, BinaryIO]]:
        return None, data

    def _classifiy_devkit(self, data: Tuple[str, BinaryIO]) -> Optional[int]:
        return {
            "meta.mat": ImageNetDemux.META,
            "ILSVRC2012_validation_ground_truth.txt": ImageNetDemux.LABEL,
        }.get(pathlib.Path(data[0]).name)

    _VAL_TEST_IMAGE_NAME_PATTERN = re.compile(r"ILSVRC2012_(val|test)_(?P<id>\d{8})[.]JPEG")

    def _val_test_image_key(self, path: pathlib.Path) -> int:
        return int(self._VAL_TEST_IMAGE_NAME_PATTERN.match(path.name)["id"])  # type: ignore[index]

    def _prepare_val_data(
        self, data: Tuple[Tuple[int, str], Tuple[str, BinaryIO]]
    ) -> Tuple[Tuple[Label, str], Tuple[str, BinaryIO]]:
        label_data, image_data = data
        _, wnid = label_data
        label = Label.from_category(self._wnid_to_category[wnid], categories=self._categories)
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

    def _datapipe(self, resource_dps: List[IterDataPipe]) -> IterDataPipe[Dict[str, Any]]:
        if self._split in {"train", "test"}:
            dp = resource_dps[0]

            # the train archive is a tar of tars
            if self._split == "train":
                dp = TarArchiveLoader(dp)

            dp = hint_shuffling(dp)
            dp = hint_sharding(dp)
            dp = Mapper(dp, self._prepare_train_data if self._split == "train" else self._prepare_test_data)
        else:  # config.split == "val":
            images_dp, devkit_dp = resource_dps

            meta_dp, label_dp = Demultiplexer(
                devkit_dp, 2, self._classifiy_devkit, drop_none=True, buffer_size=INFINITE_BUFFER_SIZE
            )

            # We cannot use self._wnids here, since we use a different order than the dataset
            meta_dp = CategoryAndWordNetIDExtractor(meta_dp)
            wnid_dp = Mapper(meta_dp, getitem(1))
            wnid_dp = Enumerator(wnid_dp, 1)
            wnid_map = IterToMapConverter(wnid_dp)

            label_dp = LineReader(label_dp, decode=True, return_path=False)
            label_dp = Mapper(label_dp, int)
            label_dp = Mapper(label_dp, wnid_map.__getitem__)
            label_dp: IterDataPipe[Tuple[int, str]] = Enumerator(label_dp, 1)
            label_dp = hint_shuffling(label_dp)
            label_dp = hint_sharding(label_dp)

            dp = IterKeyZipper(
                label_dp,
                images_dp,
                key_fn=getitem(0),
                ref_key_fn=path_accessor(self._val_test_image_key),
                buffer_size=INFINITE_BUFFER_SIZE,
            )
            dp = Mapper(dp, self._prepare_val_data)

        return Mapper(dp, self._prepare_sample)

    def __len__(self) -> int:
        return {
            "train": 1_281_167,
            "val": 50_000,
            "test": 100_000,
        }[self._split]

    def _filter_meta(self, data: Tuple[str, Any]) -> bool:
        return self._classifiy_devkit(data) == ImageNetDemux.META

    def _generate_categories(self) -> List[Tuple[str, ...]]:
        self._split = "val"
        resources = self._resources()

        devkit_dp = resources[1].load(self._root)
        meta_dp = Filter(devkit_dp, self._filter_meta)
        meta_dp = CategoryAndWordNetIDExtractor(meta_dp)

        categories_and_wnids = cast(List[Tuple[str, ...]], list(meta_dp))
        categories_and_wnids.sort(key=lambda category_and_wnid: category_and_wnid[1])
        return categories_and_wnids
