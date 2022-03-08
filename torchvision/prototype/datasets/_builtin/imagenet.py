import pathlib
import re
from typing import Any, Dict, List, Optional, Tuple, BinaryIO, Match, cast, Union

from torchdata.datapipes.iter import IterDataPipe, LineReader, IterKeyZipper, Mapper, Filter, Demultiplexer
from torchdata.datapipes.iter import TarArchiveReader
from torchvision.datasets.utils import verify_str_arg
from torchvision.prototype.datasets.utils import (
    DatasetInfo,
    ManualDownloadResource,
)
from torchvision.prototype.datasets.utils._internal import (
    INFINITE_BUFFER_SIZE,
    BUILTIN_DIR,
    path_comparator,
    Enumerator,
    getitem,
    read_mat,
    hint_sharding,
    hint_shuffling,
    path_accessor,
    TakerDataPipe,
)
from torchvision.prototype.features import Label, EncodedImage

from .._api import register_dataset, register_info

NAME = "imagenet"

CATEGORIES, WNIDS = zip(*DatasetInfo.read_categories_file(BUILTIN_DIR / f"{NAME}.categories"))
WNID_TO_CATEGORY = dict(zip(WNIDS, CATEGORIES))


@register_info(NAME)
def info() -> Dict[str, Any]:
    return dict(categories=CATEGORIES, wnids=WNIDS)


class ImageNetResource(ManualDownloadResource):
    def __init__(self, **kwargs: Any) -> None:
        super().__init__("Register on https://image-net.org/ and follow the instructions there.", **kwargs)


def load_images_dp(root: Union[str, pathlib.Path], *, split: str, **kwargs: Any) -> IterDataPipe[Tuple[str, BinaryIO]]:
    name = "test_v10102019" if split == "test" else split
    return ImageNetResource(
        file_name=f"ILSVRC2012_img_{name}.tar",
        sha256={
            "train": "b08200a27a8e34218a0e58fde36b0fe8f73bc377f4acea2d91602057c3ca45bb",
            "val": "c7e06a6c0baccf06d8dbeb6577d71efff84673a5dbdd50633ab44f8ea0456ae0",
            "test_v10102019": "9cf7f8249639510f17d3d8a0deb47cd22a435886ba8e29e2b3223e65a4079eb4",
        }[name],
    ).load(root, **kwargs)


def load_devkit_dp(root: Union[str, pathlib.Path], **kwargs: Any) -> IterDataPipe[Tuple[str, BinaryIO]]:
    return ImageNetResource(
        file_name="ILSVRC2012_devkit_t12.tar.gz",
        sha256="b59243268c0d266621fd587d2018f69e906fb22875aca0e295b48cafaa927953",
    ).load(root, **kwargs)


TRAIN_IMAGE_NAME_PATTERN = re.compile(r"(?P<wnid>n\d{8})_\d+[.]JPEG")


def prepare_train_data(data: Tuple[str, BinaryIO]) -> Tuple[Tuple[Label, str], Tuple[str, BinaryIO]]:
    path = pathlib.Path(data[0])
    wnid = cast(Match[str], TRAIN_IMAGE_NAME_PATTERN.match(path.name))["wnid"]
    label = Label.from_category(WNID_TO_CATEGORY[wnid], categories=CATEGORIES)
    return (label, wnid), data


def prepare_test_data(data: Tuple[str, BinaryIO]) -> Tuple[None, Tuple[str, BinaryIO]]:
    return None, data


def classifiy_devkit(data: Tuple[str, BinaryIO]) -> Optional[int]:
    return {
        "meta.mat": 0,
        "ILSVRC2012_validation_ground_truth.txt": 1,
    }.get(pathlib.Path(data[0]).name)


# Although the WordNet IDs (wnids) are unique, the corresponding human-readable categories are not. For example, both
# 'n02012849' and 'n03126707' are labeled 'crane' while the first means the bird and the latter means the construction
# equipment.
WNID_MAP = {
    "n03126707": "construction crane",
    "n03710721": "tank suit",
}


def extract_categories_and_wnids(data: Tuple[str, BinaryIO]) -> List[Tuple[str, str]]:
    synsets = read_mat(data[1], squeeze_me=True)["synsets"]
    return [
        (WNID_MAP.get(wnid, category.split(",", 1)[0]), wnid)
        for _, wnid, category, _, num_children, *_ in synsets
        # if num_children > 0, we are looking at a superclass that has no direct instance
        if num_children == 0
    ]


def imagenet_label_to_wnid(imagenet_label: str) -> str:
    return cast(Tuple[str, ...], WNIDS)[int(imagenet_label) - 1]


VAL_TEST_IMAGE_NAME_PATTERN = re.compile(r"ILSVRC2012_(val|test)_(?P<id>\d{8})[.]JPEG")


def val_test_image_key(path: pathlib.Path) -> int:
    return int(VAL_TEST_IMAGE_NAME_PATTERN.match(path.name)["id"])  # type: ignore[index]


def prepare_val_data(
    data: Tuple[Tuple[int, str], Tuple[str, BinaryIO]]
) -> Tuple[Tuple[Label, str], Tuple[str, BinaryIO]]:
    label_data, image_data = data
    _, wnid = label_data
    label = Label.from_category(WNID_TO_CATEGORY[wnid], categories=CATEGORIES)
    return (label, wnid), image_data


def prepare_sample(
    data: Tuple[Optional[Tuple[Label, str]], Tuple[str, BinaryIO]],
) -> Dict[str, Any]:
    label_data, (path, buffer) = data

    return dict(
        dict(zip(("label", "wnid"), label_data if label_data else (None, None))),
        path=path,
        image=EncodedImage.from_file(buffer),
    )


@register_dataset(NAME)
def imagenet(root: Union[str, pathlib.Path], *, split: str = "train", **kwargs: Any) -> TakerDataPipe:
    verify_str_arg(split, "split", ["train", "val", "test"])

    images_dp = load_images_dp(root, split=split, **kwargs)
    if split == "train":
        # the train archive is a tar of tars
        images_dp = TarArchiveReader(images_dp)
        images_dp = hint_sharding(images_dp)
        images_dp = hint_shuffling(images_dp)
        dp = Mapper(images_dp, prepare_train_data)
    elif split == "test":
        images_dp = hint_sharding(images_dp)
        images_dp = hint_shuffling(images_dp)
        dp = Mapper(images_dp, prepare_test_data)
    else:  # split == "val"
        devkit_dp = load_devkit_dp(root, **kwargs)

        meta_dp, label_dp = Demultiplexer(
            devkit_dp, 2, classifiy_devkit, drop_none=True, buffer_size=INFINITE_BUFFER_SIZE
        )

        meta_dp = Mapper(meta_dp, extract_categories_and_wnids)
        _, wnids = zip(*next(iter(meta_dp)))

        label_dp = LineReader(label_dp, decode=True, return_path=False)
        label_dp = Mapper(label_dp, imagenet_label_to_wnid)
        label_dp: IterDataPipe[Tuple[int, str]] = Enumerator(label_dp, 1)
        label_dp = hint_sharding(label_dp)
        label_dp = hint_shuffling(label_dp)

        dp = IterKeyZipper(
            label_dp,
            images_dp,
            key_fn=getitem(0),
            ref_key_fn=path_accessor(val_test_image_key),
            buffer_size=INFINITE_BUFFER_SIZE,
        )
        dp = Mapper(dp, prepare_val_data)

    dp = Mapper(dp, prepare_sample)
    return TakerDataPipe(
        dp,
        num_take={
            "train": 1_281_167,
            "val": 50_000,
            "test": 100_000,
        }[split],
    )


def generate_categories(root: Union[str, pathlib.Path], **kwargs: Any) -> List[Tuple[str, ...]]:
    devkit_dp = load_devkit_dp(root, **kwargs)

    meta_dp = Filter(devkit_dp, path_comparator("name", "meta.mat"))
    meta_dp = Mapper(meta_dp, extract_categories_and_wnids)

    categories_and_wnids = cast(List[Tuple[str, ...]], next(iter(meta_dp)))
    categories_and_wnids.sort(key=lambda category_and_wnid: category_and_wnid[1])
    return categories_and_wnids
