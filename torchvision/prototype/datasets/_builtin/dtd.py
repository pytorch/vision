import enum
import pathlib
from typing import Any, BinaryIO, Dict, List, Optional, Tuple, Union

from torchdata.datapipes.iter import CSVParser, Demultiplexer, Filter, IterDataPipe, IterKeyZipper, LineReader, Mapper
from torchvision.prototype.datasets.utils import Dataset, EncodedImage, HttpResource, OnlineResource
from torchvision.prototype.datasets.utils._internal import (
    getitem,
    hint_sharding,
    hint_shuffling,
    INFINITE_BUFFER_SIZE,
    path_comparator,
    read_categories_file,
)
from torchvision.prototype.tv_tensors import Label

from .._api import register_dataset, register_info


NAME = "dtd"


class DTDDemux(enum.IntEnum):
    SPLIT = 0
    JOINT_CATEGORIES = 1
    IMAGES = 2


@register_info(NAME)
def _info() -> Dict[str, Any]:
    return dict(categories=read_categories_file(NAME))


@register_dataset(NAME)
class DTD(Dataset):
    """DTD Dataset.
    homepage="https://www.robots.ox.ac.uk/~vgg/data/dtd/",
    """

    def __init__(
        self,
        root: Union[str, pathlib.Path],
        *,
        split: str = "train",
        fold: int = 1,
        skip_validation_check: bool = False,
    ) -> None:
        self._split = self._verify_str_arg(split, "split", {"train", "val", "test"})

        if not (1 <= fold <= 10):
            raise ValueError(f"The fold parameter should be an integer in [1, 10]. Got {fold}")
        self._fold = fold

        self._categories = _info()["categories"]

        super().__init__(root, skip_integrity_check=skip_validation_check)

    def _resources(self) -> List[OnlineResource]:
        archive = HttpResource(
            "https://www.robots.ox.ac.uk/~vgg/data/dtd/download/dtd-r1.0.1.tar.gz",
            sha256="e42855a52a4950a3b59612834602aa253914755c95b0cff9ead6d07395f8e205",
            preprocess="decompress",
        )
        return [archive]

    def _classify_archive(self, data: Tuple[str, Any]) -> Optional[int]:
        path = pathlib.Path(data[0])
        if path.parent.name == "labels":
            if path.name == "labels_joint_anno.txt":
                return DTDDemux.JOINT_CATEGORIES

            return DTDDemux.SPLIT
        elif path.parents[1].name == "images":
            return DTDDemux.IMAGES
        else:
            return None

    def _image_key_fn(self, data: Tuple[str, Any]) -> str:
        path = pathlib.Path(data[0])
        # The split files contain hardcoded posix paths for the images, e.g. banded/banded_0001.jpg
        return str(path.relative_to(path.parents[1]).as_posix())

    def _prepare_sample(self, data: Tuple[Tuple[str, List[str]], Tuple[str, BinaryIO]]) -> Dict[str, Any]:
        (_, joint_categories_data), image_data = data
        _, *joint_categories = joint_categories_data
        path, buffer = image_data

        category = pathlib.Path(path).parent.name

        return dict(
            joint_categories={category for category in joint_categories if category},
            label=Label.from_category(category, categories=self._categories),
            path=path,
            image=EncodedImage.from_file(buffer),
        )

    def _datapipe(self, resource_dps: List[IterDataPipe]) -> IterDataPipe[Dict[str, Any]]:
        archive_dp = resource_dps[0]

        splits_dp, joint_categories_dp, images_dp = Demultiplexer(
            archive_dp, 3, self._classify_archive, drop_none=True, buffer_size=INFINITE_BUFFER_SIZE
        )

        splits_dp = Filter(splits_dp, path_comparator("name", f"{self._split}{self._fold}.txt"))
        splits_dp = LineReader(splits_dp, decode=True, return_path=False)
        splits_dp = hint_shuffling(splits_dp)
        splits_dp = hint_sharding(splits_dp)

        joint_categories_dp = CSVParser(joint_categories_dp, delimiter=" ")

        dp = IterKeyZipper(
            splits_dp,
            joint_categories_dp,
            key_fn=getitem(),
            ref_key_fn=getitem(0),
            buffer_size=INFINITE_BUFFER_SIZE,
        )
        dp = IterKeyZipper(
            dp,
            images_dp,
            key_fn=getitem(0),
            ref_key_fn=self._image_key_fn,
            buffer_size=INFINITE_BUFFER_SIZE,
        )
        return Mapper(dp, self._prepare_sample)

    def _filter_images(self, data: Tuple[str, Any]) -> bool:
        return self._classify_archive(data) == DTDDemux.IMAGES

    def _generate_categories(self) -> List[str]:
        resources = self._resources()

        dp = resources[0].load(self._root)
        dp = Filter(dp, self._filter_images)

        return sorted({pathlib.Path(path).parent.name for path, _ in dp})

    def __len__(self) -> int:
        return 1_880  # All splits have the same length
