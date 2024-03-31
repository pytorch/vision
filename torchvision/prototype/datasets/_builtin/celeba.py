import csv
import pathlib
from typing import Any, BinaryIO, Dict, Iterator, List, Optional, Sequence, Tuple, Union

import torch
from torchdata.datapipes.iter import Filter, IterDataPipe, IterKeyZipper, Mapper, Zipper
from torchvision.prototype.datasets.utils import Dataset, EncodedImage, GDriveResource, OnlineResource
from torchvision.prototype.datasets.utils._internal import (
    getitem,
    hint_sharding,
    hint_shuffling,
    INFINITE_BUFFER_SIZE,
    path_accessor,
)
from torchvision.prototype.tv_tensors import Label
from torchvision.tv_tensors import BoundingBoxes

from .._api import register_dataset, register_info

csv.register_dialect("celeba", delimiter=" ", skipinitialspace=True)


class CelebACSVParser(IterDataPipe[Tuple[str, Dict[str, str]]]):
    def __init__(
        self,
        datapipe: IterDataPipe[Tuple[Any, BinaryIO]],
        *,
        fieldnames: Optional[Sequence[str]] = None,
    ) -> None:
        self.datapipe = datapipe
        self.fieldnames = fieldnames

    def __iter__(self) -> Iterator[Tuple[str, Dict[str, str]]]:
        for _, file in self.datapipe:
            try:
                lines = (line.decode() for line in file)

                if self.fieldnames:
                    fieldnames = self.fieldnames
                else:
                    # The first row is skipped, because it only contains the number of samples
                    next(lines)

                    # Empty field names are filtered out, because some files have an extra white space after the header
                    # line, which is recognized as extra column
                    fieldnames = [name for name in next(csv.reader([next(lines)], dialect="celeba")) if name]
                    # Some files do not include a label for the image ID column
                    if fieldnames[0] != "image_id":
                        fieldnames.insert(0, "image_id")

                for line in csv.DictReader(lines, fieldnames=fieldnames, dialect="celeba"):
                    yield line.pop("image_id"), line
            finally:
                file.close()


NAME = "celeba"


@register_info(NAME)
def _info() -> Dict[str, Any]:
    return dict()


@register_dataset(NAME)
class CelebA(Dataset):
    """
    - **homepage**: https://mmlab.ie.cuhk.edu.hk/projects/CelebA.html
    """

    def __init__(
        self,
        root: Union[str, pathlib.Path],
        *,
        split: str = "train",
        skip_integrity_check: bool = False,
    ) -> None:
        self._split = self._verify_str_arg(split, "split", ("train", "val", "test"))

        super().__init__(root, skip_integrity_check=skip_integrity_check)

    def _resources(self) -> List[OnlineResource]:
        splits = GDriveResource(
            "0B7EVK8r0v71pY0NSMzRuSXJEVkk",
            sha256="fc955bcb3ef8fbdf7d5640d9a8693a8431b5f2ee291a5c1449a1549e7e073fe7",
            file_name="list_eval_partition.txt",
        )
        images = GDriveResource(
            "0B7EVK8r0v71pZjFTYXZWM3FlRnM",
            sha256="46fb89443c578308acf364d7d379fe1b9efb793042c0af734b6112e4fd3a8c74",
            file_name="img_align_celeba.zip",
        )
        identities = GDriveResource(
            "1_ee_0u7vcNLOfNLegJRHmolfH5ICW-XS",
            sha256="c6143857c3e2630ac2da9f782e9c1232e5e59be993a9d44e8a7916c78a6158c0",
            file_name="identity_CelebA.txt",
        )
        attributes = GDriveResource(
            "0B7EVK8r0v71pblRyaVFSWGxPY0U",
            sha256="f0e5da289d5ccf75ffe8811132694922b60f2af59256ed362afa03fefba324d0",
            file_name="list_attr_celeba.txt",
        )
        bounding_boxes = GDriveResource(
            "0B7EVK8r0v71pbThiMVRxWXZ4dU0",
            sha256="7487a82e57c4bb956c5445ae2df4a91ffa717e903c5fa22874ede0820c8ec41b",
            file_name="list_bbox_celeba.txt",
        )
        landmarks = GDriveResource(
            "0B7EVK8r0v71pd0FJY3Blby1HUTQ",
            sha256="6c02a87569907f6db2ba99019085697596730e8129f67a3d61659f198c48d43b",
            file_name="list_landmarks_align_celeba.txt",
        )
        return [splits, images, identities, attributes, bounding_boxes, landmarks]

    def _filter_split(self, data: Tuple[str, Dict[str, str]]) -> bool:
        split_id = {
            "train": "0",
            "val": "1",
            "test": "2",
        }[self._split]
        return data[1]["split_id"] == split_id

    def _prepare_sample(
        self,
        data: Tuple[
            Tuple[str, Tuple[Tuple[str, List[str]], Tuple[str, BinaryIO]]],
            Tuple[
                Tuple[str, Dict[str, str]],
                Tuple[str, Dict[str, str]],
                Tuple[str, Dict[str, str]],
                Tuple[str, Dict[str, str]],
            ],
        ],
    ) -> Dict[str, Any]:
        split_and_image_data, ann_data = data
        _, (_, image_data) = split_and_image_data
        path, buffer = image_data

        image = EncodedImage.from_file(buffer)
        (_, identity), (_, attributes), (_, bounding_boxes), (_, landmarks) = ann_data

        return dict(
            path=path,
            image=image,
            identity=Label(int(identity["identity"])),
            attributes={attr: value == "1" for attr, value in attributes.items()},
            bounding_boxes=BoundingBoxes(
                [int(bounding_boxes[key]) for key in ("x_1", "y_1", "width", "height")],
                format="xywh",
                spatial_size=image.spatial_size,
            ),
            landmarks={
                landmark: torch.tensor((int(landmarks[f"{landmark}_x"]), int(landmarks[f"{landmark}_y"])))
                for landmark in {key[:-2] for key in landmarks.keys()}
            },
        )

    def _datapipe(self, resource_dps: List[IterDataPipe]) -> IterDataPipe[Dict[str, Any]]:
        splits_dp, images_dp, identities_dp, attributes_dp, bounding_boxes_dp, landmarks_dp = resource_dps

        splits_dp = CelebACSVParser(splits_dp, fieldnames=("image_id", "split_id"))
        splits_dp = Filter(splits_dp, self._filter_split)
        splits_dp = hint_shuffling(splits_dp)
        splits_dp = hint_sharding(splits_dp)

        anns_dp = Zipper(
            *[
                CelebACSVParser(dp, fieldnames=fieldnames)
                for dp, fieldnames in (
                    (identities_dp, ("image_id", "identity")),
                    (attributes_dp, None),
                    (bounding_boxes_dp, None),
                    (landmarks_dp, None),
                )
            ]
        )

        dp = IterKeyZipper(
            splits_dp,
            images_dp,
            key_fn=getitem(0),
            ref_key_fn=path_accessor("name"),
            buffer_size=INFINITE_BUFFER_SIZE,
            keep_key=True,
        )
        dp = IterKeyZipper(
            dp,
            anns_dp,
            key_fn=getitem(0),
            ref_key_fn=getitem(0, 0),
            buffer_size=INFINITE_BUFFER_SIZE,
        )
        return Mapper(dp, self._prepare_sample)

    def __len__(self) -> int:
        return {
            "train": 162_770,
            "val": 19_867,
            "test": 19_962,
        }[self._split]
