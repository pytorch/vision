import csv
import io
from typing import Any, Callable, Dict, List, Optional, Tuple, Iterator, Sequence

import torch
from torchdata.datapipes.iter import (
    IterDataPipe,
    Mapper,
    Shuffler,
    Filter,
    Zipper,
    IterKeyZipper,
)
from torchvision.prototype.datasets.utils import (
    Dataset,
    DatasetConfig,
    DatasetInfo,
    GDriveResource,
    OnlineResource,
    DatasetType,
)
from torchvision.prototype.datasets.utils._internal import INFINITE_BUFFER_SIZE, getitem, path_accessor


csv.register_dialect("celeba", delimiter=" ", skipinitialspace=True)


class CelebACSVParser(IterDataPipe[Tuple[str, Dict[str, str]]]):
    def __init__(
        self,
        datapipe: IterDataPipe[Tuple[Any, io.IOBase]],
        *,
        fieldnames: Optional[Sequence[str]] = None,
    ) -> None:
        self.datapipe = datapipe
        self.fieldnames = fieldnames

    def __iter__(self) -> Iterator[Tuple[str, Dict[str, str]]]:
        for _, file in self.datapipe:
            file = (line.decode() for line in file)

            if self.fieldnames:
                fieldnames = self.fieldnames
            else:
                # The first row is skipped, because it only contains the number of samples
                next(file)

                # Empty field names are filtered out, because some files have an extra white space after the header
                # line, which is recognized as extra column
                fieldnames = [name for name in next(csv.reader([next(file)], dialect="celeba")) if name]
                # Some files do not include a label for the image ID column
                if fieldnames[0] != "image_id":
                    fieldnames.insert(0, "image_id")

            for line in csv.DictReader(file, fieldnames=fieldnames, dialect="celeba"):
                yield line.pop("image_id"), line


class CelebA(Dataset):
    def _make_info(self) -> DatasetInfo:
        return DatasetInfo(
            "celeba",
            type=DatasetType.IMAGE,
            homepage="https://mmlab.ie.cuhk.edu.hk/projects/CelebA.html",
        )

    def resources(self, config: DatasetConfig) -> List[OnlineResource]:
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
        bboxes = GDriveResource(
            "0B7EVK8r0v71pbThiMVRxWXZ4dU0",
            sha256="7487a82e57c4bb956c5445ae2df4a91ffa717e903c5fa22874ede0820c8ec41b",
            file_name="list_bbox_celeba.txt",
        )
        landmarks = GDriveResource(
            "0B7EVK8r0v71pd0FJY3Blby1HUTQ",
            sha256="6c02a87569907f6db2ba99019085697596730e8129f67a3d61659f198c48d43b",
            file_name="list_landmarks_align_celeba.txt",
        )
        return [splits, images, identities, attributes, bboxes, landmarks]

    _SPLIT_ID_TO_NAME = {
        "0": "train",
        "1": "valid",
        "2": "test",
    }

    def _filter_split(self, data: Tuple[str, Dict[str, str]], *, split: str) -> bool:
        return self._SPLIT_ID_TO_NAME[data[1]["split_id"]] == split

    def _collate_anns(self, data: Tuple[Tuple[str, Dict[str, str]], ...]) -> Tuple[str, Dict[str, Dict[str, str]]]:
        (image_id, identity), (_, attributes), (_, bbox), (_, landmarks) = data
        return image_id, dict(identity=identity, attributes=attributes, bbox=bbox, landmarks=landmarks)

    def _collate_and_decode_sample(
        self,
        data: Tuple[Tuple[str, Tuple[str, List[str]], Tuple[str, io.IOBase]], Tuple[str, Dict[str, Any]]],
        *,
        decoder: Optional[Callable[[io.IOBase], torch.Tensor]],
    ) -> Dict[str, Any]:
        split_and_image_data, ann_data = data
        _, _, image_data = split_and_image_data
        path, buffer = image_data
        _, ann = ann_data

        image = decoder(buffer) if decoder else buffer

        identity = int(ann["identity"]["identity"])
        attributes = {attr: value == "1" for attr, value in ann["attributes"].items()}
        bbox = torch.tensor([int(ann["bbox"][key]) for key in ("x_1", "y_1", "width", "height")])
        landmarks = {
            landmark: torch.tensor((int(ann["landmarks"][f"{landmark}_x"]), int(ann["landmarks"][f"{landmark}_y"])))
            for landmark in {key[:-2] for key in ann["landmarks"].keys()}
        }

        return dict(
            path=path,
            image=image,
            identity=identity,
            attributes=attributes,
            bbox=bbox,
            landmarks=landmarks,
        )

    def _make_datapipe(
        self,
        resource_dps: List[IterDataPipe],
        *,
        config: DatasetConfig,
        decoder: Optional[Callable[[io.IOBase], torch.Tensor]],
    ) -> IterDataPipe[Dict[str, Any]]:
        splits_dp, images_dp, identities_dp, attributes_dp, bboxes_dp, landmarks_dp = resource_dps

        splits_dp = CelebACSVParser(splits_dp, fieldnames=("image_id", "split_id"))
        splits_dp = Filter(splits_dp, self._filter_split, fn_kwargs=dict(split=config.split))
        splits_dp = Shuffler(splits_dp, buffer_size=INFINITE_BUFFER_SIZE)

        anns_dp = Zipper(
            *[
                CelebACSVParser(dp, fieldnames=fieldnames)
                for dp, fieldnames in (
                    (identities_dp, ("image_id", "identity")),
                    (attributes_dp, None),
                    (bboxes_dp, None),
                    (landmarks_dp, None),
                )
            ]
        )
        anns_dp = Mapper(anns_dp, self._collate_anns)

        dp = IterKeyZipper(
            splits_dp,
            images_dp,
            key_fn=getitem(0),
            ref_key_fn=path_accessor("name"),
            buffer_size=INFINITE_BUFFER_SIZE,
            keep_key=True,
        )
        dp = IterKeyZipper(dp, anns_dp, key_fn=getitem(0), buffer_size=INFINITE_BUFFER_SIZE)
        return Mapper(dp, self._collate_and_decode_sample, fn_kwargs=dict(decoder=decoder))
