import csv
import io
import pathlib
from typing import Any, Callable, Dict, List, Optional, Tuple, Sequence

import torch
from torch.utils.data import IterDataPipe
from torch.utils.data.datapipes.iter import (
    Mapper,
    Shuffler,
    Filter,
    ZipArchiveReader,
)
from torchdata.datapipes.iter import KeyZipper
from torchvision.prototype.datasets.utils import (
    Dataset,
    DatasetConfig,
    DatasetInfo,
    GDriveResource,
    OnlineResource,
    DatasetType,
)
from torchvision.prototype.datasets.utils._internal import INFINITE_BUFFER_SIZE


class CelebACSVParser(IterDataPipe):
    def __init__(
        self,
        datapipe,
        *,
        has_header,
    ):
        self.datapipe = datapipe
        self.has_header = has_header
        self._fmtparams = dict(delimiter=" ", skipinitialspace=True)

    def __iter__(self):
        for _, file in self.datapipe:
            file = (line.decode() for line in file)

            if self.has_header:
                # The first row is skipped, because it only contains the number of samples
                next(file)

                # Empty field names are filtered out, because some files have an extr white space after the header
                # line, which is recognized as extra column
                fieldnames = [name for name in next(csv.reader([next(file)], **self._fmtparams)) if name]
                # Some files do not include a label for the image ID column
                if fieldnames[0] != "image_id":
                    fieldnames.insert(0, "image_id")

                for line in csv.DictReader(file, fieldnames=fieldnames, **self._fmtparams):
                    yield line.pop("image_id"), line
            else:
                for line in csv.reader(file, **self._fmtparams):
                    yield line[0], line[1:]


class CelebA(Dataset):
    @property
    def info(self) -> DatasetInfo:
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

    def _filter_split(self, data: Tuple[str, str], *, split):
        _, split_id = data
        return self._SPLIT_ID_TO_NAME[split_id[0]] == split

    def _csv_key(self, data: Tuple[str, Any]) -> str:
        return data[0]

    def _image_key(self, data: Tuple[str, Any]) -> str:
        return pathlib.Path(data[0]).name

    def _name_ann(self, data: Tuple[str, io.IOBase], *, name: str) -> Tuple[str, Dict[str, io.IOBase]]:
        return data[0], {name: data[1]}

    def _collate_partial_anns(self, data):
        (id, anns), (_, partial_anns) = data
        anns.update(partial_anns)
        return id, anns

    def _zip_anns_dp(self, anns_dps: List[IterDataPipe]) -> IterDataPipe:
        anns_dps = {
            name: CelebACSVParser(dp, has_header=has_header)
            for (name, has_header), dp in zip(
                (
                    ("identity", False),
                    ("attributes", True),
                    ("bboxes", True),
                    ("landmarks", True),
                ),
                anns_dps,
            )
        }
        anns_dp: IterDataPipe
        partial_anns_dps: Sequence[IterDataPipe]
        anns_dp, *partial_anns_dps = [
            Mapper(dp, self._name_ann, fn_kwargs=dict(name=name)) for name, dp in anns_dps.items()
        ]
        partial_anns_dp: IterDataPipe
        for partial_anns_dp in partial_anns_dps:
            anns_dp = KeyZipper(anns_dp, partial_anns_dp, lambda data: data[0], buffer_size=INFINITE_BUFFER_SIZE)
            anns_dp = Mapper(anns_dp, self._collate_partial_anns)
        return anns_dp

    def _collate_and_decode_sample(
        self, data, *, decoder: Optional[Callable[[io.IOBase], torch.Tensor]]
    ) -> Dict[str, Any]:
        split_and_image_data, ann_data = data
        _, _, image_data = split_and_image_data
        path, buffer = image_data
        _, ann = ann_data

        image = decoder(buffer) if decoder else buffer

        identity = torch.tensor(int(ann["identity"][0]))
        attributes = {attr: value == "1" for attr, value in ann["attributes"].items()}
        bboxes = torch.tensor([int(ann["bboxes"][key]) for key in ("x_1", "y_1", "width", "height")])
        landmarks = {
            landmark: torch.tensor((int(ann["landmarks"][f"{landmark}_x"]), int(ann["landmarks"][f"{landmark}_y"])))
            for landmark in {key[:-2] for key in ann["landmarks"].keys()}
        }

        return dict(
            path=path,
            image=image,
            identity=identity,
            attributes=attributes,
            bboxes=bboxes,
            landmarks=landmarks,
        )

    def _make_datapipe(
        self,
        resource_dps: List[IterDataPipe],
        *,
        config: DatasetConfig,
        decoder: Optional[Callable[[io.IOBase], torch.Tensor]],
    ) -> IterDataPipe[Dict[str, Any]]:
        splits_dp, images_dp, *anns_dps = resource_dps

        splits_dp = CelebACSVParser(splits_dp, has_header=False)
        splits_dp: IterDataPipe = Filter(splits_dp, self._filter_split, fn_kwargs=dict(split=config.split))
        splits_dp = Shuffler(splits_dp, buffer_size=INFINITE_BUFFER_SIZE)

        images_dp = ZipArchiveReader(images_dp)

        anns_dp = self._zip_anns_dp(anns_dps)

        dp = KeyZipper(
            splits_dp,
            images_dp,
            key_fn=self._csv_key,
            ref_key_fn=self._image_key,
            buffer_size=INFINITE_BUFFER_SIZE,
            keep_key=True,
        )
        dp = KeyZipper(dp, anns_dp, key_fn=self._csv_key, buffer_size=INFINITE_BUFFER_SIZE)
        return Mapper(dp, self._collate_and_decode_sample, fn_kwargs=dict(decoder=decoder))
