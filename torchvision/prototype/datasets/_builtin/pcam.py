import io
from collections import namedtuple
from typing import Any, Dict, List, Optional, Tuple, Iterator

from torchdata.datapipes.iter import IterDataPipe, Mapper, Zipper
from torchvision.prototype import features
from torchvision.prototype.datasets.utils import (
    Dataset,
    DatasetConfig,
    DatasetInfo,
    OnlineResource,
    GDriveResource,
)
from torchvision.prototype.datasets.utils._internal import (
    hint_sharding,
    hint_shuffling,
)
from torchvision.prototype.features import Label


class PCAMH5Reader(IterDataPipe[Tuple[str, io.IOBase]]):
    def __init__(
        self,
        datapipe: IterDataPipe[Tuple[str, io.IOBase]],
        key: Optional[str] = None,  # Note: this key thing might be very specific to the PCAM dataset
    ) -> None:
        self.datapipe = datapipe
        self.key = key

    def __iter__(self) -> Iterator[Tuple[str, io.IOBase]]:
        import h5py

        for _, handle in self.datapipe:
            with h5py.File(handle) as data:
                if self.key is not None:
                    data = data[self.key]
                yield from data


_Resource = namedtuple("_Resource", ("file_name", "gdrive_id", "sha256"))


class PCAM(Dataset):
    def _make_info(self) -> DatasetInfo:
        return DatasetInfo(
            "pcam",
            homepage="https://github.com/basveeling/pcam",
            categories=2,
            valid_options=dict(split=("train", "test", "val")),
            dependencies=["h5py"],
        )

    _RESOURCES = {
        "train": (
            _Resource(  # Images
                file_name="camelyonpatch_level_2_split_train_x.h5.gz",
                gdrive_id="1Ka0XfEMiwgCYPdTI-vv6eUElOBnKFKQ2",
                sha256="d619e741468a7ab35c7e4a75e6821b7e7e6c9411705d45708f2a0efc8960656c",
            ),
            _Resource(  # Targets
                file_name="camelyonpatch_level_2_split_train_y.h5.gz",
                gdrive_id="1269yhu3pZDP8UYFQs-NYs3FPwuK-nGSG",
                sha256="b74126d2c01b20d3661f9b46765d29cf4e4fba6faba29c8e0d09d406331ab75a",
            ),
        ),
        "test": (
            _Resource(  # Images
                file_name="camelyonpatch_level_2_split_test_x.h5.gz",
                gdrive_id="1qV65ZqZvWzuIVthK8eVDhIwrbnsJdbg_",
                sha256="79174c2201ad521602a5888be8f36ee10875f37403dd3f2086caf2182ef87245",
            ),
            _Resource(  # Targets
                file_name="camelyonpatch_level_2_split_test_y.h5.gz",
                gdrive_id="17BHrSrwWKjYsOgTMmoqrIjDy6Fa2o_gP",
                sha256="0a522005fccc8bbd04c5a117bfaf81d8da2676f03a29d7499f71d0a0bd6068ef",
            ),
        ),
        "val": (
            _Resource(  # Images
                file_name="camelyonpatch_level_2_split_valid_x.h5.gz",
                gdrive_id="1hgshYGWK8V-eGRy8LToWJJgDU_rXWVJ3",
                sha256="f82ee1670d027b4ec388048d9eabc2186b77c009655dae76d624c0ecb053ccb2",
            ),
            _Resource(  # Targets
                file_name="camelyonpatch_level_2_split_valid_y.h5.gz",
                gdrive_id="1bH8ZRbhSVAhScTS0p9-ZzGnX91cHT3uO",
                sha256="ce1ae30f08feb468447971cfd0472e7becd0ad96d877c64120c72571439ae48c",
            ),
        ),
    }

    def resources(self, config: DatasetConfig) -> List[OnlineResource]:
        return [  # = [images resource, targets resource]
            GDriveResource(file_name=file_name, id=gdrive_id, sha256=sha256, decompress=True)
            for file_name, gdrive_id, sha256 in self._RESOURCES[config.split]
        ]

    def _prepare_sample(self, data: Tuple[Any, Any]) -> Dict[str, Any]:
        image, target = data  # They're both numpy arrays at this point

        return {
            "image": features.Image(image),
            "label": Label(target.item()),
        }

    def _make_datapipe(
        self, resource_dps: List[IterDataPipe], *, config: DatasetConfig
    ) -> IterDataPipe[Dict[str, Any]]:

        images_dp, targets_dp = resource_dps

        images_dp = PCAMH5Reader(images_dp, key="x")
        targets_dp = PCAMH5Reader(targets_dp, key="y")

        dp = Zipper(images_dp, targets_dp)
        dp = hint_sharding(dp)
        dp = hint_shuffling(dp)
        return Mapper(dp, self._prepare_sample)
