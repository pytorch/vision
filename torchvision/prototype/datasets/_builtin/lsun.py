import functools
import io
import pathlib
import re
from typing import Any, Callable, Dict, List, Optional, Tuple, Iterator

import torch
from torchdata.datapipes.iter import IterDataPipe, Mapper, OnDiskCacheHolder, Concater, IterableWrapper
from torchvision.prototype.datasets.utils import (
    Dataset,
    DatasetConfig,
    DatasetInfo,
    HttpResource,
    OnlineResource,
    DatasetType,
)
from torchvision.prototype.datasets.utils._internal import hint_sharding, hint_shuffling
from torchvision.prototype.features import Label

# We need lmdb.Environment as annotation, but lmdb is an optional requirement at import
try:
    import lmdb

    Environment = lmdb.Environment
except ImportError:
    Environment = Any


class LmdbKeyExtractor(IterDataPipe[Tuple[str, bytes]]):
    def __init__(self, datapipe: IterDataPipe[str]) -> None:
        self.datapipe = datapipe

    def __iter__(self) -> Iterator[Tuple[str, bytes]]:
        import lmdb

        for path in self.datapipe:
            with lmdb.open(path, readonly=True) as env:
                with env.begin(write=False) as txn:
                    keys = b"\n".join(key for key in txn.cursor().iternext(keys=True, values=False))
                    yield path, keys


class LmdbLoader(IterDataPipe[Tuple[Environment, bytes]]):
    def __init__(self, datapipe: IterDataPipe[str]) -> None:
        self.datapipe = datapipe

    def __iter__(self) -> Iterator[Tuple[Environment, bytes]]:  # type: ignore[valid-type]
        import lmdb

        for cache_path in self.datapipe:
            env = lmdb.open(str(pathlib.Path(cache_path).parent), readonly=True)

            with open(cache_path, "rb") as file:
                for key in file:
                    yield env, key.strip()


class LmdbReader(IterDataPipe):
    def __init__(self, datapipe: IterDataPipe[Tuple[Environment, bytes]]):
        self.datapipe = datapipe

    def __iter__(self) -> Iterator[Tuple[str, bytes, io.BytesIO]]:
        for env, key in self.datapipe:
            with env.begin(write=False) as txn:
                yield env.path(), key, io.BytesIO(txn.get(key))


class LsunHttpResource(HttpResource):
    def __init__(self, *args: Any, extract: bool = True, **kwargs: Any) -> None:
        super().__init__(*args, extract=extract, **kwargs)

    def _loader(self, path: pathlib.Path) -> IterDataPipe[str]:
        # LMDB datasets cannot be loaded through an open file handle, but have to be loaded through the path of the
        # parent directory.
        return IterableWrapper([str(next(path.rglob("data.mdb")).parent)])


class Lsun(Dataset):
    def _make_info(self) -> DatasetInfo:
        return DatasetInfo(
            "lsun",
            type=DatasetType.IMAGE,
            categories=(
                "bedroom",
                "bridge",
                "church_outdoor",
                "classroom",
                "conference_room",
                "dining_room",
                "kitchen",
                "living_room",
                "restaurant",
                "tower",
            ),
            valid_options=dict(split=("train", "val", "test")),
            dependencies=("lmdb",),
            homepage="https://www.yf.io/p/lsun",
        )

    _CHECKSUMS = {
        ("train", "bedroom"): "a15644c2e7153106867f0adb3491fc41224102d1f01b24494fb47d9f5d1f174e",
        ("train", "bridge"): "2701b2d421bbc4d8a5e9b0652ff7c3b57cd6495da8e0e85c39533275b5a925a3",
        ("train", "church_outdoor"): "91128ae026840ac0c5982b4445ab5fc4e092d6847cca76793b2b1a0815c2e74a",
        ("train", "classroom"): "73a8a3e318819e1cc602f229673c3f51a68f3ece61e3764ce22df6abea4d0873",
        ("train", "conference_room"): "fa0a4cf72e7acfb103392eaf33640d5508a728e971f31877b01d64e1bde6068c",
        ("train", "dining_room"): "e4ee24c7c309360c3bf019123ce5bbf17434b2ba33abec2b0b07cfae715a52cb",
        ("train", "kitchen"): "b1993cf639aece5d207a27eb9ff872bcca9dff6472d8227a052c79d40ee753c4",
        ("train", "living_room"): "bd2c52b812c80f73ce3062a221396d13a52b5cce2f813b4cdf61937651281d7a",
        ("train", "restaurant"): "11a0a924b960cd0900e9b7477d684fb338bd99cc5f72db1caac592e4f497e09a",
        ("train", "tower"): "440caec74c9641cb51fd235a5970c8e4931c3af875aeb044a8bcca956e106309",
        ("val", "bedroom"): "5d022e781b241c25ec2e1f1f769afcdb8091d7fd58362667aec03137b8114b12",
        ("val", "bridge"): "83216a2974d6068c2e1d18086006e7380ff58540216f955ce87fe049b460cb0d",
        ("val", "church_outdoor"): "34635b7547a3e51a15f942a4a4082dd6bc9cca381a953515cb2275c0eed50584",
        ("val", "classroom"): "5e0e9a375d94091dfe1fa3be87d4a92f41c03f1c0b8e376acc7e05651de512d7",
        ("val", "conference_room"): "927c94df52e10b9b374748c2b83b28b5860e946b3186dfd587985e274834650f",
        ("val", "dining_room"): "bd604d4b91bb5a9611d4e0b85475efd20758390d1a4eb57b53973fcbb5aa8ab6",
        ("val", "kitchen"): "329165f35ec61c4cf49f809246de300b8baad3ffcbda1ac30c27bdd32c84369a",
        ("val", "living_room"): "30a23d9a3db5414e9c97865f60ffb2ee973bfa658a23dbca7188ea514c97c9fc",
        ("val", "restaurant"): "efaa7bcb898ad6cb73b07b89fec3a9c670f4622912eea22fab3986c2cf9a1c20",
        ("val", "tower"): "7f5257847bc01f4e40d4a1b3e24dd8fcd37063f12ca8cf31e726c2ee0b1ae104",
    }

    def resources(self, config: DatasetConfig) -> List[OnlineResource]:
        url_root = "http://dl.yf.io/lsun/scenes"
        if config.split == "test":
            return [
                LsunHttpResource(
                    f"{url_root}/test_lmdb.zip",
                    sha256="5ee4f929363f26d1f3c7db6e40e3f7a8415cf777b3c5527f5f38bf3e9520ff22",
                )
            ]
        else:
            return [
                LsunHttpResource(
                    f"{url_root}/{category}_{config.split}_lmdb.zip",
                    sha256=self._CHECKSUMS[(config.split, category)],
                )
                for category in self.categories
            ]

    _FOLDER_PATTERN = re.compile(r"(?P<category>\w*?)_(?P<split>(train|val))_lmdb")

    def _collate_and_decode_sample(
        self,
        data: Tuple[str, bytes, io.BytesIO],
        *,
        decoder: Optional[Callable[[io.IOBase], torch.Tensor]],
    ) -> Dict[str, Any]:
        path, key, buffer = data

        match = self._FOLDER_PATTERN.match(pathlib.Path(path).parent.name)
        if match:
            category = match["category"]
            label = Label(self.categories.index(category), category=category)
        else:
            label = None

        return dict(
            path=path,
            key=key,
            image=decoder(buffer) if decoder else buffer,
            label=label,
        )

    def _filepath_fn(self, path: str) -> str:
        return str(pathlib.Path(path) / "keys.cache")

    def _make_datapipe(
        self,
        resource_dps: List[IterDataPipe],
        *,
        config: DatasetConfig,
        decoder: Optional[Callable[[io.IOBase], torch.Tensor]],
    ) -> IterDataPipe[Dict[str, Any]]:
        dp = Concater(*resource_dps)

        # LMDB databases are indexed, but extracting all keys is expensive. Since we need them for shuffling, we cache
        # the keys on disk and subsequently only read them from there.
        dp = OnDiskCacheHolder(dp, filepath_fn=self._filepath_fn)
        dp = LmdbKeyExtractor(dp).end_caching(mode="wb", same_filepath_fn=True, skip_read=True)

        dp = LmdbLoader(dp)
        dp = hint_sharding(dp)
        dp = hint_shuffling(dp)
        dp = LmdbReader(dp)
        return Mapper(dp, functools.partial(self._collate_and_decode_sample, decoder=decoder))
