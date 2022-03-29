import functools
import pathlib
from typing import Any, Dict, List, Tuple
from urllib.parse import urljoin

from torchdata.datapipes.iter import IterDataPipe, Filter, LineReader, IterKeyZipper, CSVParser, Mapper
from torchvision.prototype.datasets.utils import (
    Dataset,
    DatasetConfig,
    DatasetInfo,
    HttpResource,
    OnlineResource,
)
from torchvision.prototype.datasets.utils._internal import (
    hint_sharding,
    hint_shuffling,
    path_comparator,
    getitem,
    INFINITE_BUFFER_SIZE,
)
from torchvision.prototype.features import EncodedImage, Label


class Places365(Dataset):

    # Mapping the config options into filename and sha256 checksum
    _IMAGES_MAP = {
        ("large", "train"): (
            "train_large_places365standard.tar",
            "dd232ce786836fcb0aa6d30032a95de095e79ed9ff08faeb16309a89eb8f3c97",
        ),
        ("small", "train"): (
            "train_256_places365standard.tar",
            "0ade5d7d38c682aa0203014a2b99ad3bf608bf1de0f151dc0d80efba54660a0a",
        ),
        ("large", "train-challenge"): (
            "train_large_places365challenge.tar",
            "f5ddac4de01865a89768ffe066b8f1abb4f0d7a39cd06770aea0a49e3996d91f",
        ),
        ("small", "train-challenge"): (
            "train_256_places365challenge.tar",
            "250d25cf7da1e69df2470bbc635571728b98301f52eb12ea5f21ca899911666a",
        ),
        ("large", "val"): ("val_large.tar", "ddd71c418592a4c230645e238f9e52de77247461d68cd9a14a080a9ca6f27df6"),
        ("large", "test"): ("test_large.tar", "4fae1d859035fe800a7697c27e5e69d78eb292d4cf12d84798c497b23b46b8e1"),
        ("small", "val"): ("val_256.tar", "24b4e639ef12a0012af525bc4cb443e4ab4aaea8369a1fb009b70e4a4aad5d48"),
        ("small", "test"): ("test_256.tar", "037ee8180369bdde46636341b92900d4bcb8ea000c026a1fd3e0e9827a8702a1"),
    }

    _META_MAP = {
        "standard": (
            "filelist_places365-standard.tar",
            "520699e00d69b63ddc29fac54645aa00ce1c10ca42e288960aa1cf791d6e9aa9",
        ),
        "challenge": (
            "filelist_places365-challenge.tar",
            "9821d8140c77e84bbed92ead80a14c74a2c3a24a1fe7738875b4e318d7655c36",
        ),
    }

    _BASE_URL = "http://data.csail.mit.edu/places/places365/"

    def _make_info(self) -> DatasetInfo:
        return DatasetInfo(
            name="places365",
            homepage="http://places2.csail.mit.edu/index.html",
            valid_options=dict(
                image_res=("large", "small"),
                split=("train", "train-challenge", "val", "test"),
            ),
        )

    def resources(self, config: DatasetConfig) -> List[OnlineResource]:
        variant = "standard"
        if config.split == "train-challenge":
            variant = "challenge"
        meta_filename, meta_sha256 = self._META_MAP[variant]
        meta = HttpResource(urljoin(self._BASE_URL, meta_filename), sha256=meta_sha256)

        img_filename, img_sha256 = self._IMAGES_MAP[(config.image_res, config.split)]
        images = HttpResource(urljoin(self._BASE_URL, img_filename), sha256=img_sha256)

        return [meta, images]

    def _image_key(self, data: Tuple[str, Any], *, split: str, image_res: str) -> str:
        path = pathlib.Path(data[0])
        if split.startswith("train"):
            if image_res == "small":
                root_dir = "data_256"
            else:
                root_dir = "data_large"
            idx = list(reversed(path.parts)).index(root_dir) - 1
            result = f"/{path.relative_to(path.parents[idx]).as_posix()}"
        else:
            result = path.name
        return result

    def _prepare_sample(self, data: Tuple[List[str], Tuple[str, Any]]) -> Dict[str, Any]:
        key_label, (path, buffer) = data
        if len(key_label) == 1:
            # This only happen when split == "test" since test data don't have label
            label = None
        else:
            label = Label(int(key_label[1]), categories=self.categories)
        return dict(
            label=label,
            path=path,
            image=EncodedImage.from_file(buffer),
        )

    def _make_datapipe(
        self, resource_dps: List[IterDataPipe], *, config: DatasetConfig
    ) -> IterDataPipe[Dict[str, Any]]:

        meta_dp, images_dp = resource_dps
        if config.split == "train":
            label_filename = "places365_train_standard.txt"
        else:
            label_filename = f"places365_{config.split.replace('-', '_')}.txt"
        labels_dp = Filter(meta_dp, path_comparator("name", label_filename))
        labels_dp = CSVParser(labels_dp, delimiter=" ")

        labels_dp = hint_sharding(labels_dp)
        labels_dp = hint_shuffling(labels_dp)

        # Join labels_dp with images_dp
        dp = IterKeyZipper(
            labels_dp,
            images_dp,
            key_fn=getitem(0),
            ref_key_fn=functools.partial(self._image_key, split=config.split, image_res=config.image_res),
            buffer_size=INFINITE_BUFFER_SIZE,
        )
        return Mapper(dp, self._prepare_sample)

    def _generate_categories(self, root: pathlib.Path) -> List[str]:
        resources = self.resources(self.info.make_config(variant="standard", image_res="large", split="val"))

        meta_dp = resources[0].load(root)
        categories_dp = Filter(meta_dp, path_comparator("name", "categories_places365.txt"))
        categories_dp = LineReader(categories_dp, decode=True, return_path=False)

        return [posix_path_and_label.split()[0][3:] for posix_path_and_label in categories_dp]
