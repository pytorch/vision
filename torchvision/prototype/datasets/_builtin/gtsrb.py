import io
import pathlib
from functools import partial
from typing import Any, Callable, Dict, List, Optional, Tuple

import torch
from torchdata.datapipes.iter import IterDataPipe, Mapper, Filter, IterKeyZipper, CSVDictParser
from torchvision.prototype.datasets.utils import (
    Dataset,
    DatasetConfig,
    DatasetInfo,
    OnlineResource,
    DatasetType,
    HttpResource,
)
from torchvision.prototype.datasets.utils._internal import (
    INFINITE_BUFFER_SIZE,
    path_accessor,
    getitem,
)
from torchvision.prototype.features import Label


class GTSRB(Dataset):
    def _make_info(self) -> DatasetInfo:
        return DatasetInfo(
            "gtsrb",
            type=DatasetType.IMAGE,
            homepage="https://benchmark.ini.rub.de",
            valid_options=dict(split=("train", "test")),
        )

    _URLS = {
        "train": "https://sid.erda.dk/public/archives/daaeac0d7ce1152aea9b61d9f1e19370/GTSRB-Training_fixed.zip",
        "test": "https://sid.erda.dk/public/archives/daaeac0d7ce1152aea9b61d9f1e19370/GTSRB_Final_Test_Images.zip",
        "test_gt": "https://sid.erda.dk/public/archives/daaeac0d7ce1152aea9b61d9f1e19370/GTSRB_Final_Test_GT.zip",
    }
    _CHECKSUMS = {
        "train": "df4144942083645bd60b594de348aa6930126c3e0e5de09e39611630abf8455a",
        "test": "48ba6fab7e877eb64eaf8de99035b0aaecfbc279bee23e35deca4ac1d0a837fa",
        "test_gt": "f94e5a7614d75845c74c04ddb26b8796b9e483f43541dd95dd5b726504e16d6d",
    }

    def resources(self, config: DatasetConfig) -> List[OnlineResource]:
        rsrcs = [HttpResource(self._URLS[config.split], sha256=self._CHECKSUMS[config.split])]

        if config.split == "test":
            rsrcs.append(
                HttpResource(
                    self._URLS["test_gt"],
                    sha256=self._CHECKSUMS["test_gt"],
                )
            )
        return rsrcs

    def _filter_images(self, data: Tuple[str, Any]) -> bool:
        return pathlib.Path(data[0]).suffix == ".ppm"

    def _append_label_train(self, path_and_handle: Tuple[str, Any]):
        path = path_and_handle[0]
        label = int(pathlib.Path(path).parent.stem)
        return *path_and_handle, label

    def _append_label_test(self, path_and_handle, csv_info):
        label = int(csv_info["ClassId"])
        return *path_and_handle, label

    def _collate(self, data, decoder):
        image_path, image_buffer, label = data
        return {"image_path": image_path, "image": decoder(image_buffer), "label": Label(label)}

    def _make_datapipe(
        self,
        resource_dps: List[IterDataPipe],
        *,
        config: DatasetConfig,
        decoder: Optional[Callable[[io.IOBase], torch.Tensor]],
    ) -> IterDataPipe[Dict[str, Any]]:

        if config["split"] == "train":
            images_dp = resource_dps[0]
            images_dp = Filter(images_dp, self._filter_images)
            dp = Mapper(images_dp, self._append_label_train)  # path, handle, label
        else:
            images_dp, gt_dp = resource_dps
            dp = Filter(images_dp, self._filter_images)

            gt_dp = CSVDictParser(gt_dp, fieldnames=("Filename", "ClassId"), delimiter=";")

            dp = IterKeyZipper(
                images_dp,
                gt_dp,
                key_fn=path_accessor("name"),
                ref_key_fn=getitem("Filename"),
                buffer_size=INFINITE_BUFFER_SIZE,
                merge_fn=self._append_label_test,
            )  # path, handle, label

        dp = Mapper(dp, partial(self._collate, decoder=decoder))
        return dp

    def _generate_categories(self, root: pathlib.Path) -> List[str]:
        config = self.default_config

        images_dp = self.resources(config)[0].load(root)
        images_dp = Filter(images_dp, self._filter_images)

        labels = sorted(set(pathlib.Path(path).parent.stem for path, _ in images_dp))
        return labels
