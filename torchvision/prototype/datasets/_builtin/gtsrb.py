import io
import pathlib
from functools import partial
from typing import Any, Callable, Dict, List, Optional, Tuple

import torch
from torchdata.datapipes.iter import IterDataPipe, Mapper, Filter, CSVDictParser, Zipper, Demultiplexer
from torchvision.prototype.datasets.utils import (
    Dataset,
    DatasetConfig,
    DatasetInfo,
    OnlineResource,
    DatasetType,
    HttpResource,
)
from torchvision.prototype.datasets.utils._internal import (
    path_comparator,
    hint_sharding,
    hint_shuffling,
    INFINITE_BUFFER_SIZE,
)
from torchvision.prototype.features import Label, BoundingBox


class GTSRB(Dataset):
    def _make_info(self) -> DatasetInfo:
        return DatasetInfo(
            "gtsrb",
            type=DatasetType.IMAGE,
            homepage="https://benchmark.ini.rub.de",
            categories=[f"{label:05d}" for label in range(43)],
            valid_options=dict(split=("train", "test")),
        )

    _URL_ROOT = "https://sid.erda.dk/public/archives/daaeac0d7ce1152aea9b61d9f1e19370/"
    _URLS = {
        "train": f"{_URL_ROOT}GTSRB-Training_fixed.zip",
        "test": f"{_URL_ROOT}GTSRB_Final_Test_Images.zip",
        "test_ground_truth": f"{_URL_ROOT}GTSRB_Final_Test_GT.zip",
    }
    _CHECKSUMS = {
        "train": "df4144942083645bd60b594de348aa6930126c3e0e5de09e39611630abf8455a",
        "test": "48ba6fab7e877eb64eaf8de99035b0aaecfbc279bee23e35deca4ac1d0a837fa",
        "test_ground_truth": "f94e5a7614d75845c74c04ddb26b8796b9e483f43541dd95dd5b726504e16d6d",
    }

    def resources(self, config: DatasetConfig) -> List[OnlineResource]:
        rsrcs: List[OnlineResource] = [HttpResource(self._URLS[config.split], sha256=self._CHECKSUMS[config.split])]

        if config.split == "test":
            rsrcs.append(
                HttpResource(
                    self._URLS["test_ground_truth"],
                    sha256=self._CHECKSUMS["test_ground_truth"],
                )
            )

        return rsrcs

    def _classify_train_archive(self, data: Tuple[str, Any]) -> Optional[int]:
        path = pathlib.Path(data[0])
        if path.suffix == ".ppm":
            return 0
        elif path.suffix == ".csv":
            return 1
        else:
            return None

    def _collate_and_decode(
        self, data: Tuple[Tuple[str, Any], Dict[str, Any]], decoder: Optional[Callable[[io.IOBase], torch.Tensor]]
    ) -> Dict[str, Any]:
        (image_path, image_buffer), csv_info = data
        label = int(csv_info["ClassId"])

        bbox = BoundingBox(
            torch.tensor([int(csv_info[k]) for k in ("Roi.X1", "Roi.Y1", "Roi.X2", "Roi.Y2")]),
            format="xyxy",
            image_size=(int(csv_info["Height"]), int(csv_info["Width"])),
        )

        return {
            "image_path": image_path,
            "image": decoder(image_buffer) if decoder else image_buffer,
            "label": Label(label, category=self.categories[label]),
            "bbox": bbox,
        }

    def _make_datapipe(
        self,
        resource_dps: List[IterDataPipe],
        *,
        config: DatasetConfig,
        decoder: Optional[Callable[[io.IOBase], torch.Tensor]],
    ) -> IterDataPipe[Dict[str, Any]]:

        if config.split == "train":
            images_dp, ann_dp = Demultiplexer(
                resource_dps[0], 2, self._classify_train_archive, drop_none=True, buffer_size=INFINITE_BUFFER_SIZE
            )
        else:
            images_dp, ann_dp = resource_dps
            images_dp = Filter(images_dp, path_comparator("suffix", ".ppm"))

        # The order of the image files in the the .zip archives perfectly match the order of the entries in
        # the (possibly concatenated) .csv files. So we're able to use Zipper here instead of a IterKeyZipper.
        ann_dp = CSVDictParser(ann_dp, delimiter=";")
        dp = Zipper(images_dp, ann_dp)

        dp = hint_sharding(dp)
        dp = hint_shuffling(dp)

        dp = Mapper(dp, partial(self._collate_and_decode, decoder=decoder))
        return dp
