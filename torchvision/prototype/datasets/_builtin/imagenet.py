import io
import pathlib
import re
from typing import Any, Callable, Dict, List, Optional, Tuple

import torch
from torchdata.datapipes.iter import IterDataPipe, LineReader, KeyZipper, Mapper, TarArchiveReader, Filter, Shuffler
from torchvision.prototype.datasets.utils import (
    Dataset,
    DatasetConfig,
    DatasetInfo,
    HttpResource,
    OnlineResource,
    DatasetType,
)
from torchvision.prototype.datasets.utils._internal import (
    create_categories_file,
    INFINITE_BUFFER_SIZE,
    path_comparator,
    Enumerator,
    getitem,
)

HERE = pathlib.Path(__file__).parent


class ImageNet(Dataset):
    @property
    def info(self) -> DatasetInfo:
        return DatasetInfo(
            "imagenet",
            type=DatasetType.IMAGE,
            categories=HERE / "imagenet.categories",
            homepage="https://www.image-net.org/",
            valid_options=dict(split=("train", "val")),
        )

    def resources(self, config: DatasetConfig) -> List[OnlineResource]:
        if config.split == "train":
            images = HttpResource(
                "ILSVRC2012_img_train.tar",
                sha256="b08200a27a8e34218a0e58fde36b0fe8f73bc377f4acea2d91602057c3ca45bb",
            )
        else:  # config.split == "val"
            images = HttpResource(
                "ILSVRC2012_img_val.tar",
                sha256="c7e06a6c0baccf06d8dbeb6577d71efff84673a5dbdd50633ab44f8ea0456ae0",
            )

        devkit = HttpResource(
            "ILSVRC2012_devkit_t12.tar.gz",
            sha256="b59243268c0d266621fd587d2018f69e906fb22875aca0e295b48cafaa927953",
        )

        return [images, devkit]

    _TRAIN_IMAGE_NAME_PATTERN = re.compile(r"(?P<category>n\d{8})_\d+[.]JPEG")

    def _collate_train_data(self, data: Tuple[str, io.IOBase]) -> Tuple[Tuple[None, int], Tuple[str, io.IOBase]]:
        path = pathlib.Path(data[0])
        category = self._TRAIN_IMAGE_NAME_PATTERN.match(path.name).group("category")  # type: ignore[union-attr]
        return (None, self.categories.index(category)), data

    _VAL_IMAGE_NAME_PATTERN = re.compile(r"ILSVRC2012_val_(?P<id>\d{8})[.]JPEG")

    def _val_image_key(self, data: Tuple[str, Any]) -> int:
        path = pathlib.Path(data[0])
        return int(self._VAL_IMAGE_NAME_PATTERN.match(path.name).group("id"))  # type: ignore[union-attr]

    def _collate_and_decode_sample(
        self,
        data: Tuple[Tuple[Optional[int], int], Tuple[str, io.IOBase]],
        *,
        decoder: Optional[Callable[[io.IOBase], torch.Tensor]],
    ) -> Dict[str, Any]:
        label_data, image_data = data
        _, label = label_data
        path, buffer = image_data

        category = self.categories[label]
        label = torch.tensor(label)

        return dict(path=path, image=decoder(buffer) if decoder else buffer, category=category, label=label)

    def _make_datapipe(
        self,
        resource_dps: List[IterDataPipe],
        *,
        config: DatasetConfig,
        decoder: Optional[Callable[[io.IOBase], torch.Tensor]],
    ) -> IterDataPipe[Dict[str, Any]]:
        images_dp, devkit_dp = resource_dps

        images_dp = TarArchiveReader(images_dp)

        if config.split == "train":
            # the train archive is a tar of tars
            dp = TarArchiveReader(images_dp)
            dp = Shuffler(dp, buffer_size=INFINITE_BUFFER_SIZE)
            dp = Mapper(dp, self._collate_train_data)
        else:
            devkit_dp = TarArchiveReader(devkit_dp)
            devkit_dp = Filter(devkit_dp, path_comparator("name", "ILSVRC2012_validation_ground_truth.txt"))
            devkit_dp = LineReader(devkit_dp, return_path=False)
            devkit_dp = Mapper(devkit_dp, int)
            devkit_dp = Enumerator(devkit_dp)
            devkit_dp = Shuffler(devkit_dp, buffer_size=INFINITE_BUFFER_SIZE)

            dp = KeyZipper(
                devkit_dp,
                images_dp,
                key_fn=getitem(0),
                ref_key_fn=self._val_image_key,
                buffer_size=INFINITE_BUFFER_SIZE,
            )

        return Mapper(dp, self._collate_and_decode_sample, fn_kwargs=dict(decoder=decoder))

    def generate_categories_file(self, root):
        resources = self.resources(DatasetConfig(split="train"))
        dp = resources[0].to_datapipe(pathlib.Path(root) / self.name)
        dp = TarArchiveReader(dp)
        categories = sorted(pathlib.Path(path).stem for path, _ in dp)
        create_categories_file(HERE, self.name, categories)


if __name__ == "__main__":
    from torchvision.prototype.datasets import home

    root = home()
    ImageNet().generate_categories_file(root)
