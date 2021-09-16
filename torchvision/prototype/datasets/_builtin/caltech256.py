import io
import pathlib
from typing import Any, Callable, Dict, List, Optional, Tuple

import torch
from torch.utils.data import IterDataPipe
from torch.utils.data.datapipes.iter import (
    Mapper,
    TarArchiveReader,
)

from torchvision.prototype.datasets.utils import (
    Dataset,
    DatasetConfig,
    DatasetInfo,
    HttpResource,
)
from torchvision.prototype.datasets.utils._internal import create_categories_file

__all__ = ["Caltech256"]

HERE = pathlib.Path(__file__).parent


class Caltech256(Dataset):
    @property
    def info(self) -> DatasetInfo:
        return DatasetInfo(
            "caltech256",
            categories=HERE / "caltech256.categories",
            homepage="http://www.vision.caltech.edu/Image_Datasets/Caltech256",
        )

    def resources(self, config: DatasetConfig) -> List[HttpResource]:
        return [
            HttpResource(
                "http://www.vision.caltech.edu/Image_Datasets/Caltech256/256_ObjectCategories.tar",
                sha256="08ff01b03c65566014ae88eb0490dbe4419fc7ac4de726ee1163e39fd809543e",
            )
        ]

    def _collate_and_decode_sample(
        self,
        data: Tuple[str, io.BufferedIOBase],
        *,
        decoder: Optional[Callable[[str, io.BufferedIOBase], torch.Tensor]],
    ) -> Dict[str, Any]:
        path, image = data

        dir_name = pathlib.Path(path).parent.name
        label_str, category = dir_name.split(".")
        label = torch.tensor(int(label_str))

        if decoder:
            image = decoder(path, image)

        return dict(label=label, category=category, image=image)

    def _make_datapipe(
        self,
        resource_dps: List[IterDataPipe],
        *,
        config: DatasetConfig,
        decoder: Optional[Callable[[str, io.BufferedIOBase], torch.Tensor]],
    ) -> IterDataPipe[Dict[str, Any]]:
        dp = resource_dps[0]
        dp = TarArchiveReader(dp)
        return Mapper(
            dp, self._collate_and_decode_sample, fn_kwargs=dict(decoder=decoder)
        )

    def generate_categories_file(self, root):
        dp = self.resources(self.default_config)[0].to_datapipe(
            pathlib.Path(root) / self.name
        )
        dp = TarArchiveReader(dp)
        dir_names = {pathlib.Path(path).parent.name for path, _ in dp}
        categories = [name.split(".")[1] for name in sorted(dir_names)]
        create_categories_file(HERE, self.name, categories)


if __name__ == "__main__":
    from torchvision.prototype.datasets import home

    Caltech256().generate_categories_file(home())
