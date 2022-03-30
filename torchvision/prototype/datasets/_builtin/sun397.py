import pathlib
from typing import Any, Dict, List, Tuple

from torchdata.datapipes.iter import IterDataPipe, Mapper, Filter, LineReader, IterKeyZipper
from torchvision.prototype.datasets.utils import Dataset, DatasetConfig, DatasetInfo, HttpResource, OnlineResource
from torchvision.prototype.datasets.utils._internal import (
    path_comparator,
    hint_sharding,
    hint_shuffling,
    INFINITE_BUFFER_SIZE,
    getitem,
)
from torchvision.prototype.features import EncodedImage, Label


class SUN397(Dataset):
    def _make_info(self) -> DatasetInfo:
        # split can be either "all" where it get all the images (suitable for those who want custom data split)
        # or it can be "train-{fold}" or "test-{fold}" where fold is a number between 1 to 10, this is suitable for
        # those who want to reproduce the paper's result
        split_choices = ["all"]
        split_choices.extend([f"{split_type}-{fold}" for split_type in ["train", "test"] for fold in range(1, 11)])

        # The default split = "all" for backward compatibility with previous datasets API
        return DatasetInfo(
            "sun397",
            homepage="https://vision.princeton.edu/projects/2010/SUN/",
            valid_options=dict(
                split=split_choices,
            ),
        )

    def resources(self, config: DatasetConfig) -> List[OnlineResource]:
        images = HttpResource(
            "http://vision.princeton.edu/projects/2010/SUN/SUN397.tar.gz",
            sha256="f404130965a7ad77bed5ececc71e720ab50f3cc1e1bb257257610d38f3b928ec",
            preprocess="decompress",
        )
        splits = HttpResource(
            "https://vision.princeton.edu/projects/2010/SUN/download/Partitions.zip",
            sha256="e3311ea5f812d4cc6faef194688343d1111f6cd6f70f2c62f393c2b8d6ba1ec9",
        )
        return [images, splits]

    def _image_key(self, data: Tuple[str, Any]) -> str:
        path = pathlib.Path(data[0])
        idx = list(reversed(path.parts)).index("SUN397") - 1
        return f"/{path.relative_to(path.parents[idx]).as_posix()}"

    def _prepare_sample(self, data: Tuple[str, Tuple[str, Any]]) -> Dict[str, Any]:
        key, (path, buffer) = data
        category = "/".join(key.split("/")[2:-1])
        return dict(
            label=Label.from_category(category, categories=self.categories),
            path=path,
            image=EncodedImage.from_file(buffer),
        )

    def _prepare_sample_all(self, data: Tuple[str, Any]) -> Dict[str, Any]:
        path, buffer = data
        key = self._image_key(data)
        category = "/".join(key.split("/")[2:-1])
        return dict(
            label=Label.from_category(category, categories=self.categories),
            path=path,
            image=EncodedImage.from_file(buffer),
        )

    def _make_datapipe(
        self, resource_dps: List[IterDataPipe], *, config: DatasetConfig
    ) -> IterDataPipe[Dict[str, Any]]:
        images_dp, splits_dp = resource_dps

        images_dp = Filter(images_dp, path_comparator("suffix", ".jpg"))
        if config.split == "all":
            dp = images_dp
            return Mapper(dp, self._prepare_sample_all)
        else:
            split_type, fold = config.split.split("-")
            splits_dp = Filter(
                splits_dp,
                path_comparator("name", f"{split_type.capitalize()}ing_{int(fold):02d}.txt"),
            )
            splits_dp = LineReader(splits_dp, decode=True, return_path=False)
            splits_dp = hint_sharding(splits_dp)
            splits_dp = hint_shuffling(splits_dp)

            dp = IterKeyZipper(
                splits_dp,
                images_dp,
                key_fn=getitem(),
                ref_key_fn=self._image_key,
                buffer_size=INFINITE_BUFFER_SIZE,
            )
            return Mapper(dp, self._prepare_sample)

    def _generate_categories(self, root: pathlib.Path) -> List[str]:
        resources = self.resources(self.default_config)
        meta_dp = resources[1].load(root)
        categories_dp = Filter(meta_dp, path_comparator("name", "ClassName.txt"))
        categories_dp = LineReader(categories_dp, decode=True, return_path=False)
        return [category[3:] for category in categories_dp]
