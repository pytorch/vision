import json
from pathlib import Path
from typing import Callable, Optional

import PIL.Image

from .utils import verify_str_arg, download_and_extract_archive
from .vision import VisionDataset


class Food101(VisionDataset):
    """`The Food-101 Data Set <https://data.vision.ee.ethz.ch/cvl/datasets_extra/food-101/>`_.

    The Food-101 is a challenging data set of 101 food categories, with 101'000 images.
    For each class, 250 manually reviewed test images are provided as well as 750 training images.
    On purpose, the training images were not cleaned, and thus still contain some amount of noise.
    This comes mostly in the form of intense colors and sometimes wrong labels. All images were
    rescaled to have a maximum side length of 512 pixels.


    Args:
        root (string): Root directory of the dataset.
        split (string, optional): The dataset split, supports ``"train"`` (default) and ``"test"``.

        transform (callable, optional): A function/transform that  takes in an PIL image and returns a transformed
            version. E.g, ``transforms.RandomCrop``.
        target_transform (callable, optional): A function/transform that takes in the target and transforms it.
    """

    _URL = "http://data.vision.ee.ethz.ch/cvl/food-101.tar.gz"
    _MD5 = "85eeb15f3717b99a5da872d97d918f87"

    def __init__(
        self,
        root: str,
        split: str = "train",
        download: bool = True,
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
    ) -> None:
        super().__init__(root, transform=transform, target_transform=target_transform)
        self._split = verify_str_arg(split, "split", ("train", "test"))
        self._root_path = Path(self.root)

        if download:
            self._download()

        if not self._check_exists():
            raise RuntimeError("Dataset not found. You can use download=True to download it")

        self._labels = []
        self._image_files = []
        with open(self._meta_folder / f"{split}.json", "r") as f:
            metadata = json.loads(f.read())

        self.classes = sorted(set(metadata.keys()))
        self.class_to_idx = dict(zip(self.classes, range(len(self.classes))))

        for class_label, im_paths in metadata.items():
            self._labels += [self.class_to_idx[class_label]] * len(im_paths)
            self._image_files += im_paths

    def __len__(self) -> int:
        return len(self._image_files)

    def __getitem__(self, idx):
        image_file, label = self._image_files[idx], self._labels[idx]
        image = PIL.Image.open(image_file).convert("RGB")

        if self.transform:
            image = self.transform(image)

        if self.target_transform:
            label = self.target_transform(label)

        return image, label

    def extra_repr(self) -> str:
        return f"split={self._split}"

    @property
    def _base_folder(self):
        return self._root_path / "food-101"

    @property
    def _meta_folder(self) -> str:
        return self._base_folder / "meta"

    @property
    def _images_folder(self) -> str:
        return self._base_folder / "images"

    def _check_exists(self) -> bool:
        return (
            self._base_folder.exists()
            and self._base_folder.is_dir()
            and self._meta_folder.exists()
            and self._meta_folder.is_dir()
            and self._images_folder.exists()
            and self._images_folder.is_dir()
        )

    def _download(self) -> None:
        if self._check_exists():
            return
        download_and_extract_archive(self._URL, download_root=self._base_folder, md5=self._MD5)
