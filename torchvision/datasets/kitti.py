import os
from collections import namedtuple
from typing import Any, Callable, NamedTuple, Optional, Tuple
from urllib.error import URLError

import pandas as pd
from PIL import Image

from .utils import download_and_extract_archive
from .vision import VisionDataset


class Kitti(VisionDataset):
    """`KITTI <http://www.cvlibs.net/datasets/kitti>`_ Dataset.
    Args:
        root (string): Root directory where images are downloaded to.
            Expects the following folder structure if download=False:

            .. code::

                <root>
                 └─ Kitti
                     └─ raw
                         ├── training
                         |      ├── image_2
                         |      └── label_2
                         └── testing
                                └── image_2
        split (string): The dataset split to use. One of {``train``, ``test``}.
            Defaults to ``train``.
        transform (callable, optional): A function/transform that  takes in an PIL image
            and returns a transformed version. E.g, ``transforms.ToTensor``
        target_transform (callable, optional): A function/transform that takes in the
            target and transforms it.
        transforms (callable, optional): A function/transform that takes input sample
            and its target as entry and returns a transformed version.
        download (bool, optional): If true, downloads the dataset from the internet and
            puts it in root directory. If dataset is already downloaded, it is not
            downloaded again.
    """

    mirrors = [
        "https://s3.eu-central-1.amazonaws.com/avg-kitti/",
    ]
    resources = [
        "data_object_image_2.zip",
        "data_object_label_2.zip",
    ]
    image_dir_name = "image_2"
    labels_dir_name = "label_2"

    def __init__(
        self,
        root: str,
        split: str = None,
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
        transforms: Optional[Callable] = None,
        download: bool = False,
    ):
        super().__init__(
            root,
            transform=transform,
            target_transform=target_transform,
            transforms=transforms,
        )
        self.TargetTuple = namedtuple(
            "TargetTuple",
            [
                "type",
                "truncated",
                "occluded",
                "alpha",
                "bbox",
                "dimensions",
                "location",
                "rotation_y",
            ],
        )
        self.images = []
        self.targets = []
        self.root = root
        self.split = split
        self.transform = transform
        self.target_transform = target_transform
        self.transforms = transforms

        if download:
            self.download()
        if not self._check_exists():
            raise RuntimeError(
                "Dataset not found. You may use download=True to download it."
            )

        location = "testing" if self.split == "test" else "training"
        image_dir = os.path.join(self.raw_folder, location, self.image_dir_name)
        if location == "training":
            labels_dir = os.path.join(self.raw_folder, location, self.labels_dir_name)
        for img_file in os.listdir(image_dir):
            self.images.append(os.path.join(image_dir, img_file))
            if location == "training":
                self.targets.append(
                    os.path.join(labels_dir, f"{img_file.split('.')[0]}.txt")
                )

    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        """Get item at a given index.
        Args:
            index (int): Index
        Returns:
            tuple: (image, target), where
            target is a namedtuple with the following fields:
                type: Int64Tensor[N]
                truncated: FloatTensor[N]
                occluded: Int64Tensor[N]
                alpha: FloatTensor[N]
                bbox: FloatTensor[N, 4]
                dimensions: FloatTensor[N, 3]
                locations: FloatTensor[N, 3]
                rotation_y: FloatTensor[N]
                score: FloatTensor[N]
        """
        image = Image.open(self.images[index])
        target = None if self.split == "test" else self._parse_target(index)
        if self.transforms:
            image, target = self.transforms(image, target)
        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            target = self.target_transform(target)
        return image, target

    def _parse_target(self, index: int) -> NamedTuple:
        target_df = pd.read_csv(self.targets[index], delimiter=" ", header=None)
        return self.TargetTuple(
            type=target_df.iloc[:, 0].values,
            truncated=target_df.iloc[:, 1].values,
            occluded=target_df.iloc[:, 2].values,
            alpha=target_df.iloc[:, 3].values,
            bbox=target_df.iloc[:, 4:8].values,
            dimensions=target_df.iloc[:, 8:11].values,
            location=target_df.iloc[:, 11:14].values,
            rotation_y=target_df.iloc[:, 14].values,
        )

    def __len__(self) -> int:
        return len(self.images)

    @property
    def raw_folder(self) -> str:
        return os.path.join(self.root, self.__class__.__name__, "raw")

    def _check_exists(self) -> bool:
        """Check if the data directory exists."""
        location = "testing" if self.split == "test" else "training"
        folders = [self.image_dir_name]
        if self.split != "test":
            folders.append(self.labels_dir_name)
        return all(
            os.path.isdir(os.path.join(self.raw_folder, location, fname))
            for fname in folders
        )

    def download(self) -> None:
        """Download the KITTI data if it doesn't exist already."""

        if self._check_exists():
            return

        os.makedirs(self.raw_folder, exist_ok=True)

        # download files
        for fname in self.resources:
            for mirror in self.mirrors:
                url = f"{mirror}{fname}"
                try:
                    print(f"Downloading {url}")
                    download_and_extract_archive(
                        url=url,
                        download_root=self.raw_folder,
                        filename=fname,
                    )
                except URLError as error:
                    print(f"Error downloading {fname}: {error}")
