import csv
import os
from typing import Any, Callable, Optional, Tuple

import PIL

from .folder import make_dataset
from .utils import download_and_extract_archive
from .vision import VisionDataset


class GTSRB(VisionDataset):
    """`German Traffic Sign Recognition Benchmark (GTSRB) <https://benchmark.ini.rub.de/>`_ Dataset.

    Args:
        root (string): Root directory of the dataset.
        train (bool, optional): If True, creates dataset from training set, otherwise
            creates from test set.
        transform (callable, optional): A function/transform that  takes in an PIL image and returns a transformed
            version. E.g, ``transforms.RandomCrop``.
        target_transform (callable, optional): A function/transform that takes in the target and transforms it.
        download (bool, optional): If True, downloads the dataset from the internet and
            puts it in root directory. If dataset is already downloaded, it is not
            downloaded again.
    """

    # Ground Truth for the test set
    gt_url = "https://sid.erda.dk/public/archives/daaeac0d7ce1152aea9b61d9f1e19370/GTSRB_Final_Test_GT.zip"
    gt_csv = "GT-final_test.csv"
    gt_md5 = "fe31e9c9270bbcd7b84b7f21a9d9d9e5"

    # URLs for the test and train set
    urls = (
        "https://sid.erda.dk/public/archives/daaeac0d7ce1152aea9b61d9f1e19370/GTSRB_Final_Test_Images.zip",
        "https://sid.erda.dk/public/archives/daaeac0d7ce1152aea9b61d9f1e19370/GTSRB-Training_fixed.zip",
    )

    md5 = ("c7e4e6327067d32654124b0fe9e82185", "513f3c79a4c5141765e10e952eaa2478")

    extension = (".ppm",)

    def __init__(
        self,
        root: str,
        train: bool = True,
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
        download: bool = False,
    ) -> None:

        super().__init__(root, transform=transform, target_transform=target_transform)

        self.root = os.path.expanduser(root)

        self.train = train
        self.url = self.urls[self.train]
        self.md5 = self.md5[self.train]

        if download:
            self.download()

        if not self._check_exists():
            raise RuntimeError("Dataset not found. You can use download=True to download it")

        if train:
            samples = make_dataset(self._target_folder, extensions=self.extension)
        else:
            with open(os.path.join(self._base_folder, self.gt_csv)) as csv_file:
                data = list(csv.reader(csv_file, delimiter=";", skipinitialspace=True))
            data = data[1:]
            samples = [(os.path.join(self._target_folder, row[0]), int(row[-1])) for row in data]

        self.samples = samples
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> Tuple[Any, Any]:

        path, target = self.samples[index]
        sample = PIL.Image.open(path).convert("RGB")

        if self.transform is not None:
            sample = self.transform(sample)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return sample, target

    @property
    def _base_folder(self) -> str:
        return os.path.join(self.root, type(self).__name__)

    @property
    def _target_folder(self) -> str:
        return os.path.join(self._base_folder, "Training" if self.train else "Final_Test/Images")

    def _check_exists(self) -> bool:
        return os.path.exists(self._target_folder) and os.path.isdir(self._target_folder)

    def download(self) -> None:
        if self._check_exists():
            return

        download_and_extract_archive(self.url, download_root=self.root, md5=self.md5)

        if not self.train:
            # Download Ground Truth for the test set
            download_and_extract_archive(
                self.gt_url, download_root=self.root, extract_root=self._base_folder, md5=self.gt_md5
            )
