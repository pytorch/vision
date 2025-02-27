import os
from pathlib import Path
from typing import Callable, List, Optional, Union

import numpy as np
import torch
from PIL import Image

from .utils import download_url
from .vision import VisionDataset


class PhotoTour(VisionDataset):
    """`Multi-view Stereo Correspondence <https://phototour.cs.washington.edu/patches/default.htm>`_ Dataset.

    Args:
        root (str or ``pathlib.Path``): Root directory where images are.
        name (string): Name of the dataset to load.
        transform (callable, optional): A function/transform that takes in a PIL image
            and returns a transformed version.
        download (bool, optional): If true, downloads the dataset from the internet and
            puts it in root directory. If dataset is already downloaded, it is not
            downloaded again.
    """

    urls = {
        "trevi": [
            "https://phototour.cs.washington.edu/patches/trevi.zip",
            "trevi.zip",
            "d49ab428f154554856f83dba8aa76539",
        ],
        "notredame": [
            "https://phototour.cs.washington.edu/patches/notredame.zip",
            "notredame.zip",
            "0f801127085e405a61465605ea80c595",
        ],
        "halfdome": [
            "https://phototour.cs.washington.edu/patches/halfdome.zip",
            "halfdome.zip",
            "db871c5a86f4878c6754d0d12146440b",
        ],
    }
    means = {
        "trevi": 0.4832,
        "notredame": 0.4757,
        "halfdome": 0.4718,
    }
    stds = {
        "trevi": 0.1913,
        "notredame": 0.1931,
        "halfdome": 0.1791,
    }
    lens = {
        "trevi": 101120,
        "notredame": 104196,
        "halfdome": 107776,
    }
    image_ext = "bmp"
    info_file = "info.txt"

    def __init__(
        self,
        root: Union[str, Path],
        name: str,
        train: bool = True,
        transform: Optional[Callable] = None,
        download: bool = False,
    ) -> None:
        super().__init__(root, transform=transform)
        self.name = name
        self.data_dir = os.path.join(self.root, name)
        self.data_down = os.path.join(self.root, f"{name}.zip")
        self.data_file = os.path.join(self.root, f"{name}.pt")

        self.train = train
        self.mean = self.means[name]
        self.std = self.stds[name]

        if download:
            self.download()

        if not self._check_datafile_exists():
            self.cache()

        # load the serialized data
        self.data, self.labels = torch.load(self.data_file, weights_only=True)

    def __getitem__(self, index: int) -> torch.Tensor:
        """
        Args:
            index (int): Index

        Returns:
            torch.Tensor: The image patch.
        """
        data = self.data[index]
        if self.transform is not None:
            data = self.transform(data)
        return data

    def __len__(self) -> int:
        return len(self.data)

    def _check_datafile_exists(self) -> bool:
        return os.path.exists(self.data_file)

    def _check_downloaded(self) -> bool:
        return os.path.exists(self.data_dir)

    def download(self) -> None:
        if self._check_datafile_exists():
            return

        if not self._check_downloaded():
            # download files
            url = self.urls[self.name][0]
            filename = self.urls[self.name][1]
            md5 = self.urls[self.name][2]
            fpath = os.path.join(self.root, filename)

            download_url(url, self.root, filename, md5)

            import zipfile

            with zipfile.ZipFile(fpath, "r") as z:
                z.extractall(self.data_dir)

            os.unlink(fpath)

    def cache(self) -> None:
        # process and save as torch files
        dataset = (
            read_image_file(self.data_dir, self.image_ext, self.lens[self.name]),
            read_info_file(self.data_dir, self.info_file),
        )

        with open(self.data_file, "wb") as f:
            torch.save(dataset, f)

    def extra_repr(self) -> str:
        return f"Dataset: {self.name}"


def read_image_file(data_dir: str, image_ext: str, n: int) -> torch.Tensor:
    """Return a Tensor containing the patches"""

    def PIL2array(_img: Image.Image) -> np.ndarray:
        """Convert PIL image type to numpy 2D array"""
        # Ensure the patch size is exactly 64x64
        if _img.size != (64, 64):
            raise ValueError(f"Invalid patch size: {_img.size}")
        return np.array(_img.getdata(), dtype=np.uint8).reshape(64, 64)

    def find_files(_data_dir: str, _image_ext: str) -> List[str]:
        """Return a list with the file names of the images containing the patches"""
        files = []
        for file_dir in os.listdir(_data_dir):
            if file_dir.endswith(_image_ext):
                files.append(os.path.join(_data_dir, file_dir))
        return sorted(files)

    patches = []
    list_files = find_files(data_dir, image_ext)

    for fpath in list_files:
        img = Image.open(fpath)
        for y in range(0, img.height, 64):
            for x in range(0, img.width, 64):
                patch = img.crop((x, y, x + 64, y + 64))
                try:
                    patches.append(PIL2array(patch))
                except ValueError as e:
                    print(f"Skipping invalid patch at ({x}, {y}) in {fpath}: {e}")
    return torch.ByteTensor(np.array(patches[:n]))


def read_info_file(data_dir: str, info_file: str) -> torch.Tensor:
    """Return a Tensor containing the list of labels."""
    with open(os.path.join(data_dir, info_file)) as f:
        labels = [int(line.split()[0]) for line in f]
    return torch.LongTensor(labels)
