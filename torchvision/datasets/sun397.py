from pathlib import Path
from typing import Any, Tuple, Callable, Optional

import PIL.Image

from .utils import verify_str_arg, download_and_extract_archive
from .vision import VisionDataset


class SUN397(VisionDataset):
    """`The SUN397 Data Set <https://vision.princeton.edu/projects/2010/SUN/>`_.

    The SUN397 or Scene UNderstanding (SUN) is a dataset for scene recognition consisting of
    397 categories with 108'754 images. The dataset also provides 10 partitions for training
    and testing, with each partition consisting of 50 images per class.

    Args:
        root (string): Root directory of the dataset.
        split (string, optional): The dataset split, supports ``"train"`` (default) and ``"test"``.
        partition (int, optional): A valid partition can be an integer from 1 to 10 or None,
            for the entire dataset.
        download (bool, optional): If true, downloads the dataset from the internet and
            puts it in root directory. If dataset is already downloaded, it is not
            downloaded again.
        transform (callable, optional): A function/transform that  takes in an PIL image and returns a transformed
            version. E.g, ``transforms.RandomCrop``.
        target_transform (callable, optional): A function/transform that takes in the target and transforms it.
    """

    _DATASET_URL = "http://vision.princeton.edu/projects/2010/SUN/SUN397.tar.gz"
    _DATASET_MD5 = "8ca2778205c41d23104230ba66911c7a"
    _PARTITIONS_URL = "https://vision.princeton.edu/projects/2010/SUN/download/Partitions.zip"
    _PARTITIONS_MD5 = "29a205c0a0129d21f36cbecfefe81881"

    def __init__(
        self,
        root: str,
        split: str = "train",
        partition: Optional[int] = 1,
        download: bool = True,
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
    ) -> None:
        super().__init__(root, transform=transform, target_transform=target_transform)
        self.split = verify_str_arg(split, "split", ("train", "test"))
        self.partition = partition
        self._data_dir = Path(self.root) / "SUN397"

        if self.partition is not None:
            if self.partition < 0 or self.partition > 10:
                raise RuntimeError(f"The partition parameter should be an int in [1, 10] or None, got {partition}.")

        if download:
            self._download()

        if not self._check_exists():
            raise RuntimeError("Dataset not found. You can use download=True to download it")

        with open(self._data_dir / "ClassName.txt") as f:
            self.classes = [c[3:].strip() for c in f]

        self.class_to_idx = dict(zip(self.classes, range(len(self.classes))))
        if self.partition is not None:
            with open(self._data_dir / f"{self.split.title()}ing_{self.partition:02d}.txt", "r") as f:
                self._image_files = [self._data_dir.joinpath(*line.strip()[1:].split("/")) for line in f]
        else:
            self._image_files = list(self._data_dir.rglob("sun_*.jpg"))

        self._labels = [
            self.class_to_idx["/".join(path.relative_to(self._data_dir).parts[1:-1])] for path in self._image_files
        ]

    def __len__(self) -> int:
        return len(self._image_files)

    def __getitem__(self, idx) -> Tuple[Any, Any]:
        image_file, label = self._image_files[idx], self._labels[idx]
        image = PIL.Image.open(image_file).convert("RGB")

        if self.transform:
            image = self.transform(image)

        if self.target_transform:
            label = self.target_transform(label)

        return image, label

    def _check_exists(self) -> bool:
        return self._data_dir.exists() and self._data_dir.is_dir()

    def extra_repr(self) -> str:
        return "Split: {split}".format(**self.__dict__)

    def _download(self) -> None:
        if self._check_exists():
            return
        download_and_extract_archive(self._DATASET_URL, download_root=self.root, md5=self._DATASET_MD5)
        download_and_extract_archive(self._PARTITIONS_URL, download_root=str(self._data_dir), md5=self._PARTITIONS_MD5)
