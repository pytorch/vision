import os
from typing import Optional, Callable, Union

import PIL.Image

from .utils import verify_str_arg, download_and_extract_archive
from .vision import VisionDataset


class DTD(VisionDataset):
    """`Describable Textures Dataset (DTD) <https://www.robots.ox.ac.uk/~vgg/data/dtd/>`_.

    Args:
        root (string): Root directory of the dataset.
        split (string, optional): The dataset split, supports ``"train"`` (default), ``"val"``, or ``"test"``.
        fold (string or int, optional): The dataset fold. Should be ``1 <= fold <= 10``. Defaults to ``1``.
        transform (callable, optional): A function/transform that  takes in an PIL image and returns a transformed
            version. E.g, ``transforms.RandomCrop``.
        target_transform (callable, optional): A function/transform that takes in the target and transforms it.
    """

    _URL = "https://www.robots.ox.ac.uk/~vgg/data/dtd/download/dtd-r1.0.1.tar.gz"
    _MD5 = "fff73e5086ae6bdbea199a49dfb8a4c1"

    def __init__(
        self,
        root: str,
        split: str = "train",
        fold: Union[str, int] = 1,
        download: bool = True,
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
    ) -> None:
        super().__init__(root, transform=transform, target_transform=target_transform)
        self._split = verify_str_arg(split, "split", ("train", "val", "test"))
        self._fold = verify_str_arg(str(fold), "fold", [str(i) for i in range(1, 11)])

        if download:
            self._download()

        if not self._check_exists():
            raise RuntimeError("Dataset not found. You can use download=True to download it")

        image_files = []
        classes = []
        with open(os.path.join(self._meta_folder, f"{self._split}{self._fold}.txt")) as file:
            for line in file:
                cls, name = line.strip().split("/")
                image_files.append(os.path.join(self._images_folder, cls, name))
                classes.append(cls)
        self._image_files = image_files

        self.classes = sorted(set(classes))
        self.class_to_idx = dict(zip(self.classes, range(len(self.classes))))
        self._labels = [self.class_to_idx[cls] for cls in classes]

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
        return f"split={self._split}, fold={self._fold}"

    @property
    def _base_folder(self):
        return os.path.join(self.root, type(self).__name__.lower())

    @property
    def _data_folder(self) -> str:
        return os.path.join(self._base_folder, "dtd")

    @property
    def _meta_folder(self) -> str:
        return os.path.join(self._data_folder, "labels")

    @property
    def _images_folder(self) -> str:
        return os.path.join(self._data_folder, "images")

    def _check_exists(self) -> bool:
        return os.path.exists(self._data_folder) and os.path.isdir(self._data_folder)

    def _download(self) -> None:
        if self._check_exists():
            return
        download_and_extract_archive(self._URL, download_root=self._base_folder, md5=self._MD5)
