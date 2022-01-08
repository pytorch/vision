from pathlib import Path
from typing import Any, Tuple, Callable, Optional

import numpy as np
import PIL.Image

from .utils import verify_str_arg, download_and_extract_archive, download_url
from .vision import VisionDataset


class Flowers102(VisionDataset):
    """`Oxford 102 Flower <https://www.robots.ox.ac.uk/~vgg/data/flowers/102/>`_ Dataset.

    .. warning::

        This class needs `scipy <https://docs.scipy.org/doc/>`_ to load target files from `.mat` format.

    Oxford 102 Flower is an image classification dataset consisting of 102 flower categories. The
    flowers chosen to be flower commonly occurring in the United Kingdom. Each class consists of
    between 40 and 258 images.

    The images have large scale, pose and light variations. In addition, there are categories that
    have large variations within the category and several very similar categories.

    Args:
        root (string): Root directory of the dataset.
        split (string, optional): The dataset split, supports ``"train"`` (default), ``"val"``, or ``"test"``.
        transform (callable, optional): A function/transform that takes in an PIL image and returns a
            transformed version. E.g, ``transforms.RandomCrop``.
        target_transform (callable, optional): A function/transform that takes in the target and transforms it.
    """

    def __init__(
        self,
        root: str,
        split: str = "train",
        download: bool = True,
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
    ) -> None:
        super().__init__(root, transform=transform, target_transform=target_transform)
        self._split = verify_str_arg(split, "split", ("train", "valid", "test"))
        self._base_folder = Path(self.root) / "flowers-102"
        self._meta_folder = self._base_folder / "labels"
        self._images_folder = self._base_folder / "jpg"

        if download:
            self._download()

        if not self._check_exists():
            raise RuntimeError("Dataset not found. You can use download=True to download it")

        self._labels = []
        self._image_files = []

        from scipy.io import loadmat

        # Read the label ids
        label_mat = loadmat(self._meta_folder / "imagelabels.mat")
        labels = label_mat["labels"][0]

        self.classes = np.unique(labels).tolist()
        self.class_to_idx = dict(zip(self.classes, range(len(self.classes))))

        # Read the image ids
        set_ids = loadmat(self._meta_folder / "setid.mat")
        splits_map = {"train": "trnid", "valid": "valid", "test": "tstid"}

        image_ids = set_ids[splits_map[self._split]][0]

        for image_id in image_ids:
            self._labels.append(self.class_to_idx[labels[image_id - 1]])
            self._image_files.append(self._images_folder / f"image_{image_id:05d}.jpg")

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

    def extra_repr(self) -> str:
        return f"split={self._split}"

    def _check_exists(self) -> bool:
        return all(folder.exists() and folder.is_dir() for folder in (self._meta_folder, self._images_folder))

    def _download(self) -> None:
        if self._check_exists():
            return

        download_and_extract_archive(
            "https://www.robots.ox.ac.uk/~vgg/data/flowers/102/102flowers.tgz",
            download_root=str(self._base_folder),
            md5="52808999861908f626f3c1f4e79d11fa",
        )

        download_url(
            "https://www.robots.ox.ac.uk/~vgg/data/flowers/102/setid.mat",
            str(self._meta_folder),
            md5="a5357ecc9cb78c4bef273ce3793fc85c",
        )

        download_url(
            "https://www.robots.ox.ac.uk/~vgg/data/flowers/102/imagelabels.mat",
            str(self._meta_folder),
            md5="e0620be6f572b9609742df49c70aed4d",
        )
