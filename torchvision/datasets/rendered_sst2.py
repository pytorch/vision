from pathlib import Path
from typing import Any, Tuple, Callable, Optional

import PIL.Image

from .utils import verify_str_arg, download_and_extract_archive
from .vision import VisionDataset


class RenderedSST2(VisionDataset):
    """`The Rendered SST2 Dataset <https://github.com/openai/CLIP/blob/main/data/rendered-sst2.md>`_.

    Rendered SST2 is a image classification dataset used to evaluate the models capability on optical
    character recognition. This dataset was generated bu rendering sentences in the Standford Sentiment
    Treebank v2 dataset.

    This dataset contains two classes (positive and negative) and is divided in three splits: a  train
    split containing 6920 images (3610 positive and 3310 negative), a validation split containing 872 images
    (444 positive and 428 negative), and a test split containing 1821 images (909 positive and 912 negative).

    Args:
        root (string): Root directory of the dataset.
        split (string, optional): The dataset split, supports ``"train"`` (default), `"valid"` and ``"test"``.
        transform (callable, optional): A function/transform that  takes in an PIL image and returns a transformed
            version. E.g, ``transforms.RandomCrop``.
        target_transform (callable, optional): A function/transform that takes in the target and transforms it.
    """

    _URL = "https://openaipublic.azureedge.net/clip/data/rendered-sst2.tgz"
    _MD5 = "2384d08e9dcfa4bd55b324e610496ee5"

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
        self._base_folder = Path(self.root) / "rendered-sst2"
        self.classes = ["negative", "positive"]
        self.class_to_idx = {"negative": 0, "positive": 1}

        if download:
            self._download()

        if not self._check_exists():
            raise RuntimeError("Dataset not found. You can use download=True to download it")

        self._labels = []
        self._image_files = []

        for p in (self._base_folder / self._split).glob("**/*.png"):
            self._labels.append(self.class_to_idx[p.parent.name])
            self._image_files.append(p)
        print(self._labels)
        print(self._image_files)

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
        for class_label in set(self.classes):
            if not (
                (self._base_folder / self._split / class_label).exists()
                and (self._base_folder / self._split / class_label).is_dir()
            ):
                return False
        return True

    def _download(self) -> None:
        if self._check_exists():
            return
        download_and_extract_archive(self._URL, download_root=self.root, md5=self._MD5)
