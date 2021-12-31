from pathlib import Path
from typing import Any, Tuple, Callable, Optional

import PIL.Image

from .folder import find_classes, make_dataset
from .utils import verify_str_arg, download_and_extract_archive
from .vision import VisionDataset


class Country211(VisionDataset):
    """`The Country211 Data Set <https://github.com/openai/CLIP/blob/main/data/country211.md>`_.

    filtered the YFCC100m dataset that have GPS coordinate corresponding to a ISO-3166 country code
    and created a balanced dataset by sampling 150 train images, 50 validation images,
    and 100 test images images for each country.


    Args:
        root (string): Root directory of the dataset.
        split (string, optional): The dataset split, supports ``"train"`` (default) and ``"test"``.
        transform (callable, optional): A function/transform that  takes in an PIL image and returns a transformed
            version. E.g, ``transforms.RandomCrop``.
        target_transform (callable, optional): A function/transform that takes in the target and transforms it.
    """

    _URL = "https://openaipublic.azureedge.net/clip/data/country211.tgz"

    def __init__(
        self,
        root: str,
        split: str = "train",
        download: bool = True,
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
        extensions: Tuple[str, ...] = ("jpg", "png"),
    ) -> None:
        super().__init__(root, transform=transform, target_transform=target_transform)
        self._split = verify_str_arg(split, "split", ("train", "valid", "test"))
        self._base_folder = Path(self.root) / "country211"

        if download:
            self._download()

        self.split_folder = self._base_folder / self._split

        self.classes, class_to_idx = find_classes(str(self.split_folder))
        self.samples = make_dataset(str(self.split_folder), class_to_idx, extensions, is_valid_file=None)

    def __len__(self) -> int:
        return len(self._image_files)

    def __getitem__(self, idx) -> Tuple[Any, Any]:
        image_file, label = self.samples[idx][0], self.samples[idx][1]
        image = PIL.Image.open(image_file).convert("RGB")

        if self.transform:
            image = self.transform(image)

        if self.target_transform:
            label = self.target_transform(label)

        return image, label

    def _check_exists(self) -> bool:
        return all(
            folder.exists() and folder.is_dir() for folder in (Path(self.root), self._base_folder, self._images_folder)
        )

    def _download(self) -> None:
        if self._check_exists():
            return
        download_and_extract_archive(self._URL, download_root=self.root, md5=None)
