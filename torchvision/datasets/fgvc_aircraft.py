from __future__ import annotations

import os
from typing import Any, Callable, List, Optional, Tuple

import PIL.Image

from .utils import download_and_extract_archive, verify_str_arg
from .vision import VisionDataset

annotation_level_to_file = {
    "variant": "variants.txt",
    "family": "families.txt",
    "manufacturer": "manufacturers.txt",
}


class FGVCAircraft(VisionDataset):
    """`FGVC Aircraft <https://www.robots.ox.ac.uk/~vgg/data/fgvc-aircraft/>`_ Dataset.

    The dataset contains 10,200 images of aircraft, with 100 images for each of 102
    different aircraft model variants, most of which are airplanes.
    Aircraft models are organized in a four-levels hierarchy. The four levels, from
    finer to coarser, are:

    - Model, e.g. Boeing 737-76J. Since certain models are nearly visually indistinguishable,
        this level is not used in the evaluation.
    - Variant, e.g. Boeing 737-700. A variant collapses all the models that are visually
        indistinguishable into one class. The dataset comprises 102 different variants.
    - Family, e.g. Boeing 737. The dataset comprises 70 different families.
    - Manufacturer, e.g. Boeing. The dataset comprises 41 different manufacturers.

    Args:
        root (string): Root directory of the FGVC Aircraft dataset.
        split (string, optional): The dataset split, supports ``train``, ``val``,
            ``trainval`` and ``test``.
        download (bool, optional): If True, downloads the dataset from the internet and
            puts it in root directory. If dataset is already downloaded, it is not
            downloaded again.
        annotation_level (str, optional): The annotation level, supports ``variant``,
            ``family`` and ``manufacturer``.
        transform (callable, optional): A function/transform that  takes in an PIL image
            and returns a transformed version. E.g, ``transforms.RandomCrop``
        target_transform (callable, optional): A function/transform that takes in the
            target and transforms it.
    """

    _URL = "https://www.robots.ox.ac.uk/~vgg/data/fgvc-aircraft/archives/"
    _URL_FILE = "fgvc-aircraft-2013b.tar.gz"

    def __init__(
        self,
        root: str,
        split: str = "trainval",
        download: bool = False,
        annotation_level: str = "variant",
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
    ) -> None:
        super().__init__(root, transform=transform, target_transform=target_transform)
        self._split = verify_str_arg(split, "split", ("train", "val", "trainval", "test"))
        self._annotation_level = verify_str_arg(
            annotation_level, "annotation_level", ("variant", "family", "manufacturer")
        )

        self._data_path = os.path.join(self.root, "fgvc-aircraft-2013b")
        if download:
            self._download()

        if not self._check_exists():
            raise RuntimeError("Dataset not found. You can use download=True to download it")

        self.classes = self._get_classes(self._data_path)

        # Parse the downloaded files
        self._image_folder = os.path.join(self.root, self._split)
        self.class_to_idx = dict(zip(self.classes, range(len(self.classes))))

        self._image_files = []
        self._labels = []

        image_data_folder = os.path.join(self._data_path, "data", "images")
        labels_path = os.path.join(self._data_path, "data", f"images_{self._annotation_level}_{self._split}.txt")

        with open(labels_path, "r") as labels_file:
            lines = [line.strip() for line in labels_file]
            for line in lines:
                image_name, label_name = line.strip().split(" ", 1)
                self._image_files.append(os.path.join(image_data_folder, f"{image_name}.jpg"))
                self._labels.append(self.class_to_idx[label_name])

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

    def _download(self):
        """
        Download the FGVC Aircraft dataset archive and extract it under root.
        """
        if self._check_exists():
            return
        download_and_extract_archive(self._URL + self._URL_FILE, self.root)

    def _check_exists(self) -> bool:
        return os.path.exists(self._data_path) and os.path.isdir(self._data_path)

    def _get_classes(self, input_path: str) -> List[str]:
        annotation_file = os.path.join(input_path, "data", annotation_level_to_file[self._annotation_level])
        with open(annotation_file, "r") as f:
            return [line.strip() for line in f]
