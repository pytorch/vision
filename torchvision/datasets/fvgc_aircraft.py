import os
import shutil
from typing import Any, Callable, List, Optional, Tuple

import PIL.Image

from .utils import download_and_extract_archive, verify_str_arg
from .vision import VisionDataset


class FVGCAircraft(VisionDataset):
    """`FVGC Aircraft <https://www.robots.ox.ac.uk/~vgg/data/fgvc-aircraft/>`_ Dataset.

    The dataset contains 10,200 images of aircraft, with 100 images for each of 102
    different aircraft model variants, most of which are airplanes.

    Args:
        root (string): Root directory of the FVGC Aircraft dataset.
        split (string, optional): The dataset split, supports ``train``, ``val``,
            ``trainval`` and ``test``.
        download (bool, optional): If True, downloads the dataset from the internet and
            puts it in root directory. If dataset is already downloaded, it is not
            downloaded again.
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
        download: Optional[str] = None,
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(root, transform=transform, target_transform=target_transform)
        self._split = verify_str_arg(split, "split", ("train", "val", "trainval", "test"))

        self._data_path = os.path.join(self.root, "fgvc-aircraft-2013b")
        if download:
            self._download()

        if not self._check_exists():
            raise RuntimeError("Dataset not found. You can use download=True to download it")

        self._label_names = sorted(self._get_label_names(self._data_path))

        # Parse the downloaded files
        self._image_folder = os.path.join(self.root, self._split)
        self._create_fgvc_aircrafts_disk_folder(self._data_path)

        self._label_name_to_idx = dict(zip(self._label_names, range(len(self._label_names))))

        self._image_files = []
        self._labels = []
        for label_name in self._label_names:
            img_rel_folder = os.path.join(self._image_folder, label_name)
            img_file_name_list = [
                f for f in os.listdir(img_rel_folder) if os.path.isfile(os.path.join(img_rel_folder, f))
            ]
            self._labels += [self._label_name_to_idx[label_name]] * len(img_file_name_list)
            self._image_files += [os.path.join(img_rel_folder, img_name) for img_name in img_file_name_list]

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

    def _create_fgvc_aircrafts_disk_folder(self, input_path: str):
        img_data_folder = os.path.join(input_path, "data", "images")
        labels_path = os.path.join(input_path, "data", f"images_variant_{self._split}.txt")
        for label in self._label_names:
            os.makedirs(os.path.join(self._image_folder, label), exist_ok=True)

        with open(labels_path, "r") as labels_file:
            lines = [line.strip() for line in labels_file]
            for line in lines:
                line = line.split(" ")
                image_name = line[0]
                label_name = self._parse_aircraft_name(" ".join(line[1:]))
                shutil.copy(
                    src=os.path.join(img_data_folder, f"{image_name}.jpg"),
                    dst=os.path.join(self._image_folder, label_name),
                )

    def _get_label_names(self, input_path: str) -> List[str]:
        variants_file = os.path.join(input_path, "data", "variants.txt")
        with open(variants_file, "r") as f:
            return [self._parse_aircraft_name(line.strip()) for line in f]

    def _parse_aircraft_name(self, name: str) -> str:
        return name.replace("/", "-").replace(" ", "-")
