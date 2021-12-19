import os
from typing import Any

from .folder import ImageFolder
from .utils import download_and_extract_archive, check_integrity


class EuroSAT(ImageFolder):
    """RGB version of the `EuroSAT <https://arxiv.org/pdf/1709.00029.pdf>`_ Dataset.

    Args:
        root (string): Root directory of dataset where ``EuroSAT.zip`` exists.
        download (bool, optional): If True, downloads the dataset from the internet and
            puts it in root directory. If dataset is already downloaded, it is not
            downloaded again.
        transform (callable, optional): A function/transform that  takes in an PIL image
            and returns a transformed version. E.g, ``transforms.RandomCrop``
        target_transform (callable, optional): A function/transform that takes in the
            target and transforms it.
    """

    url = "https://madm.dfki.de/files/sentinel/EuroSAT.zip"
    md5 = "c8fa014336c82ac7804f0398fcb19387"
    filename = "EuroSAT.zip"

    _classes = [
        "Annual Crop",
        "Forest",
        "Herbaceous Vegetation",
        "Highway",
        "Industrial Buildings",
        "Pasture",
        "Permanent Crop",
        "Residential Buildings",
        "River",
        "Sea & Lake",
    ]

    def __init__(
        self,
        root: str,
        download: bool = False,
        **kwargs: Any,
    ) -> None:
        self.root = os.path.expanduser(root)

        if download:
            self.download()

        if not self._check_exists():
            raise RuntimeError("Dataset not found. You can use download=True to download it")

        super().__init__(os.path.join(self.data_folder, "2750"), **kwargs)
        self.classes = self._classes
        self.class_to_idx = {_class: i for i, _class in enumerate(self.classes)}

    def __len__(self) -> int:
        return len(self.data)

    @property
    def data_folder(self) -> str:
        return os.path.join(self.root, self.__class__.__name__)

    def _check_exists(self) -> bool:
        return check_integrity(os.path.join(self.data_folder, self.filename))

    def download(self) -> None:

        if self._check_exists():
            return

        os.makedirs(self.data_folder, exist_ok=True)
        print(f"Downloading {self.url}")
        download_and_extract_archive(self.url, download_root=self.data_folder, filename=self.filename, md5=self.md5)
