from os import path
from typing import Any, Tuple
from urllib.parse import urljoin

from .folder import ImageFolder
from .utils import verify_str_arg


class Places365(ImageFolder):
    _BASE_URL = "http://data.csail.mit.edu/places/places365/"
    # {variant: (archive, md5)}
    _DEVKIT_META = {
        "standard": ("filelist_places365-standard.tar", "35a0585fee1fa656440f3ab298f8479c"),
        "challenge": ("filelist_places365-standard.tar", ""),
    }
    # {(split, high_res): (archive, md5)}
    _IMAGE_ARCHIVE_META = {
        ("train-standard", True): ("train_large_places365standard.tar", "67e186b496a84c929568076ed01a8aa1"),
        ("train-challenge", True): ("train_large_places365challenge.tar", "605f18e68e510c82b958664ea134545f"),
        ("val", True): ("val_large.tar", "9b71c4993ad89d2d8bcbdc4aef38042f"),
        ("test", True): ("test_large.tar", "41a4b6b724b1d2cd862fb3871ed59913"),
        ("train-standard", False): ("train_256_places365standard.tar", "53ca1c756c3d1e7809517cc47c5561c5"),
        ("train-challenge", False): ("train_256_places365challenge.tar", "741915038a5e3471ec7332404dfb64ef"),
        ("val", False): ("val_256.tar", "e27b17d8d44f4af9a78502beb927f808"),
        ("test", False): ("test_256.tar", "f532f6ad7b582262a2ec8009075e186b"),
    }
    _SPLITS = ("train-standard", "train-challenge", "val", "test")

    def __init__(
        self, root: str, split: str = "train-standard", high_res: bool = True, download: bool = False, **kwargs: Any
    ) -> None:
        self.root = root = path.abspath(path.expanduser(root))
        self.split = self._verify_split(split)
        self.high_res = high_res

        if download:
            self.download()

        super().__init__(root, **kwargs)
        self.root = root

    def _verify_split(self, split: str) -> str:
        return verify_str_arg(split, "split", self._SPLITS)

    @property
    def variant(self) -> str:
        return 'challenge' if self.split == 'train_challenge' else 'standard'

    @property
    def _archive_meta(self) -> Tuple[str, str]:
        return self._IMAGE_ARCHIVE_META[(self.split, self.high_res)]

    @property
    def archive(self) -> str:
        return self._archive_meta[0]

    @property
    def md5(self) -> str:
        return self._archive_meta[1]

    @property
    def url(self) -> str:
        return urljoin(self._BASE_URL, self.archive)

    @property
    def split_dir(self) -> str:
        return path.join(self.root, path.splitext(self.archive)[0])

    @property
    def meta_file(self) -> str:
        return path.join(self.root, f"meta-{self.variant}.bin")

    def download(self):
        if not path.exists(self.meta_file):
            self._parse_devkit()

        # fail if split dir is already present

        self._load_meta()

    def _parse_devkit(self):
        print("Downloading and parsing the devkit. This is only needed for the first run.")

    def _load_meta(self):
        # fail if not present
        pass

    def extra_repr(self) -> str:
        return "\n".join(("Split: {split}", "High resolution: {high_res}", "Variant: {variant}")).format(**self.__dict__)
