import os
import os.path
from typing import Callable, Optional, Any, Tuple

from PIL import Image

from .utils import download_and_extract_archive, download_url
from .vision import VisionDataset


class StanfordCars(VisionDataset):
    """`Stanford Cars <https://ai.stanford.edu/~jkrause/cars/car_dataset.html>`_ Dataset

    .. warning::

        This class needs `scipy <https://docs.scipy.org/doc/>`_ to load target files from `.mat` format.

    Args:
        root (string): Root directory of dataset
        train (bool, optional):If True, creates dataset from training set, otherwise creates from test set
        transform (callable, optional): A function/transform that  takes in an PIL image
            and returns a transformed version. E.g, ``transforms.RandomCrop``
        target_transform (callable, optional): A function/transform that takes in the
            target and transforms it.
        download (bool, optional): If True, downloads the dataset from the internet and
            puts it in root directory. If dataset is already downloaded, it is not
            downloaded again."""

    urls = (
        "https://ai.stanford.edu/~jkrause/car196/cars_test.tgz",
        "https://ai.stanford.edu/~jkrause/car196/cars_train.tgz",
    )  # test and train image urls

    md5s = (
        "4ce7ebf6a94d07f1952d94dd34c4d501",
        "065e5b463ae28d29e77c1b4b166cfe61",
    )  # md5checksum for test and train data

    annot_urls = (
        "https://ai.stanford.edu/~jkrause/car196/cars_test_annos_withlabels.mat",
        "https://ai.stanford.edu/~jkrause/cars/car_devkit.tgz",
    )  # annotations and labels for test and train

    annot_md5s = (
        "b0a2b23655a3edd16d84508592a98d10",
        "c3b158d763b6e2245038c8ad08e45376",
    )  # md5 checksum for annotations

    def __init__(
        self,
        root: str,
        train: bool = True,
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
        download: bool = False,
    ) -> None:

        try:
            import scipy.io as sio
        except ImportError:
            raise RuntimeError("Scipy is not found. This dataset needs to have scipy installed: pip install scipy")

        super().__init__(root, transform=transform, target_transform=target_transform)

        self.train = train

        if download:
            self.download()

        if not self._check_exists():
            raise RuntimeError("Dataset not found. You can use download=True to download it")

        self._samples = [
            (
                os.path.join(self.root, f"cars_{'train' if self.train else 'test'}", annotation["fname"]),
                annotation["class"] - 1,  # Beware stanford cars target mapping  starts from 1
            )
            for annotation in sio.loadmat(
                os.path.join(
                    self.root,
                    *["devkit", "cars_train_annos.mat"] if self.train else ["cars_test_annos_withlabels.mat"],
                ),
                squeeze_me=True,
            )["annotations"]
        ]

        class_names = sio.loadmat(os.path.join(self.root, "devkit", "cars_meta.mat"))["class_names"][0]
        self.classes = {class_name[0]: i for i, class_name in enumerate(class_names)}

    def __len__(self) -> int:
        return len(self._samples)

    def __getitem__(self, idx: int) -> Tuple[Any, Any]:
        """Returns pil_image and class_id for given index"""
        image_path, target = self._samples[idx]
        pil_image = Image.open(image_path).convert("RGB")

        if self.transform is not None:
            pil_image = self.transform(pil_image)
        if self.target_transform is not None:
            target = self.target_transform(target)
        return pil_image, target

    def download(self) -> None:
        if self._check_exists():
            return

        download_and_extract_archive(url=self.urls[self.train], download_root=self.root, md5=self.md5s[self.train])
        download_and_extract_archive(url=self.annot_urls[1], download_root=self.root, md5=self.annot_md5s[1])
        if not self.train:
            download_url(
                url=self.annot_urls[0],
                root=self.root,
                md5=self.annot_md5s[0],
            )

    def _check_exists(self) -> bool:
        return (
            os.path.exists(os.path.join(self.root, f"cars_{'train' if self.train else 'test'}"))
            and os.path.isdir(os.path.join(self.root, f"cars_{'train' if self.train else 'test'}"))
            and os.path.exists(os.path.join(self.root, "devkit", "cars_meta.mat"))
            if self.train
            else os.path.exists(os.path.join(self.root, "cars_test_annos_withlabels.mat"))
        )
