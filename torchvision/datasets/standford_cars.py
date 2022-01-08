import os
import os.path
from typing import Callable, Optional
from PIL import Image

from .utils import download_and_extract_archive, download_url
from .vision import VisionDataset


class StanfordCars(VisionDataset):
    """`Stanford Cars <https://ai.stanford.edu/~jkrause/cars/car_dataset.html>`_ Dataset

    .. warning::

        This class needs `scipy <https://docs.scipy.org/doc/>`_ to load target files from `.mat` format.

    Args:
        root (string): Root directory of dataset
        train (bool, optional):
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
    )

    md5s = ("4ce7ebf6a94d07f1952d94dd34c4d501", "065e5b463ae28d29e77c1b4b166cfe61")  # md5checksum for test and train

    annot_urls = (
        "https://ai.stanford.edu/~jkrause/car196/cars_test_annos_withlabels.mat",
        "https://ai.stanford.edu/~jkrause/cars/car_devkit.tgz",
    )
    annot_md5s = (
        "b0a2b23655a3edd16d84508592a98d10",
        "c3b158d763b6e2245038c8ad08e45376",
    )

    def __init__(
            self,
            root: str,
            train: bool = True,
            transform: Optional[Callable] = None,
            target_transform: Optional[Callable] = None,
            download: bool = False,
    ) -> None:

        try:
            from scipy.io import loadmat

            self._loadmat = loadmat
        except ImportError:
            raise RuntimeError("Scipy is not found. This dataset needs to have scipy installed: pip install scipy")

        super().__init__(root, transform=transform, target_transform=target_transform)

        self.train = train

        if download:
            self.download()

        if not self._check_exists():
            raise RuntimeError("Dataset not found. You can use download=True to download it")

        self._samples = self._make_dataset()
        self.classes = self._get_classes_name()  # class_id to class_name mapping

    def _get_class_names(self) -> dict:
        """
        Returns Mapping of class ids to class names in form of Dictionary
        """
        meta_data = self._loadmat(os.path.join(self.root, "devkit/cars_meta.mat"))
        class_names = meta_data["class_names"][0]
        return {
            class_name[0].replace(" ", "_").replace("/", "_"): i
            for i, class_name in enumerate(class_names)
        }

    def _make_dataset(self):
        """
        Returns Annotations for training data and testing data
        """
        annotations = None
        if self.train:
            annotations = self._loadmat(os.path.join(self.root, "devkit/cars_train_annos.mat"))
        else:
            annotations = self._loadmat(os.path.join(self.root, "cars_test_annos_withlabels.mat"))
        samples = []
        annotations = annotations["annotations"][0]
        for index in range(len(annotations)):
            target = annotations[index][4][0, 0]
            image_file = annotations[index][5][0]
            # Beware: Stanford cars targets starts at 1
            target = target - 1
            samples.append((image_file, target))
        return samples

    def __len__(self) -> int:
        return len(self._samples)

    def __getitem__(self, idx: int) -> (Image, int):
        """Returns pil_image and class_id for given index"""
        image_file, target = self._samples[idx]
        image_path = os.path.join(self.root, f"cars_{'train' if self.train else 'test'}", image_file)
        pil_image = Image.open(image_path).convert("RGB")

        if self.transform is not None:
            pil_image = self.transform(pil_image)
        if self.target_transform is not None:
            target = self.target_transform(target)
        return pil_image, target

    def download(self) -> None:
        if self._check_exists():
            return
        else:
            download_and_extract_archive(
                url=self.urls[self.train],
                download_root=self.root,
                extract_root=self.root,
                md5=self.md5s[self.train]
            )
            download_and_extract_archive(
                url=self.annot_urls[1], download_root=self.root, extract_root=self.root,
                md5=self.annot_md5s[1]
            )
            if not self.train:
                download_url(
                    url=self.annot_urls[0], filename="cars_test_annos_withlabels.mat", root=self.root,
                    md5=self.annot_md5s[0]
                )

    def _check_exists(self) -> bool:
        return os.path.exists(os.path.join(self.root, f"cars_{'train' if self.train else 'test'}")) and os.path.isdir(
            os.path.join(self.root, f"cars_{'train' if self.train else 'test'}")) and os.path.exists(
            os.path.join(self.root, "devkit/cars_meta.mat")) if self.train else os.path.exists(os.path.join(self.root,"cars_test_annos_withlabels.mat"))
