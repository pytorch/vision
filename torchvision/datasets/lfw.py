import os
from typing import Any, Callable, Optional, Tuple
from PIL import Image
from torchvision.datasets import VisionDataset
from torchvision.datasets.utils import check_integrity, download_and_extract_archive, download_url, verify_str_arg


class _LFW(VisionDataset):

    base_folder = 'lfw-py'
    download_url_prefix = "http://vis-www.cs.umass.edu/lfw/"

    file_dict = {
        'original': ("lfw", "lfw.tgz", "a17d05bd522c52d84eca14327a23d494"),
        'funneled': ("lfw_funneled", "lfw-funneled.tgz", "1b42dfed7d15c9b2dd63d5e5840c86ad"),
        'deepfunneled': ("lfw-deepfunneled", "lfw-deepfunneled.tgz", "68331da3eb755a505a502b5aacb3c201")
    }

    def __init__(
        self,
        root: str,
        train: bool,
        image_set: str,
        view: str,
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
        download: bool = False,
    ):
        super(_LFW, self).__init__(os.path.join(root, self.base_folder),
                                   transform=transform, target_transform=target_transform)

        image_set = verify_str_arg(image_set.lower(), 'image_set', self.file_dict.keys())
        images_dir, self.filename, self.md5 = self.file_dict[image_set]

        self.view = verify_str_arg(view.lower(), 'view', ['people', 'pairs'])
        self.split = "Train" if train else "Test"

        if download:
            self.download()

        if not self._check_integrity():
            raise RuntimeError('Dataset not found or corrupted.' +
                               ' You can use download=True to download it')

        self.images_dir = os.path.join(self.root, images_dir)

    def loader(self, path: str) -> Image.Image:
        with open(path, 'rb') as f:
            img = Image.open(f)
            return img.convert('RGB')

    def _check_integrity(self):
        fpath = os.path.join(self.root, self.filename)
        fname = os.path.join(self.root, f"{self.view}Dev{self.split}.txt")
        if not check_integrity(fpath, self.md5) or not check_integrity(fname):
            return False
        return True

    def download(self):
        if self._check_integrity():
            print('Files already downloaded and verified')
            return
        url = f"{self.download_url_prefix}{self.filename}"
        download_and_extract_archive(url, self.root, filename=self.filename, md5=self.md5)
        download_url(f"{self.download_url_prefix}{self.view}Dev{self.split}.txt", self.root)

    def __len__(self):
        return len(self.data)


class LFWPeople(_LFW):
    """`LFW <http://vis-www.cs.umass.edu/lfw/>`_ Dataset.
    Args:
        root (string): Root directory of dataset where directory
            ``lfw-py`` exists or will be saved to if download is set to True.
        train (bool, optional): If True, creates dataset from "DevTrain" set, otherwise
            creates from "DevTest" set.
        image_set (str, optional): Type of image funneling to use, ``original``, ``funneled`` or
            ``deepfunneled``. Defaults to ``original``.
        transform (callable, optional): A function/transform that  takes in an PIL image
            and returns a transformed version. E.g, ``transforms.RandomCrop``
        target_transform (callable, optional): A function/transform that takes in the
            target and transforms it.
        download (bool, optional): If true, downloads the dataset from the internet and
            puts it in root directory. If dataset is already downloaded, it is not
            downloaded again.
    """

    def __init__(
        self,
        root: str,
        train: bool = True,
        image_set: str = "original",
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
        download: bool = False,
    ):
        super(LFWPeople, self).__init__(root, train, image_set, "people",
                                        transform, target_transform, download)

        self.people_file = os.path.join(self.root, f"peopleDev{self.split}.txt")
        self.cls_to_names, self.data, self.targets = self._get_people(self.images_dir, self.people_file)

    def _get_people(self, images_dir, people_file):
        with open(people_file, 'r') as f:
            lines = f.readlines()
            n_lines = int(lines[0])
            people = [line.strip().split("\t") for line in lines[1: n_lines + 1]]

            cls_to_names = []
            data = []
            targets = []
            for cls, (identity, num_imgs) in enumerate(people):
                cls_to_names.append(identity)
                for num in range(1, int(num_imgs) + 1):
                    img = os.path.join(images_dir, identity, f"{identity}_{num:04d}.jpg")
                    if os.path.exists(img):
                        data.append(img)
                        targets.append(cls)

        return cls_to_names, data, targets

    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        """
        Args:
            index (int): Index

        Returns:
            tuple: Tuple (image, target) where target is the identity of the person.
        """
        img = self.loader(self.data[index])
        target = self.targets[index]

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target

    def extra_repr(self) -> str:
        return "Split: {} \nNo. of classes: {}".format(self.split, len(self.cls_to_names))


class LFWPairs(_LFW):
    """`LFW <http://vis-www.cs.umass.edu/lfw/>`_ Dataset.
    Args:
        root (string): Root directory of dataset where directory
            ``lfw-py`` exists or will be saved to if download is set to True.
        train (bool, optional): If True, creates dataset from "DevTrain" set, otherwise
            creates from "DevTest" set.
        image_set (str, optional): Type of image funneling to use, ``original``, ``funneled`` or
            ``deepfunneled``. Defaults to ``original``.
        transform (callable, optional): A function/transform that  takes in an PIL image
            and returns a transformed version. E.g, ``transforms.RandomCrop``
        target_transform (callable, optional): A function/transform that takes in the
            target and transforms it.
        download (bool, optional): If true, downloads the dataset from the internet and
            puts it in root directory. If dataset is already downloaded, it is not
            downloaded again.
    """

    def __init__(
        self,
        root: str,
        train: bool = True,
        image_set: str = "original",
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
        download: bool = False,
    ):
        super(LFWPairs, self).__init__(root, train, image_set, "pairs",
                                       transform, target_transform, download)

        self.pairs_file = os.path.join(self.root, f"pairsDev{self.split}.txt")
        self.pair_names, self.data, self.targets = self._get_pairs(self.images_dir, self.pairs_file)

    def _get_pairs(self, images_dir, pairs_file):
        with open(pairs_file, 'r') as f:
            lines = f.readlines()
            n_pairs = int(lines[0])
            matched_pairs = [line.strip().split("\t") for line in lines[1: n_pairs + 1]]
            unmatched_pairs = [line.strip().split("\t") for line in lines[n_pairs + 1: 2 * n_pairs + 1]]

            pair_names = []
            data = []
            targets = []
            for pair in matched_pairs:
                img1 = os.path.join(images_dir, pair[0], f"{pair[0]}_{int(pair[1]):04d}.jpg")
                img2 = os.path.join(images_dir, pair[0], f"{pair[0]}_{int(pair[2]):04d}.jpg")
                same = 1  # same = True
                if os.path.exists(img1) and os.path.exists(img2):
                    pair_names.append((pair[0], pair[0]))
                    data.append((img1, img2))
                    targets.append(same)
            for pair in unmatched_pairs:
                img1 = os.path.join(images_dir, pair[0], f"{pair[0]}_{int(pair[1]):04d}.jpg")
                img2 = os.path.join(images_dir, pair[2], f"{pair[2]}_{int(pair[3]):04d}.jpg")
                same = 0  # same = False
                if os.path.exists(img1) and os.path.exists(img2):
                    pair_names.append((pair[0], pair[2]))
                    data.append((img1, img2))
                    targets.append(same)

        return pair_names, data, targets

    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image1, image2, target) where target is `0` for different indentities and `1` for same identities.
        """
        img1, img2 = self.data[index]
        img1, img2 = self.loader(img1), self.loader(img2)
        target = self.targets[index]

        if self.transform is not None:
            img1, img2 = self.transform(img1), self.transform(img2)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return (img1, img2), target

    def extra_repr(self) -> str:
        return "Split: {}".format(self.split)
