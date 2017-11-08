from __future__ import print_function
from PIL import Image
from functools import reduce
import os
import random
import torch.utils.data as data
from .utils import download_url, check_integrity, list_dir, list_files


class Omniglot(data.Dataset):
    """`Omniglot <https://github.com/brendenlake/omniglot>`_ Dataset.
    Args:
        root (string): Root directory of dataset where directory
            ``omniglot-py`` exists.
        background (bool, optional): If True, creates dataset from the "background" set, otherwise
            creates from the "evaluation" set. This terminology is defined by the authors.
        transform (callable, optional): A function/transform that  takes in an PIL image
            and returns a transformed version. E.g, ``transforms.RandomCrop``
        target_transform (callable, optional): A function/transform that takes in the
            target and transforms it.
        download (bool, optional): If true, downloads the dataset zip files from the internet and
            puts it in root directory. If the zip files are already downloaded, they are not
            downloaded again.
        force_extract (bool, optional): If true, extracts the downloaded zip file irrespective
            of the existence of an extracted folder with the same name
    """
    folder = 'omniglot-py'
    download_url_prefix = 'https://github.com/brendenlake/omniglot/raw/master/python'
    zips_md5 = [
        ['images_background', '68d2efa1b9178cc56df9314c21c6e718'],
        ['images_evaluation', '6b91aef0f799c5bb55b94e3f2daec811'],
    ]

    def __init__(self, root, background=True,
                 transform=None, target_transform=None,
                 download=False,
                 force_extract=False):
        self.root = os.path.join(os.path.expanduser(root), self.folder)
        self.background = background
        self.transform = transform
        self.target_transform = target_transform

        if download:
            self.download(force_extract)

        if not self._check_integrity():
            raise RuntimeError('Dataset not found or corrupted.' +
                               ' You can use download=True to download it')

        self.target_folder = os.path.join(self.root, self._get_target_folder())
        self._alphabets = list_dir(self.target_folder)
        self._characters = sum(
            [
                [
                    os.path.join(alphabet, character)
                    for character in list_dir(os.path.join(self.target_folder, alphabet))
                ]
                for alphabet in self._alphabets
            ],
            []
        )
        self._character_images = [
            [
                (image, idx)
                for image in list_files(os.path.join(self.target_folder, character), '.png')
            ]
            for idx, character in enumerate(self._characters)
        ]
        self._flat_character_images = sum(self._character_images, [])

    def __len__(self):
        return len(self._flat_character_images)

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is index of the target character class.
        """
        image_name, character_class = self._flat_character_images[index]
        image_path = os.path.join(self.target_folder, self._characters[character_class], image_name)
        image = Image.open(image_path, mode='r').convert('L')

        if self.transform:
            image = self.transform(image)

        if self.target_transform:
            character_class = self.target_transform(character_class)

        return image, character_class

    def _check_integrity(self):
        for fzip in self.zips_md5:
            filename, md5 = fzip[0] + '.zip', fzip[1]
            fpath = os.path.join(self.root, filename)
            if not check_integrity(fpath, md5):
                return False
        return True

    def download(self, force_extract=False):
        import zipfile

        if self._check_integrity():
            print('Files already downloaded and verified')
            return

        for fzip in self.zips_md5:
            filename, md5 = fzip[0], fzip[1]
            zip_filename = filename + '.zip'
            url = self.download_url_prefix + '/' + zip_filename
            download_url(url, self.root, zip_filename, md5)

            if not os.path.isdir(os.path.join(self.root, filename)) or force_extract is True:
                print('Extracting downloaded file: ' + os.path.join(self.root, zip_filename))
                with zipfile.ZipFile(os.path.join(self.root, zip_filename), 'r') as zip_file:
                    zip_file.extractall(self.root)

    def _get_target_folder(self):
        return 'images_background' if self.background is True else 'images_evaluation'


class OmniglotRandomPair(Omniglot):
    """`OmniglotRandomPair <https://github.com/brendenlake/omniglot>`_ Dataset.

    This is a subclass of the Omniglot dataset. This instead it returns
    a randomized pair of images with similarity label (0 or 1)

    Args:
        pair_count (int, optional): The total number of image pairs to generate. Defaults to
            10000
    """
    def __init__(self, *args, pair_count=10000, **kwargs):
        super(self.__class__, self).__init__(*args, **kwargs)

        self.pair_count = pair_count
        self._precompute_pairs()

    def __len__(self):
        return len(self.pairs_list)

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image0, image1, is_match) a random pair of images from the Omniglot characters
                with corresponding label 1 if it is matching pair and 0 otherwise
        """

        target_pair, is_match = self.pairs_list[index]
        target_image_names = [self._character_images[i][j] for i, j in target_pair]
        target_image_paths = [
            os.path.join(self.target_folder, self._characters[cid], name)
            for name, cid in target_image_names
        ]
        images = [Image.open(path, mode='r').convert('L') for path in target_image_paths]

        if self.transform is not None:
            images = [self.transform(image) for image in images]

        if self.target_transform is not None:
            is_match = self.target_transform(is_match)

        return images[0], images[1], is_match

    def _precompute_pairs(self):
        """A utility wrapper to randomly generate pairs of images

        Args:

        Returns:
            list(tuple((cid0, id0), (cid1, id1), is_match)), a list of 3-tuples where the first two
                items of the tuple contains a character id and corresponding randomly chose image id
                and the last item is 1 or 0 based on whether the image pair is from the same character
                or not respectively
        """
        is_match = [random.randint(0, 1) for _ in range(self.pair_count)]

        cid0_list = [random.randint(0, len(self._characters) - 1) for _ in range(self.pair_count)]
        c0_list = [random.randint(0, len(self._character_images[cid]) - 1) for cid in cid0_list]

        cid1_list = [
            cid0_list[idx] if is_match[idx] == 1 else self._generate_pair(cid0_list[idx])
            for idx in range(self.pair_count)
        ]
        c1_list = [random.randint(0, len(self._character_images[cid]) - 1) for cid in cid1_list]

        self.pairs_list = [
            (((cid0_list[idx], c0_list[idx]), (cid1_list[idx], c1_list[idx])), is_match[idx])
            for idx in range(self.pair_count)
        ]

    def _generate_pair(self, character_id):
        pair_id = random.randint(0, len(self._characters) - 1)
        while pair_id == character_id:
            pair_id = random.randint(0, len(self._characters) - 1)
        return pair_id


