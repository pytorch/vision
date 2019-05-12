from __future__ import print_function

import os

from PIL import Image
import torch.utils.data as data

from .utils import download_url, check_integrity
from .vision import VisionDataset


class ssTEM(VisionDataset):
    """Dataset for `ISBI Challenge: Segmentation of neuronal structures
    in EM stacks <http://brainiac2.mit.edu/isbi_challenge/>`_.

    Args:
        root (string): Root directory where dataset exists or will be saved
            to if download is set to True.
        train (bool, optional): If True, creates dataset from training set,
            otherwise creates from test set.
        transforms (callable, optional): A function/transform that takes in
            two PIL images and returns transformed versions.
        download (bool, optional): If true, downloads the dataset from the
            internet and puts it in root directory. If dataset is already
            downloaded, it is not downloaded again.
    """
    base_url = 'http://brainiac2.mit.edu/isbi_challenge/sites/default/files/'

    meta = {
        'train': {
            'data': {
                'filename': 'train-volume.tif',
                'md5': '465461edbe0254630c4ec5577f1e7764'
            },
            'labels': {
                'filename': 'train-labels.tif',
                'md5': '657fe6b728c6dd0152e295c6d800001d'
            }
        },
        'test': {
            'data': {
                'filename': 'test-volume.tif',
                'md5': '9767660d7abe4e0ecbdd0061a16058ad'
            }
        }
    }

    def __init__(self, root, train=True, transforms=None, download=False):
        # Lazy import
        import skimage.io

        super(ssTEM, self).__init__(root, transforms)

        self.train = train

        if download:
            self.download()

        if not self._check_integrity():
            raise RuntimeError('Dataset not found or corrupted.' +
                               ' You can use download=True to download it')

        if train:
            self.data = skimage.io.imread(os.path.join(
                self.root, self.meta['train']['data']['filename']))
            self.labels = skimage.io.imread(os.path.join(
                self.root, self.meta['train']['labels']['filename']))
        else:
            self.data = skimage.io.imread(os.path.join(
                self.root, self.meta['test']['data']['filename']))

    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        img = self.data[index]
        img = Image.fromarray(img)

        target = None
        if self.train:
            target = self.labels[index]
            target = Image.fromarray(img)

        if self.transforms is not None:
            img, target = self.transforms(img, target)

        return img, target

    def __len__(self):
        return len(self.data)

    def _check_integrity(self):
        if self.train:
            dataset = 'train'
            records = ['data', 'labels']
        else:
            dataset = 'test'
            records = ['data']

        for record in records:
            filename = self.meta[dataset][record]['filename']
            fpath = os.path.join(self.root, filename)
            md5 = self.meta[dataset][record]['md5']

            if not check_integrity(fpath, md5):
                return False

        return True

    def download(self):
        if self._check_integrity():
            print('Files already downloaded and verified')
            return

        if self.train:
            dataset = 'train'
            records = ['data', 'labels']
        else:
            dataset = 'test'
            records = ['data']

        for record in records:
            filename = self.meta[dataset][record]['filename']
            md5 = self.meta[dataset][record]['md5']

            download_url(self.base_url + filename, self.root, filename, md5)
