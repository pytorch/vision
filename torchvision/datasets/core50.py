#!/usr/bin/env python
# -*- coding: utf-8 -*-

################################################################################
# Date: 22-11-2017                                                             #
# Author: Vincenzo Lomonaco                                                    #
# E-mail: vincenzo.lomonaco@unibo.it                                           #
# Website: vincenzolomonaco.com                                                #
################################################################################

""" CORe50 Dataset class """

from __future__ import print_function
import os
import os.path
import torch.utils.data as data
from .utils import download_url, check_integrity
from torchvision.datasets.folder import pil_loader


class CORE50(data.Dataset):
    """`CORE50 <https://vlomonaco.github.io/core50/>`_ Dataset, specifically
        designed for Continuous Learning and Robotic Vision applications.
        For more information and additional materials visit the official
        website `CORE50 <https://vlomonaco.github.io/core50/>`

    Args:
        root (string): Root directory of the dataset where the ``CORe50``
            dataset exists or should be downloaded.
        check_integrity (bool, optional): If True check the integrity of the
            Dataset before trying to load it.
        scenario (string, optional): One of the three scenarios of the CORe50
            benchmark ``ni``, ``nc`` or ``nic``.
        train (bool, optional): If True, creates the dataset from the training
            set, otherwise creates from test set.
        img_size (string, optional): One of the two img sizes available among
            ``128x128`` or ``350x350``.
        cumul (bool, optional): If True the cumulative scenario is assumed, the
            incremental scenario otherwise. Practically speaking ``cumul=True``
            means that for batch=i also batch=0,...i-1 will be added to the
            available training data.
        run (int, optional): One of the 10 runs (from 0 to 9) in which the
            training batch order is changed as in the official benchmark.
        batch (int, optional): One of the training incremental batches from 0 to
            max-batch - 1. Remember that for the ``ni``, ``nc`` and ``nic`` we
            have respectively 8, 9 and 79 incremental batches. If
            ``train=False`` this parameter will be ignored.
        transform (callable, optional): A function/transform that takes in an
            PIL image and returns a transformed version. E.g,
            ``transforms.ToTensor()``
        target_transform (callable, optional): A function/transform that takes
            in the target and transforms it.
        download (bool, optional): If true, downloads the dataset from the
            internet and puts it in root directory. If dataset is already
            downloaded, it is not downloaded again.

    Example:

        .. code:: python

            training_data = datasets.CORE50(
                '~/data/core50', transform=transforms.ToTensor(), download=True
            )
            training_loader = torch.utils.data.DataLoader(
                training_data, batch_size=128, shuffle=True, num_workers=4
            )
            test_data = datasets.CORE50(
                '~/data/core50', transform=transforms.ToTensor(), train=False,
                download=True
            )
            test_loader = torch.utils.data.DataLoader(
                training_data, batch_size=128, shuffle=True, num_workers=4
            )

            for batch in training_loader:
                imgs, labels = batch
                ...

        This is the simplest way of using the Dataset with the common Train/Test
        split. If you want to use the benchmark as in the original CORe50 paper
        (that is for continuous learning) you need to play with the parameters
        ``scenario``, ``cumul``, ``run`` and ``batch`` hence creating a number
        of Dataset objects (one for each incremental training batch and one for
        the test set).

    """
    ntrain_batch = {
        'ni': 8,
        'nc': 9,
        'nic': 79
    }
    urls = {
        '128x128': 'http://bias.csr.unibo.it/maltoni/download/core50/'
                   'core50_128x128.zip',
        '350x350': 'http://bias.csr.unibo.it/maltoni/download/core50/'
                   'core50_350x350.zip',
        'filelists': 'https://vlomonaco.github.io/core50/data/'
                     'batches_filelists.zip'
    }
    filenames = {
        '128x128': 'core50_128x128.zip',
        '350x350': 'core50_350x350.zip',
        'filelists': 'batches_filelists.zip'
    }
    md5s = {
        'core50_128x128.zip': '745f3373fed08d69343f1058ee559e13',
        'core50_350x350.zip': 'e304258739d6cd4b47e19adfa08e7571',
        'batches_filelists.zip': 'e3297508a8998ba0c99a83d6b36bde62'
    }

    def __init__(self, root, check_integrity=True, scenario='ni', train=True,
                 img_size='128x128', run=0, batch=7, cumul=True, transform=None,
                 target_transform=None, download=False):

        self.root = os.path.expanduser(root)
        self.img_size = img_size
        self.scenario = scenario
        self.run = run
        self.batch = batch
        self.transform = transform
        self.target_transform = target_transform

        # To be filled
        self.fpath = None
        self.img_paths = []
        self.labels = []

        # Downloading files if needed
        if download:
            self.download()

        if check_integrity:
            print("Making sure CORe50 exists and it's not corrupted...")
            if not self._check_integrity():
                raise RuntimeError('Dataset not found or corrupted.' +
                                   ' You can use download=True to download it')

        if cumul:
            suffix = 'cum'
        else:
            suffix = 'inc'

        if train:
            self.fpath = os.path.join(
                scenario.upper() + '_' + suffix, 'run' + str(run),
                'train_batch_' + str(batch).zfill(2) + '_filelist.txt'
            )
        else:
            # it's the last one, hence the test batch
            self.fpath = os.path.join(
                scenario.upper() + '_' + suffix, 'run' + str(run),
                'test_filelist.txt'
            )

        # Loading the filelist
        path = os.path.join(self.root, self.filenames['filelists'][:-4],
                            self.fpath)
        with open(path, 'r') as f:
            for i, line in enumerate(f):
                if line.strip():
                    path, label = line.split()
                    self.labels.append(int(label))
                    self.img_paths.append(path)

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is index of the target class.
        """

        fpath = self.img_paths[index]
        target = self.labels[index]
        img = pil_loader(
            os.path.join(self.root, self.filenames[self.img_size][:-4], fpath)
        )

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target

    def __len__(self):

        return len(self.labels)

    def _check_integrity(self):

        # removing files we have not downloaded
        if self.img_size == '128x128':
            del self.md5s['core50_350x350.zip']
        else:
            del self.md5s['core50_128x128.zip']

        root = self.root
        for filename, md5 in self.md5s.items():
            fpath = os.path.join(root, filename)
            if not check_integrity(fpath, md5):
                return False
        return True

    def download(self):
        import zipfile

        if self._check_integrity():
            print('Files already downloaded and verified')
            return

        root = self.root

        # Downloading the dataset and filelists
        for name in (self.img_size, 'filelists'):
            download_url(
                self.urls[name], root, self.filenames[name],
                self.md5s[self.filenames[name]]
            )

            # extract file
            cwd = os.getcwd()
            zip = zipfile.ZipFile(os.path.join(root, self.filenames[name]), "r")
            os.chdir(root)
            zip.extractall()
            zip.close()
            os.chdir(cwd)
