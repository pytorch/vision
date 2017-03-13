from __future__ import print_function
import scipy.io as sio
import torch.utils.data as data
from PIL import Image
import os
import os.path
import errno
import numpy as np
import sys


class SVHN(data.Dataset):
    url = ""
    filename = ""
    file_md5 = ""

    split_list = {
        'train': ["http://ufldl.stanford.edu/housenumbers/train_32x32.mat",
                  "train_32x32.mat", "e26dedcc434d2e4c54c9b2d4a06d8373"],
        'test': ["http://ufldl.stanford.edu/housenumbers/test_32x32.mat",
                 "test_32x32.mat", "eb5a983be6a315427106f1b164d9cef3"],
        'extra': ["http://ufldl.stanford.edu/housenumbers/extra_32x32.mat",
                  "extra_32x32.mat", "a93ce644f1a588dc4d68dda5feec44a7"]}

    def __init__(self, root, split='train', transform=None, target_transform=None, download=False):
        self.root = root
        self.transform = transform
        self.target_transform = target_transform
        self.split = split  # training set or test set or extra set

        if self.split in self.split_list:
            self.url = self.split_list[split][0]
            self.filename = self.split_list[split][1]
            self.file_md5 = self.split_list[split][2]

            if download:
                self.download()

            if not self._check_integrity():
                    raise RuntimeError('Dataset not found or corrupted.' +
                                       ' You can use download=True to download it')

            # reading(loading) mat file as array
            loaded_mat = sio.loadmat(os.path.join(root, self.filename))

            if self.split != 'test':
                self.train_data = loaded_mat['X']
                self.train_labels = loaded_mat['y']
                self.train_data = np.transpose(self.train_data, (3, 2, 1, 0))
            else:
                self.test_data = loaded_mat['X']
                self.test_labels = loaded_mat['y']
                self.test_data = np.transpose(self.test_data, (3, 2, 1, 0))
        else:
            print ("Wrong dataset entered! Please use split=train or split=extra or split=test")

    def __getitem__(self, index):
        if self.split == 'train' or self.split == 'extra':
            img, target = self.train_data[index], self.train_labels[index]
        elif self.split == 'test':
            img, target = self.test_data[index], self.test_labels[index]

        # doing this so that it is consistent with all other datasets
        # to return a PIL Image
        img = Image.fromarray(np.transpose(img, (1, 2, 0)))

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target

    def __len__(self):
        return len(self.train_data)

    def _check_integrity(self):
        import hashlib
        root = self.root
        md5 = self.split_list[self.split][2]
        fpath = os.path.join(root, self.filename)
        if not os.path.isfile(fpath):
            return False
        md5c = hashlib.md5(open(fpath, 'rb').read()).hexdigest()
        if md5c != md5:
            return False
        return True

    def download(self):
        from six.moves import urllib
        import tarfile
        import hashlib

        root = self.root
        fpath = os.path.join(root, self.filename)

        try:
            os.makedirs(root)
        except OSError as e:
            if e.errno == errno.EEXIST:
                pass
            else:
                raise

        if self._check_integrity():
            print('Files already downloaded and verified')
            return

        print ("about to download")
        # downloads file
        if os.path.isfile(fpath):
            print('Using downloaded file: ' + fpath)
        else:
            print('Downloading ' + self.url + ' to ' + fpath)
            urllib.request.urlretrieve(self.url, fpath)
            print ('Downloaded!')
