from __future__ import print_function
import torch.utils.data as data
from PIL import Image
import os
import os.path
import numpy as np
from .utils import download_url
from torch.utils.model_zoo import tqdm


class USPS(data.Dataset):
    """`USPS <https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/multiclass.html#usps>`_ Dataset.
    The data-format is : [label [index:value ]*256 \n] * num_lines, where ``label`` lies in ``[1, 10]``.
    The value for each pixel lies in ``[-1, 1]``. Here we transform the ``label`` into ``[0, 9]``
    and make pixel values in ``[0, 255]``.

    Args:
        root (string): Root directory of dataset to store``USPS`` data files.
        split (string): One of {'train', 'test'}.
            Accordingly dataset is selected.
        transform (callable, optional): A function/transform that  takes in an PIL image
            and returns a transformed version. E.g, ``transforms.RandomCrop``
        target_transform (callable, optional): A function/transform that takes in the
            target and transforms it.
        download (bool, optional): If true, downloads the dataset from the internet and
            puts it in root directory. If dataset is already downloaded, it is not
            downloaded again.

    """
    split_list = {
        'train': [
            "https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/multiclass/usps.bz2",
            "usps.bz2", 7291
        ],
        'test': [
            "https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/multiclass/usps.t.bz2",
            "usps.t.bz2", 2007
        ],
    }

    def __init__(self,
                 root,
                 split='train',
                 transform=None,
                 target_transform=None,
                 download=False):
        self.root = os.path.expanduser(root)
        self.transform = transform
        self.target_transform = target_transform
        self.split = split  # training set or test set

        if self.split not in self.split_list:
            raise ValueError('Wrong split entered! Please use split="train" '
                             'or split="test"')

        self.url = self.split_list[split][0]
        self.filename = self.split_list[split][1]
        self.total_images = self.split_list[split][2]

        full_path = os.path.join(self.root, self.filename)

        if download and not os.path.exists(full_path):
            self.download()

        import bz2
        fp = bz2.open(full_path)

        datas = []
        targets = []
        for line in tqdm(
                fp, desc='processing data', total=self.total_images):
            label, *pixels = line.decode().split()
            pixels = [float(x.split(':')[-1]) for x in pixels]
            im = np.asarray(pixels).reshape((16, 16))
            im = (im + 1) / 2 * 255
            im = im.astype(dtype=np.uint8)
            datas.append(im)
            targets.append(int(label) - 1)

        assert len(targets) == self.total_images, \
            'total number of images are wrong! maybe the download is corrupted?'

        self.data = np.stack(datas, axis=0)
        self.targets = targets
        self.labels = list(range(10))

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        img, target = self.data[index], int(self.targets[index])

        # doing this so that it is consistent with all other datasets
        # to return a PIL Image
        img = Image.fromarray(img, mode='L')

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target

    def __len__(self):
        return len(self.data)

    def download(self):
        download_url(self.url, self.root, self.filename, md5=None)

    def __repr__(self):
        fmt_str = 'Dataset ' + self.__class__.__name__ + '\n'
        fmt_str += '    Number of datapoints: {}\n'.format(self.__len__())
        fmt_str += '    Split: {}\n'.format(self.split)
        fmt_str += '    Root Location: {}\n'.format(self.root)
        tmp = '    Transforms (if any): '
        fmt_str += '{0}{1}\n'.format(
            tmp,
            self.transform.__repr__().replace('\n', '\n' + ' ' * len(tmp)))
        tmp = '    Target Transforms (if any): '
        fmt_str += '{0}{1}'.format(
            tmp,
            self.target_transform.__repr__().replace('\n',
                                                     '\n' + ' ' * len(tmp)))
        return fmt_str
