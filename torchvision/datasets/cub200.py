from __future__ import print_function
import os
import errno
import torch
import numpy as np
from PIL import Image
import torch.utils.data as data


def pil_loader(path):
    with open(path, 'rb') as f:
        with Image.open(f) as img:
            return img.convert('RGB')


def accimage_loader(path):
    import accimage
    try:
        return accimage.Image(path)
    except IOError:
        return pil_loader(path)


def default_loader(path):
    from torchvision import get_image_backend
    if get_image_backend() == 'accimage':
        return accimage_loader(path)
    else:
        return pil_loader(path)


def build_set(root, year, train):
    """
       Function to return the lists of paths with the corresponding labels for the images

    Args:
        root (string): Root directory of dataset
        year (int): Year/version of the dataset. Available options are 2010 and 2011
        train (bool, optional): If true, returns the list pertaining to training images and labels, else otherwise
    Returns:
        return_list: list of 2-tuples with 1st location specifying path and 2nd location specifying the class
    """
    if year == 2010:
        images_file_path = os.path.join(root, 'images/')

        if train:
            lists_path = os.path.join(root, 'lists/train.txt')
        else:
            lists_path = os.path.join(root, 'lists/test.txt')

        files = np.genfromtxt(lists_path, dtype=str)

        imgs = []
        classes = []
        class_to_idx = []

        for fname in files:
            full_path = os.path.join(images_file_path, fname)
            imgs.append((full_path, int(fname[0:3]) - 1))
            if os.path.split(fname)[0][4:] not in classes:
                classes.append(os.path.split(fname)[0][4:])
                class_to_idx.append(int(fname[0:3]) - 1)

        return imgs, classes, class_to_idx

    elif year == 2011:
        images_file_path = os.path.join(root, 'CUB_200_2011/images/')

        all_images_list_path = os.path.join(root, 'CUB_200_2011/images.txt')
        all_images_list = np.genfromtxt(all_images_list_path, dtype=str)
        train_test_list_path = os.path.join(root, 'CUB_200_2011/train_test_split.txt')
        train_test_list = np.genfromtxt(train_test_list_path, dtype=int)

        imgs = []
        classes = []
        class_to_idx = []

        for i in range(0, len(all_images_list)):
            fname = all_images_list[i, 1]
            full_path = os.path.join(images_file_path, fname)
            if train_test_list[i, 1] == 1 and train:
                imgs.append((full_path, int(fname[0:3]) - 1))
            elif train_test_list[i, 1] == 0 and not train:
                imgs.append((full_path, int(fname[0:3]) - 1))
            if os.path.split(fname)[0][4:] not in classes:
                classes.append(os.path.split(fname)[0][4:])
                class_to_idx.append(int(fname[0:3]) - 1)

        return imgs, classes, class_to_idx


class CUB200(data.Dataset):
    """`CUB200 <http://www.vision.caltech.edu/visipedia/CUB-200.html>`_ Dataset.
       `CUB200 <http://www.vision.caltech.edu/visipedia/CUB-200-2011.html>`_ Dataset.

    Args:
        root (string): Root directory of dataset the images and corresponding lists exist
            inside raw folder
        train (bool, optional): If True, creates dataset from ``training.pt``,
            otherwise from ``test.pt``.
        download (bool, optional): If true, downloads the dataset from the internet and
            puts it in root directory. If dataset is already downloaded, it is not
            downloaded again.
        transform (callable, optional): A function/transform that  takes in an PIL image
            and returns a transformed version. E.g, ``transforms.RandomCrop``
        target_transform (callable, optional): A function/transform that takes in the
            target and transforms it.
        year (int): Year/version of the dataset. Available options are 2010 and 2011
    """
    urls = []
    raw_folder = 'raw'

    def __init__(self, root, year, train=True, transform=None, target_transform=None, download=False,
                 loader=default_loader):
        self.root = os.path.expanduser(root)
        self.transform = transform
        self.target_transform = target_transform
        self.train = train
        self.year = year
        self.loader = loader

        assert year == 2010 or year == 2011, "Invalid version of CUB200 dataset"
        if year == 2010:
            self.urls = ['http://www.vision.caltech.edu/visipedia-data/CUB-200/images.tgz',
                         'http://www.vision.caltech.edu/visipedia-data/CUB-200/lists.tgz']

        elif year == 2011:
            self.urls = ['http://www.vision.caltech.edu/visipedia-data/CUB-200-2011/CUB_200_2011.tgz']

        if download:
            self.download()

        if not self._check_exists():
            raise RuntimeError('Dataset not found. You can use download=True to download it')

        self.imgs, self.classes, self.class_to_idx = build_set(os.path.join(self.root, self.raw_folder),
                                                               self.year, self.train)

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        path, target = self.data_set[index]
        img = self.loader(path)

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            img = self.target_transform(img)

        return img, target

    def _check_exists(self):
        pth = os.path.join(self.root, self.raw_folder)
        if self.year == 2010:
            return os.path.exists(os.path.join(pth, 'images/')) and os.path.exists(os.path.join(pth, 'lists/'))
        elif self.year == 2011:
            return os.path.exists(os.path.join(pth, 'CUB_200_2011/'))

    def __len__(self):
        return len(self.imgs)

    def download(self):
        from six.moves import urllib
        import tarfile

        if self._check_exists():
            return

        try:
            os.makedirs(os.path.join(self.root, self.raw_folder))
        except OSError as e:
            if e.errno == errno.EEXIST:
                pass
            else:
                raise

        for url in self.urls:
            print('Downloading ' + url)
            data = urllib.request.urlopen(url)
            filename = url.rpartition('/')[2]
            file_path = os.path.join(self.root, self.raw_folder, filename)
            with open(file_path, 'wb') as f:
                f.write(data.read())
            tar = tarfile.open(file_path, 'r')
            for item in tar:
                tar.extract(item, file_path.replace(filename, ''))
            os.unlink(file_path)

        print('Done!')
