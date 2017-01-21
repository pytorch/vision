from __future__ import print_function

import errno
import hashlib
import os
import sys
import tarfile

import torch.utils.data as data
from PIL import Image

from six.moves import urllib


class VOCSegmentation(data.Dataset):
    CLASSES = [
        'background', 'aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus',
        'car', 'cat', 'chair', 'cow', 'diningtable', 'dog', 'horse',
        'motorbike', 'person', 'potted-plant', 'sheep', 'sofa', 'train',
        'tv/monitor', 'ambigious'
    ]

    URL = "http://host.robots.ox.ac.uk/pascal/VOC/voc2012/VOCtrainval_11-May-2012.tar"
    FILE = "VOCtrainval_11-May-2012.tar"
    MD5 = '6cd6e144f989b92b3379bac3b3de84fd'
    BASE_DIR = 'VOCdevkit/VOC2012'

    def __init__(self,
                 root,
                 train=True,
                 transform=None,
                 target_transform=None,
                 download=False):
        self.root = root
        _voc_root = os.path.join(self.root, self.BASE_DIR)
        _mask_dir = os.path.join(_voc_root, 'SegmentationClass')
        _image_dir = os.path.join(_voc_root, 'JPEGImages')
        self.transform = transform
        self.target_transform = target_transform
        self.train = train

        if download:
            self._download()

        if not self._check_integrity():
            raise RuntimeError('Dataset not found or corrupted.' +
                               ' You can use download=True to download it')
        # train/val/test splits are pre-cut
        _splits_dir = os.path.join(_voc_root, 'ImageSets/Segmentation')
        _split_f = os.path.join(_splits_dir, 'train.txt')
        if not self.train:
            _split_f = os.path.join(_splits_dir, ' trainval.txt')

        self.images = []
        self.masks = []
        with open(os.path.join(_split_f), "r") as lines:
            for line in lines:
                _image = os.path.join(_image_dir, line.rstrip('\n') + ".jpg")
                _mask = os.path.join(_mask_dir, line.rstrip('\n') + ".png")
                assert os.path.isfile(_image)
                assert os.path.isfile(_mask)
                self.images.append(_image)
                self.masks.append(_mask)

        assert (len(self.images) == len(self.masks))

    def __getitem__(self, index):
        _img = Image.open(self.images[index]).convert('RGB')
        _target = Image.open(self.masks[index])

        if self.transform is not None:
            print("transform was not none")
            _img = self.transform(_img)
        # todo(bdd) : perhaps transformations should be applied differently to masks? 
        if self.target_transform is not None:
            _target = self.target_transform(_target)

        return _img, _target

    def __len__(self):
        return len(self.images)

    def _check_integrity(self):
        _fpath = os.path.join(self.root, self.FILE)
        if not os.path.isfile(_fpath):
            print("{} does not exist".format(_fpath))
            return False
        _md5c = hashlib.md5(open(_fpath, 'rb').read()).hexdigest()
        if _md5c != self.MD5:
            print(" MD5({}) did not match MD5({}) expected for {}".format(
                _md5c, self.MD5, _fpath))
            return False
        return True

    def _download(self):
        _fpath = os.path.join(self.root, self.FILE)

        try:
            os.makedirs(self.root)
        except OSError as e:
            if e.errno == errno.EEXIST:
                pass
            else:
                raise

        if self._check_integrity():
            print('Files already downloaded and verified')
            return
        else:
            print('Downloading ' + self.URL + ' to ' + _fpath)

            def _progress(count, block_size, total_size):
                sys.stdout.write('\r>> %s %.1f%%' %
                                 (_fpath, float(count * block_size) /
                                  float(total_size) * 100.0))
                sys.stdout.flush()

            urllib.request.urlretrieve(self.URL, _fpath, _progress)

        # extract file
        cwd = os.getcwd()
        print('Extracting tar file')
        tar = tarfile.open(_fpath)
        os.chdir(self.root)
        tar.extractall()
        tar.close()
        os.chdir(cwd)
        print('Done!')


if __name__ == '__main__':
    # todo(bdd) : sanity checking seen in tests/cifar.py ... remove before merging,
    pascal = VOCSegmentation('/tmp/pascal-voc/')
    print(pascal[3])
    # (<PIL.Image.Image image mode=RGB size=500x375 at 0x7EFED5975D10>, <PIL.Image.Image image mode=RGB size=500x375 at 0x7EFED5975D90>)
    # import torch
    # import torchvision.transforms as transforms
    # transform = transforms.ToTensor()
    # dataset = VOCSegmentation(
    #     '/tmp/pascal-voc/', transform=transform, target_transform=transform)
    # dataloader = torch.utils.data.DataLoader(
    #     dataset, batch_size=1, shuffle=True, num_workers=2)

    # for i, data in enumerate(dataloader):
    #     print(data)
    #     if i == 10:
    #         break
    # miter = dataloader.__iter__()

    # def getBatch():
    #     global miter
    #     try:
    #         return miter.next()
    #     except StopIteration:
    #         miter = dataloader.__iter__()
    #         return miter.next()

    # i = 0
    # while True:
    #     print(i)
    #     img, target = getBatch()
    #     i += 1
    # print(*pascal.CLASSES, sep='\n')
    # print(*pascal.images, sep='\n')
    # print(*pascal.masks, sep='\n')
