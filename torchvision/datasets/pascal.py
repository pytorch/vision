from __future__ import print_function

import errno
import hashlib
import os
import sys
import tarfile

import torch.utils.data as data
from PIL import Image

from six.moves import urllib


class PascalVOC(data.Dataset):
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
        voc_root = os.path.join(self.root, self.BASE_DIR)
        mask_dir = os.path.join(voc_root, 'SegmentationClass')
        image_dir = os.path.join(voc_root, 'JPEGImages')
        self.transform = transform
        self.target_transform = target_transform
        self.train = train

        if download:
            self.download()

        if not self._check_integrity():
            raise RuntimeError('Dataset not found or corrupted.' +
                               ' You can use download=True to download it')
        # train/val/test splits are pre-cut
        splits_dir = os.path.join(voc_root, 'ImageSets/Segmentation')
        split_f = os.path.join(splits_dir, 'train.txt')
        if not self.train:
            split_f = os.path.join(splits_dir, ' trainval.txt')

        self.images = []
        self.masks = []
        with open(os.path.join(split_f), "r") as lines:
            for line in lines:
                image = os.path.join(image_dir, line.rstrip('\n') + ".jpg")
                mask = os.path.join(mask_dir, line.rstrip('\n') + ".png")
                assert os.path.isfile(image)
                assert os.path.isfile(mask)
                self.images.append(image)
                self.masks.append(mask)

        assert (len(self.images) == len(self.masks))

    def __getitem__(self, index):
        img = Image.open(self.images[index]).convert('RGB')
        target = Image.open(self.masks[index]).convert('RGB')
        if self.transform is not None:
            img = self.transform(img)
        # todo(bdd) : perhaps transformations should be applied differently to masks? 
        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target

    def __len__(self):
        return len(self.images)

    def _check_integrity(self):
        fpath = os.path.join(self.root, self.FILE)
        if not os.path.isfile(fpath):
            print("{} does not exist".format(fpath))
            return False
        md5c = hashlib.md5(open(fpath, 'rb').read()).hexdigest()
        if md5c != self.MD5:
            print(" MD5({}) did not match MD5({}) expected for {}".format(
                md5c, self.MD5, fpath))
            return False
        return True

    def download(self):
        fpath = os.path.join(self.root, self.FILE)

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

        # downloads file
        if os.path.isfile(fpath) and \
           hashlib.md5(open(fpath, 'rb').read()).hexdigest() == self.MD5:
            print('Using downloaded file: ' + fpath)
        else:
            print('Downloading ' + self.URL + ' to ' + fpath)

            def _progress(count, block_size, total_size):
                sys.stdout.write('\r>> %s %.1f%%' %
                                 (fpath, float(count * block_size) /
                                  float(total_size) * 100.0))
                sys.stdout.flush()

            urllib.request.urlretrieve(self.URL, fpath, _progress)

        # extract file
        cwd = os.getcwd()
        print('Extracting tar file')
        tar = tarfile.open(fpath)
        os.chdir(self.root)
        tar.extractall()
        tar.close()
        os.chdir(cwd)
        print('Done!')


if __name__ == '__main__':
    # todo(bdd) : sanity checking, remove before merging
    pascal = PascalVOC('/tmp/pascal-voc/')
    # print(*pascal.CLASSES, sep='\n')
    # print(*pascal.images, sep='\n')
    # print(*pascal.masks, sep='\n')
