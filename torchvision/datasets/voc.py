import os
import sys
import tarfile
import torch.utils.data as data
if sys.version_info[0] == 2:
    import xml.etree.cElementTree as ET
else:
    import xml.etree.ElementTree as ET

from PIL import Image
from .utils import download_url, check_integrity

VOC_CLASSES = [
    'background', 'aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus',
    'car', 'cat', 'chair', 'cow', 'diningtable', 'dog', 'horse', 'motorbike',
    'person', 'pottedplant', 'sheep', 'sofa', 'train', 'tvmonitor'
]
DATASET_YEAR_DICT = {
    '2012': [
        'http://host.robots.ox.ac.uk/pascal/VOC/voc2012/VOCtrainval_11-May-2012.tar',
        'VOCtrainval_11-May-2012.tar', '6cd6e144f989b92b3379bac3b3de84fd',
        ' VOCdevkit/VOC2012'
    ],
    '2011': [
        'http://host.robots.ox.ac.uk/pascal/VOC/voc2011/VOCtrainval_25-May-2011.tar',
        'VOCtrainval_25-May-2011.tar', '6c3384ef61512963050cb5d687e5bf1e',
        'TrainVal/VOCdevkit/VOC2011'
    ],
    '2010': [
        'http://host.robots.ox.ac.uk/pascal/VOC/voc2010/VOCtrainval_03-May-2010.tar',
        'VOCtrainval_03-May-2010.tar', 'da459979d0c395079b5c75ee67908abb',
        'VOCdevkit/VOC2010'
    ],
    '2009': [
        'http://host.robots.ox.ac.uk/pascal/VOC/voc2009/VOCtrainval_11-May-2009.tar',
        'VOCtrainval_11-May-2009.tar', '59065e4b188729180974ef6572f6a212',
        'VOCdevkit/VOC2009'
    ],
    '2008': [
        'http://host.robots.ox.ac.uk/pascal/VOC/voc2008/VOCtrainval_14-Jul-2008.tar',
        'VOCtrainval_14-Jul-2008.tar', '2629fa636546599198acfcfbfcf1904a',
        'VOCdevkit/VOC2008'
    ],
    '2007': [
        'http://host.robots.ox.ac.uk/pascal/VOC/voc2007/VOCtrainval_06-Nov-2007.tar',
        'VOCtrainval_06-Nov-2007.tar', 'c52e279531787c972589f7e41ab4ae64',
        'VOCdevkit/VOC2007'
    ]
}


class VOCSegmentation(data.Dataset):
    """`Pascal VOC <http://host.robots.ox.ac.uk/pascal/VOC/>`_ Segmentation Dataset.

    Args:
        root (string): Root directory of the VOC Dataset.
        year (string, optional): The dataset year, supports years 2007 to 2012.
        image_set (string, optional): Select the image_set to use, ``train``, ``trainval`` or ``val``
        download (bool, optional): If true, downloads the dataset from the internet and
            puts it in root directory. If dataset is already downloaded, it is not
            downloaded again.
        transform (callable, optional): A function/transform that  takes in an PIL image
            and returns a transformed version. E.g, ``transforms.RandomCrop``
        target_transform (callable, optional): A function/transform that takes in the
            target and transforms it.
    """

    def __init__(self,
                 root,
                 year='2012',
                 image_set='train',
                 download=False,
                 transform=None,
                 target_transform=None):
        self.root = root
        self.year = year
        self.url = DATASET_YEAR_DICT[year][0]
        self.filename = DATASET_YEAR_DICT[year][1]
        self.md5 = DATASET_YEAR_DICT[year][2]
        self.transform = transform
        self.target_transform = target_transform
        self.image_set = image_set
        _base_dir = DATASET_YEAR_DICT[year][3]
        _voc_root = os.path.join(self.root, _base_dir)
        _image_dir = os.path.join(_voc_root, 'JPEGImages')
        _mask_dir = os.path.join(_voc_root, 'SegmentationClass')

        if download:
            download_extract(self.url, self.root, self.filename, self.md5)

        if not os.path.isdir(_voc_root):
            raise RuntimeError('Dataset not found or corrupted.' +
                               ' You can use download=True to download it')

        _splits_dir = os.path.join(_voc_root, 'ImageSets/Segmentation')

        _split_f = os.path.join(_splits_dir, image_set.rstrip('\n') + '.txt')

        if not os.path.exists(_split_f):
            raise ValueError(
                'Wrong image_set entered! Please use image_set="train" '
                'or image_set="trainval" or image_set="val"')

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
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is the image segmentation.
        """
        _img = Image.open(self.images[index]).convert('RGB')
        _target = Image.open(self.masks[index])

        if self.transform is not None:
            _img = self.transform(_img)

        if self.target_transform is not None:
            _target = self.target_transform(_target)

        return _img, _target

    def __len__(self):
        return len(self.images)


class VOCDetection(data.Dataset):
    """`Pascal VOC <http://host.robots.ox.ac.uk/pascal/VOC/>`_ Detection Dataset.

    Args:
        root (string): Root directory of the VOC Dataset.
        year (string, optional): The dataset year, supports years 2007 to 2012.
        image_set (string, optional): Select the image_set to use, ``train``, ``trainval`` or ``val``
        download (bool, optional): If true, downloads the dataset from the internet and
            puts it in root directory. If dataset is already downloaded, it is not
            downloaded again.
        class_to_ind (dict, optional): dictionary lookup of classnames -> indexes
            (default: alphabetic indexing of VOC's 20 classes).
        keep_difficult (boolean, optional): keep difficult instances or not.
        transform (callable, optional): A function/transform that  takes in an PIL image
            and returns a transformed version. E.g, ``transforms.RandomCrop``
        target_transform (callable, optional): A function/transform that takes in the
            target and transforms it.
    """

    def __init__(self,
                 root,
                 year='2012',
                 image_set='train',
                 download=False,
                 class_to_ind=None,
                 keep_difficult=False,
                 transform=None,
                 target_transform=None):
        self.root = root
        self.year = year
        self.url = DATASET_YEAR_DICT[year][0]
        self.filename = DATASET_YEAR_DICT[year][1]
        self.md5 = DATASET_YEAR_DICT[year][2]
        self.transform = transform
        self.target_transform = target_transform
        self.image_set = image_set
        self.class_to_ind = class_to_ind or dict(
            zip(VOC_CLASSES, range(len(VOC_CLASSES))))
        self.keep_difficult = keep_difficult
        _base_dir = DATASET_YEAR_DICT[year][3]
        _voc_root = os.path.join(self.root, _base_dir)
        _image_dir = os.path.join(_voc_root, 'JPEGImages')
        _annotation_dir = os.path.join(_voc_root, 'Annotations')

        if download:
            download_extract(self.url, self.root, self.filename, self.md5)

        if not os.path.isdir(_voc_root):
            raise RuntimeError('Dataset not found or corrupted.' +
                               ' You can use download=True to download it')

        _splits_dir = os.path.join(_voc_root, 'ImageSets/Main')

        _split_f = os.path.join(_splits_dir, image_set.rstrip('\n') + '.txt')

        if not os.path.exists(_split_f):
            raise ValueError(
                'Wrong image_set entered! Please use image_set="train" '
                'or image_set="trainval" or image_set="val" or a valid'
                'image_set from the VOC ImageSets/Main folder.')

        self.images = []
        self.annotations = []
        with open(os.path.join(_split_f), "r") as lines:
            for line in lines:
                _image = os.path.join(_image_dir, line.rstrip('\n') + ".jpg")
                _annotation = os.path.join(_annotation_dir,
                                           line.rstrip('\n') + ".xml")
                assert os.path.isfile(_image)
                assert os.path.isfile(_annotation)
                self.images.append(_image)
                self.annotations.append(_annotation)

        assert (len(self.images) == len(self.annotations))

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is a list of bounding boxes of
                relative coordinates like``[[xmin, ymin, xmax, ymax, ind], [...], ...]``.
        """
        _img = Image.open(self.images[index]).convert('RGB')
        _target = self._get_bboxes(ET.parse(self.annotations[index]).getroot())

        if self.transform is not None:
            _img = self.transform(_img)

        if self.target_transform is not None:
            _target = self.target_transform(_target)

        return _img, _target

    def __len__(self):
        return len(self.images)

    def _get_bboxes(self, target):
        res = []
        for obj in target.iter('object'):
            difficult = int(obj.find('difficult').text) == 1
            if not self.keep_difficult and difficult:
                continue
            name = obj.find('name').text.lower().strip()
            bbox = obj.find('bndbox')
            width = int(target.find('size').find('width').text)
            height = int(target.find('size').find('height').text)
            bndbox = []
            for i, cur_bb in enumerate(bbox):
                bb_sz = int(cur_bb.text) - 1
                # relative coordinates
                bb_sz = bb_sz / width if i % 2 == 0 else bb_sz / height
                bndbox.append(bb_sz)

            label_ind = self.class_to_ind[name]
            bndbox.append(label_ind)
            res.append(bndbox)  # [xmin, ymin, xmax, ymax, ind]
        return res


def download_extract(url, root, filename, md5):
    download_url(url, root, filename, md5)
    with tarfile.open(os.path.join(root, filename), "r") as tar:
        tar.extractall(path=root)
