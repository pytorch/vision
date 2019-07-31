from .lsun import LSUN, LSUNClass
from .folder import ImageFolder, DatasetFolder
from .coco import CocoCaptions, CocoDetection
from .cifar import CIFAR10, CIFAR100
from .stl10 import STL10
from .mnist import MNIST, EMNIST, FashionMNIST, KMNIST, QMNIST
from .svhn import SVHN
from .phototour import PhotoTour
from .fakedata import FakeData
from .semeion import SEMEION
from .omniglot import Omniglot
from .sbu import SBU
from .flickr import Flickr8k, Flickr30k
from .voc import VOCSegmentation, VOCDetection
from .cityscapes import Cityscapes
from .imagenet import ImageNet
from .caltech import Caltech101, Caltech256
from .celeba import CelebA
from .sbd import SBDataset
from .vision import VisionDataset
from .usps import USPS
from .kinetics import Kinetics400
from .hmdb51 import HMDB51
from .ucf101 import UCF101

__all__ = ('LSUN', 'LSUNClass',
           'ImageFolder', 'DatasetFolder', 'FakeData',
           'CocoCaptions', 'CocoDetection',
           'CIFAR10', 'CIFAR100', 'EMNIST', 'FashionMNIST', 'QMNIST',
           'MNIST', 'KMNIST', 'STL10', 'SVHN', 'PhotoTour', 'SEMEION',
           'Omniglot', 'SBU', 'Flickr8k', 'Flickr30k',
           'VOCSegmentation', 'VOCDetection', 'Cityscapes', 'ImageNet',
           'Caltech101', 'Caltech256', 'CelebA', 'SBDataset', 'VisionDataset',
           'USPS', 'Kinetics400', 'HMDB51', 'UCF101')
