from .caltech import Caltech101, Caltech256
from .celeba import CelebA
from .cifar import CIFAR10, CIFAR100
from .cityscapes import Cityscapes
from .coco import CocoCaptions, CocoDetection
from .fakedata import FakeData
from .flickr import Flickr8k, Flickr30k
from .folder import DatasetFolder, ImageFolder
from .hmdb51 import HMDB51
from .imagenet import ImageNet
from .inaturalist import INaturalist
from .kinetics import Kinetics, Kinetics400
from .kitti import Kitti
from .lsun import LSUN, LSUNClass
from .mnist import EMNIST, KMNIST, MNIST, QMNIST, FashionMNIST
from .omniglot import Omniglot
from .phototour import PhotoTour
from .places365 import Places365
from .sbd import SBDataset
from .sbu import SBU
from .semeion import SEMEION
from .stl10 import STL10
from .svhn import SVHN
from .ucf101 import UCF101
from .usps import USPS
from .vision import VisionDataset
from .voc import VOCDetection, VOCSegmentation
from .widerface import WIDERFace

__all__ = (
    "LSUN",
    "LSUNClass",
    "ImageFolder",
    "DatasetFolder",
    "FakeData",
    "CocoCaptions",
    "CocoDetection",
    "CIFAR10",
    "CIFAR100",
    "EMNIST",
    "FashionMNIST",
    "QMNIST",
    "MNIST",
    "KMNIST",
    "STL10",
    "SVHN",
    "PhotoTour",
    "SEMEION",
    "Omniglot",
    "SBU",
    "Flickr8k",
    "Flickr30k",
    "VOCSegmentation",
    "VOCDetection",
    "Cityscapes",
    "ImageNet",
    "Caltech101",
    "Caltech256",
    "CelebA",
    "WIDERFace",
    "SBDataset",
    "VisionDataset",
    "USPS",
    "Kinetics400",
    "Kinetics",
    "HMDB51",
    "UCF101",
    "Places365",
    "Kitti",
    "INaturalist",
)
