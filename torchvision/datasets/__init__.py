from .lsun import LSUN, LSUNClass
from .folder import ImageFolder, DatasetFolder
from .coco import CocoCaptions, CocoDetection
from .cifar import CIFAR10, CIFAR100
from .stl10 import STL10
from .mnist import MNIST, EMNIST, FashionMNIST
from .svhn import SVHN
from .phototour import PhotoTour
from .fakedata import FakeData
from .semeion import SEMEION
from .omniglot import Omniglot

__all__ = ('LSUN', 'LSUNClass',
           'ImageFolder', 'DatasetFolder', 'FakeData',
           'CocoCaptions', 'CocoDetection',
           'CIFAR10', 'CIFAR100', 'EMNIST', 'FashionMNIST',
           'MNIST', 'STL10', 'SVHN', 'PhotoTour', 'SEMEION',
           'Omniglot')
