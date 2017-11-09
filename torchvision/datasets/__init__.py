from .lsun import LSUN, LSUNClass
from .folder import ImageFolder
from .coco import CocoCaptions, CocoDetection
from .cifar import CIFAR10, CIFAR100
from .stl10 import STL10
from .mnist import MNIST, FashionMNIST
from .svhn import SVHN
from .phototour import PhotoTour
from .fakedata import FakeData
from .semeion import SEMEION

__all__ = ('LSUN', 'LSUNClass',
           'ImageFolder', 'FakeData',
           'CocoCaptions', 'CocoDetection',
           'CIFAR10', 'CIFAR100', 'FashionMNIST',
           'MNIST', 'STL10', 'SVHN', 'PhotoTour', 'SEMEION')
