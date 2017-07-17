from .lsun import LSUN, LSUNClass
from .folder import ImageFolder
from .coco import CocoCaptions, CocoDetection
from .cifar import CIFAR10, CIFAR100
from .stl10 import STL10
from .mnist import MNIST
from .svhn import SVHN
from .phototour import PhotoTour
from .camvid import CamVid

__all__ = ('LSUN', 'LSUNClass',
           'ImageFolder',
           'CocoCaptions', 'CocoDetection',
           'CIFAR10', 'CIFAR100',
           'MNIST', 'STL10', 'SVHN', 'PhotoTour')
