from .lsun import LSUN, LSUNClass
from .folder import ImageClassFolder, ImageFolder
from .coco import CocoCaptions, CocoDetection
from .cifar import CIFAR10, CIFAR100
from .mnist import MNIST

__all__ = ('LSUN', 'LSUNClass', 'ImageFolder', 'ImageClassFolder',
           'CocoCaptions', 'CocoDetection', 'CIFAR10', 'CIFAR100', 'MNIST')
