from .lsun import LSUN, LSUNClass  # noqa
from .folder import ImageFolder, DatasetFolder  # noqa
from .coco import CocoCaptions, CocoDetection  # noqa
from .cifar import CIFAR10, CIFAR100  # noqa
from .stl10 import STL10  # noqa
from .mnist import MNIST, EMNIST, FashionMNIST  # noqa
from .svhn import SVHN  # noqa
from .phototour import PhotoTour  # noqa
from .fakedata import FakeData  # noqa
from .semeion import SEMEION  # noqa
from .omniglot import Omniglot  # noqa

__all__ = ('LSUN', 'LSUNClass',
           'ImageFolder', 'DatasetFolder', 'FakeData',
           'CocoCaptions', 'CocoDetection',
           'CIFAR10', 'CIFAR100', 'EMNIST', 'FashionMNIST',
           'MNIST', 'STL10', 'SVHN', 'PhotoTour', 'SEMEION',
           'Omniglot')
