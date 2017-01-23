"""The models subpackage contains definitions for the following model
architectures:

-  `AlexNet`_
-  `VGG`_
-  `ResNet`_
_  `InceptionV4`_

You can construct a model with random weights by calling its constructor:

.. code:: python

    import torchvision.models as models
    resnet18 = models.resnet18()
    alexnet = models.alexnet()
    inceptionv4 = models.inceptionv4()

We provide pre-trained models for the ResNet variants and AlexNet, using the
PyTorch :mod:`torch.utils.model_zoo`. These can  constructed by passing
``pretrained=True``:

.. code:: python

    import torchvision.models as models
    resnet18 = models.resnet18(pretrained=True)
    alexnet = models.alexnet(pretrained=True)
    inceptionv4 = models.inceptionv4(pretrained=True)

.. _AlexNet: https://arxiv.org/abs/1404.5997
.. _VGG: https://arxiv.org/abs/1409.1556
.. _ResNet: https://arxiv.org/abs/1512.03385
.. _InceptionV4: https://arxiv.org/abs/1602.07261
"""

from .alexnet import *
from .resnet import *
from .vgg import *
from .inceptionv4 import *
