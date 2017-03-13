"""The models subpackage contains definitions for the following model
architectures:

-  `AlexNet`_
-  `VGG`_
-  `ResNet`_
-  `SqueezeNet`_

You can construct a model with random weights by calling its constructor:

.. code:: python

    import torchvision.models as models
    resnet18 = models.resnet18()
    alexnet = models.alexnet()
    squeezenet = models.squeezenet1_0()

We provide pre-trained models for the ResNet variants and AlexNet, using the
PyTorch :mod:`torch.utils.model_zoo`. These can  constructed by passing
``pretrained=True``:

.. code:: python

    import torchvision.models as models
    resnet18 = models.resnet18(pretrained=True)
    alexnet = models.alexnet(pretrained=True)

ImageNet 1-crop error rates (224x224)

======================== =============   =============
Network                  Top-1 error     Top-5 error
======================== =============   =============
ResNet-18                30.24           10.92
ResNet-34                26.70           8.58
ResNet-50                23.85           7.13
ResNet-101               22.63           6.44
ResNet-152               21.69           5.94
Inception v3             22.55           6.44
AlexNet                  43.45           20.91
VGG-11                   32.14           12.12
VGG-13                   31.04           11.40
VGG-16                   29.11           10.17
VGG-19                   28.42           9.69
SqueezeNet 1.0           41.90           19.58
SqueezeNet 1.1           41.81           19.38
======================== =============   =============


.. _AlexNet: https://arxiv.org/abs/1404.5997
.. _VGG: https://arxiv.org/abs/1409.1556
.. _ResNet: https://arxiv.org/abs/1512.03385
.. _SqueezeNet: https://arxiv.org/abs/1602.07360
"""

from .alexnet import *
from .resnet import *
from .vgg import *
from .squeezenet import *
from .inception import *
