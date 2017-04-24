"""The models subpackage contains definitions for the following model
architectures:

-  `AlexNet`_
-  `VGG`_
-  `ResNet`_
-  `DenseNet`_
-  `SqueezeNet`_
-  `MobileNet`_

You can construct a model with random weights by calling its constructor:

.. code:: python

    import torchvision.models as models
    resnet18 = models.resnet18()
    alexnet = models.alexnet()
    squeezenet = models.squeezenet1_0()
    densenet = models.densenet_161()

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
VGG-11                   30.98           11.37
VGG-13                   30.07           10.75
VGG-16                   28.41           9.62
VGG-19                   27.62           9.12
SqueezeNet 1.0           41.90           19.58
SqueezeNet 1.1           41.81           19.38
Densenet-121             25.35           7.83
Densenet-169             24.00           7.00
Densenet-201             22.80           6.43
Densenet-161             22.35           6.20
======================== =============   =============


.. _AlexNet: https://arxiv.org/abs/1404.5997
.. _VGG: https://arxiv.org/abs/1409.1556
.. _ResNet: https://arxiv.org/abs/1512.03385
.. _SqueezeNet: https://arxiv.org/abs/1602.07360
.. _DenseNet: https://arxiv.org/abs/1608.06993
"""

from .alexnet import *
from .resnet import *
from .vgg import *
from .squeezenet import *
from .mobilenet import *
from .inception import *
from .densenet import *
