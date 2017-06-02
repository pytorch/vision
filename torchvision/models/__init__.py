"""The models subpackage contains definitions for the following model
architectures:

-  `AlexNet`_
-  `VGG`_
-  `ResNet`_
-  `SqueezeNet`_
-  `DenseNet`_

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

All pre-trained models expect input images normalized in the same way,
i.e. mini-batches of 3-channel RGB images of shape (3 x H x W),
where H and W are expected to be atleast 224.
The images have to be loaded in to a range of [0, 1] and then normalized
using ``mean = [0.485, 0.456, 0.406]`` and ``std = [0.229, 0.224, 0.225]``.
You can use the following transform to normalize::

    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])

An example of such normalization can be found in the imagenet example
`here <https://github.com/pytorch/examples/blob/42e5b996718797e45c46a25c55b031e6768f8440/imagenet/main.py#L89-L101>`_

ImageNet 1-crop error rates (224x224)

================================  =============   =============
Network                           Top-1 error     Top-5 error
================================  =============   =============
ResNet-18                         30.24           10.92
ResNet-34                         26.70           8.58
ResNet-50                         23.85           7.13
ResNet-101                        22.63           6.44
ResNet-152                        21.69           5.94
Inception v3                      22.55           6.44
AlexNet                           43.45           20.91
VGG-11                            30.98           11.37
VGG-13                            30.07           10.75
VGG-16                            28.41           9.62
VGG-19                            27.62           9.12
VGG-11 with batch normalization   29.62           10.19
VGG-13 with batch normalization   28.45           9.63
VGG-16 with batch normalization   26.63           8.50
VGG-19 with batch normalization   25.76           8.15
SqueezeNet 1.0                    41.90           19.58
SqueezeNet 1.1                    41.81           19.38
Densenet-121                      25.35           7.83
Densenet-169                      24.00           7.00
Densenet-201                      22.80           6.43
Densenet-161                      22.35           6.20
================================  =============   =============


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
from .inception import *
from .densenet import *
