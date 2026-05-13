VGG
===

.. currentmodule:: torchvision.models

The VGG model is based on the `Very Deep Convolutional Networks for Large-Scale
Image Recognition <https://arxiv.org/abs/1409.1556>`_ paper.


Model builders
--------------

The following model builders can be used to instantiate a VGG model, with or
without pre-trained weights. All the model builders internally rely on the
``torchvision.models.vgg.VGG`` base class. Please refer to the `source code
<https://github.com/pytorch/vision/blob/main/torchvision/models/vgg.py>`_ for
more details about this class.

.. autosummary::
    :toctree: generated/
    :template: function.rst

    vgg11
    vgg11_bn
    vgg13
    vgg13_bn
    vgg16
    vgg16_bn
    vgg19
    vgg19_bn
