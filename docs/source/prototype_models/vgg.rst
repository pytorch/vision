VGG
===

The VGG model is based on the `Very Deep Convolutional Networks for Large-Scale
Image Recognition <https://arxiv.org/abs/1409.1556>`_ paper.


Model builders
--------------

The following model builders can be used to instanciate a VGG model, with or
without pre-trained weights. All the model buidlers internally rely on the
``torchvision.models.vgg.VGG`` base class. Please refer to the `source code
<https://github.com/pytorch/vision/blob/main/torchvision/models/vgg.py>`_ for
more details about this class.

.. currentmodule:: torchvision.prototype.models

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



Model builder weights
---------------------

It bothers me that we have to display these in this page. I wish there was a way
to not have this section in this page, and just have these weights sort of
inlined in their respective model buidlers page (instead of having a separate
page for them.)

.. autosummary::
    :toctree: generated/
    :template: class.rst

    VGG11_Weights
    VGG11_BN_Weights
    VGG13_Weights
    VGG13_BN_Weights
    VGG16_Weights
    VGG16_BN_Weights
    VGG19_Weights
    VGG19_BN_Weights