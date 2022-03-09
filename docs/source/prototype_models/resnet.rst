ResNet
======

The ResNet model is based on the `Deep Residual Learning for Image Recognition
<https://arxiv.org/abs/1512.03385>`_ paper.


Model builders
--------------

The following model builders can be used to instanciate a ResNet model, with or
without pre-trained weights. All the model builders internally rely on the
``torchvision.models.resnet.ResNet`` base class. Please refer to the `source
code
<https://github.com/pytorch/vision/blob/main/torchvision/models/resnet.py>`_ for
more details about this class.

.. currentmodule:: torchvision.prototype.models

.. autosummary::
    :toctree: generated/
    :template: function.rst

    resnet18
    resnet34
    resnet50
    resnet101
    resnet152





Model builder weights
---------------------

It bothers me that we have to display these in this page. I wish there was a way
to not have this section in this page, and just have these weights sort of
inlined in their respective model buidlers page (instead of having a separate
page for them.)

.. autosummary::
    :toctree: generated/
    :template: class.rst

    ResNet18_Weights
    ResNet34_Weights
    ResNet50_Weights
    ResNet101_Weights
    ResNet152_Weights
