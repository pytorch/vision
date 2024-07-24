ResNeXt
=======

.. currentmodule:: torchvision.models

The ResNext model is based on the `Aggregated Residual Transformations for Deep Neural Networks <https://arxiv.org/abs/1611.05431v2>`__
paper.


Model builders
--------------

The following model builders can be used to instantiate a ResNext model, with or
without pre-trained weights. All the model builders internally rely on the
``torchvision.models.resnet.ResNet`` base class. Please refer to the `source
code
<https://github.com/pytorch/vision/blob/main/torchvision/models/resnet.py>`_ for
more details about this class.

.. autosummary::
    :toctree: generated/
    :template: function.rst

    resnext50_32x4d
    resnext101_32x8d
    resnext101_64x4d
