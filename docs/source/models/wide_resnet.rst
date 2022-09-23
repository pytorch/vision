Wide ResNet
===========

.. currentmodule:: torchvision.models

The Wide ResNet model is based on the `Wide Residual Networks <https://arxiv.org/abs/1605.07146>`__
paper.


Model builders
--------------

The following model builders can be used to instantiate a Wide ResNet model, with or
without pre-trained weights. All the model builders internally rely on the
``torchvision.models.resnet.ResNet`` base class. Please refer to the `source
code
<https://github.com/pytorch/vision/blob/main/torchvision/models/resnet.py>`_ for
more details about this class.

.. autosummary::
    :toctree: generated/
    :template: function.rst

    wide_resnet50_2
    wide_resnet101_2
