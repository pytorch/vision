EfficientNet
============

.. currentmodule:: torchvision.models

The EfficientNet model is based on the `EfficientNet: Rethinking Model Scaling for Convolutional Neural Networks <https://arxiv.org/abs/1905.11946>`__
paper.


Model builders
--------------

The following model builders can be used to instantiate an EfficientNet model, with or
without pre-trained weights. All the model builders internally rely on the
``torchvision.models.efficientnet.EfficientNet`` base class. Please refer to the `source
code
<https://github.com/pytorch/vision/blob/main/torchvision/models/efficientnet.py>`_ for
more details about this class.

.. autosummary::
    :toctree: generated/
    :template: function.rst

    efficientnet_b0
    efficientnet_b1
    efficientnet_b2
    efficientnet_b3
    efficientnet_b4
    efficientnet_b5
    efficientnet_b6
    efficientnet_b7
