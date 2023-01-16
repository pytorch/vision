EfficientNetV2
==============

.. currentmodule:: torchvision.models

The EfficientNetV2 model is based on the `EfficientNetV2: Smaller Models and Faster Training <https://arxiv.org/abs/2104.00298>`__
paper.


Model builders
--------------

The following model builders can be used to instantiate an EfficientNetV2 model, with or
without pre-trained weights. All the model builders internally rely on the
``torchvision.models.efficientnet.EfficientNet`` base class. Please refer to the `source
code
<https://github.com/pytorch/vision/blob/main/torchvision/models/efficientnet.py>`_ for
more details about this class.

.. autosummary::
    :toctree: generated/
    :template: function.rst

    efficientnet_v2_s
    efficientnet_v2_m
    efficientnet_v2_l
