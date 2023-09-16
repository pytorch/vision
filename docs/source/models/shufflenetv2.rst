ShuffleNet V2
=============

.. currentmodule:: torchvision.models

The ShuffleNet V2 model is based on the `ShuffleNet V2: Practical Guidelines for Efficient
CNN Architecture Design <https://arxiv.org/abs/1807.11164>`__ paper.


Model builders
--------------

The following model builders can be used to instantiate a ShuffleNetV2 model, with or
without pre-trained weights. All the model builders internally rely on the
``torchvision.models.shufflenetv2.ShuffleNetV2`` base class. Please refer to the `source
code
<https://github.com/pytorch/vision/blob/main/torchvision/models/shufflenetv2.py>`_ for
more details about this class.

.. autosummary::
    :toctree: generated/
    :template: function.rst

    shufflenet_v2_x0_5
    shufflenet_v2_x1_0
    shufflenet_v2_x1_5
    shufflenet_v2_x2_0
