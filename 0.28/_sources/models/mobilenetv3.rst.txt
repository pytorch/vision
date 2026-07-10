MobileNet V3
============

.. currentmodule:: torchvision.models

The MobileNet V3 model is based on the `Searching for MobileNetV3 <https://arxiv.org/abs/1905.02244>`__ paper.


Model builders
--------------

The following model builders can be used to instantiate a MobileNetV3 model, with or
without pre-trained weights. All the model builders internally rely on the
``torchvision.models.mobilenetv3.MobileNetV3`` base class. Please refer to the `source
code
<https://github.com/pytorch/vision/blob/main/torchvision/models/mobilenetv3.py>`_ for
more details about this class.

.. autosummary::
    :toctree: generated/
    :template: function.rst

    mobilenet_v3_large
    mobilenet_v3_small
