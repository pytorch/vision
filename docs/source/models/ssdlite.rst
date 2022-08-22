SSDlite
=======

.. currentmodule:: torchvision.models.detection

The SSDLite model is based on the `SSD: Single Shot MultiBox Detector
<https://arxiv.org/abs/1512.02325>`__, `Searching for MobileNetV3
<https://arxiv.org/abs/1905.02244>`__ and `MobileNetV2: Inverted Residuals and Linear
Bottlenecks <https://arxiv.org/abs/1801.04381>__` papers.

.. betastatus:: detection module

Model builders
--------------

The following model builders can be used to instantiate a SSD Lite model, with or
without pre-trained weights. All the model builders internally rely on the
``torchvision.models.detection.ssd.SSD`` base class. Please refer to the `source
code
<https://github.com/pytorch/vision/blob/main/torchvision/models/detection/ssd.py>`_ for
more details about this class.

.. autosummary::
    :toctree: generated/
    :template: function.rst

    ssdlite320_mobilenet_v3_large
