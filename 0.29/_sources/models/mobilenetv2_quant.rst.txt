Quantized MobileNet V2
======================

.. currentmodule:: torchvision.models.quantization

The Quantized MobileNet V2 model is based on the `MobileNetV2: Inverted Residuals and Linear
Bottlenecks <https://arxiv.org/abs/1801.04381>`__ paper.


Model builders
--------------

The following model builders can be used to instantiate a quantized MobileNetV2
model, with or without pre-trained weights. All the model builders internally
rely on the ``torchvision.models.quantization.mobilenetv2.QuantizableMobileNetV2``
base class. Please refer to the `source code
<https://github.com/pytorch/vision/blob/main/torchvision/models/quantization/mobilenetv2.py>`_
for more details about this class.

.. autosummary::
    :toctree: generated/
    :template: function.rst

    mobilenet_v2
