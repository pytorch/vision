Quantized ResNet
================

.. currentmodule:: torchvision.models.quantization

The Quantized ResNet model is based on the `Deep Residual Learning for Image Recognition
<https://arxiv.org/abs/1512.03385>`_ paper.


Model builders
--------------

The following model builders can be used to instantiate a quantized ResNet
model, with or without pre-trained weights. All the model builders internally
rely on the ``torchvision.models.quantization.resnet.QuantizableResNet``
base class. Please refer to the `source code
<https://github.com/pytorch/vision/blob/main/torchvision/models/quantization/resnet.py>`_
for more details about this class.

.. autosummary::
    :toctree: generated/
    :template: function.rst

    resnet18
    resnet50
