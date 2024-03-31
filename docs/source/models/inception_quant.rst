Quantized InceptionV3
=====================

.. currentmodule:: torchvision.models.quantization

The Quantized Inception model is based on the `Rethinking the Inception Architecture for
Computer Vision <https://arxiv.org/abs/1512.00567>`__ paper.


Model builders
--------------

The following model builders can be used to instantiate a quantized Inception
model, with or without pre-trained weights. All the model builders internally
rely on the ``torchvision.models.quantization.inception.QuantizableInception3``
base class. Please refer to the `source code
<https://github.com/pytorch/vision/blob/main/torchvision/models/quantization/inception.py>`_
for more details about this class.

.. autosummary::
    :toctree: generated/
    :template: function.rst

    inception_v3
