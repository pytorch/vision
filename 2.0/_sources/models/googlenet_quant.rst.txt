Quantized GoogLeNet
===================

.. currentmodule:: torchvision.models.quantization

The Quantized GoogleNet model is based on the `Going Deeper with Convolutions <https://arxiv.org/abs/1409.4842>`__
paper.


Model builders
--------------

The following model builders can be used to instantiate a quantized GoogLeNet
model, with or without pre-trained weights. All the model builders internally
rely on the ``torchvision.models.quantization.googlenet.QuantizableGoogLeNet``
base class. Please refer to the `source code
<https://github.com/pytorch/vision/blob/main/torchvision/models/quantization/googlenet.py>`_
for more details about this class.

.. autosummary::
    :toctree: generated/
    :template: function.rst

    googlenet
