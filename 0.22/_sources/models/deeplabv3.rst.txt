DeepLabV3
=========

.. currentmodule:: torchvision.models.segmentation

The DeepLabV3 model is based on the `Rethinking Atrous Convolution for Semantic
Image Segmentation <https://arxiv.org/abs/1706.05587>`__ paper.

.. betastatus:: segmentation module


Model builders
--------------

The following model builders can be used to instantiate a DeepLabV3 model with
different backbones, with or without pre-trained weights. All the model builders
internally rely on the ``torchvision.models.segmentation.deeplabv3.DeepLabV3`` base class. Please
refer to the `source code
<https://github.com/pytorch/vision/blob/main/torchvision/models/segmentation/deeplabv3.py>`_
for more details about this class.

.. autosummary::
    :toctree: generated/
    :template: function.rst

    deeplabv3_mobilenet_v3_large
    deeplabv3_resnet50
    deeplabv3_resnet101
