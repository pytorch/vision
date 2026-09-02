FCN
===

.. currentmodule:: torchvision.models.segmentation

The FCN model is based on the `Fully Convolutional Networks for Semantic
Segmentation <https://arxiv.org/abs/1411.4038>`__
paper.

.. betastatus:: segmentation module


Model builders
--------------

The following model builders can be used to instantiate a FCN model, with or
without pre-trained weights. All the model builders internally rely on the
``torchvision.models.segmentation.FCN`` base class. Please refer to the `source
code
<https://github.com/pytorch/vision/blob/main/torchvision/models/segmentation/fcn.py>`_ for
more details about this class.

.. autosummary::
    :toctree: generated/
    :template: function.rst

    fcn_resnet50
    fcn_resnet101
