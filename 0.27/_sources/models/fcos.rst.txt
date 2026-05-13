FCOS
=========

.. currentmodule:: torchvision.models.detection

The FCOS model is based on the `FCOS: Fully Convolutional One-Stage Object Detection
<https://arxiv.org/abs/1904.01355>`__ paper.

.. betastatus:: detection module

Model builders
--------------

The following model builders can be used to instantiate a FCOS model, with or
without pre-trained weights. All the model builders internally rely on the
``torchvision.models.detection.fcos.FCOS`` base class. Please refer to the `source code
<https://github.com/pytorch/vision/blob/main/torchvision/models/detection/fcos.py>`_ for
more details about this class.

.. autosummary::
    :toctree: generated/
    :template: function.rst

    fcos_resnet50_fpn
