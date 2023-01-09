Video ResNet
============

.. currentmodule:: torchvision.models.video

The VideoResNet model is based on the `A Closer Look at Spatiotemporal
Convolutions for Action Recognition <https://arxiv.org/abs/1711.11248>`__ paper.

.. betastatus:: video module


Model builders
--------------

The following model builders can be used to instantiate a VideoResNet model, with or
without pre-trained weights. All the model builders internally rely on the
``torchvision.models.video.resnet.VideoResNet`` base class. Please refer to the `source
code
<https://github.com/pytorch/vision/blob/main/torchvision/models/video/resnet.py>`_ for
more details about this class.

.. autosummary::
    :toctree: generated/
    :template: function.rst

    r3d_18
    mc3_18
    r2plus1d_18
