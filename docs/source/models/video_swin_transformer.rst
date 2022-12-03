Video SwinTransformer
=====================

.. currentmodule:: torchvision.models.video

The Video SwinTransformer model is based on the `Video Swin Transformer <https://arxiv.org/abs/2106.13230>`__ paper.

.. betastatus:: video module


Model builders
--------------

The following model builders can be used to instantiate a VideoResNet model, with or
without pre-trained weights. All the model builders internally rely on the
``torchvision.models.video.swin_transformer.SwinTransformer3d`` base class. Please refer to the `source
code
<https://github.com/pytorch/vision/blob/main/torchvision/models/video/swin_transformer.py>`_ for
more details about this class.

.. autosummary::
    :toctree: generated/
    :template: function.rst

    swin3d_t
    swin3d_s
    swin3d_b
