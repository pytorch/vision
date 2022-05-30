Video ResNet
============

.. currentmodule:: torchvision.models.video

The MViTv2 model is based on the
`MViTv2: Improved Multiscale Vision Transformers for Classification and Detection
<https://arxiv.org/abs/2112.01526>`__ and `Multiscale Vision Transformers
<https://arxiv.org/abs/2104.11227>`__ papers.


Model builders
--------------

The following model builders can be used to instantiate a MViTV2 model, with or
without pre-trained weights. All the model builders internally rely on the
``torchvision.models.video.MViTV2`` base class. Please refer to the `source
code
<https://github.com/pytorch/vision/blob/main/torchvision/models/video/mvitv2.py>`_ for
more details about this class.

.. autosummary::
    :toctree: generated/
    :template: function.rst

    mvitv2_t
    mvitv2_s
    mvitv2_b
