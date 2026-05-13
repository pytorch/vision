Video MViT
==========

.. currentmodule:: torchvision.models.video

The MViT model is based on the
`MViTv2: Improved Multiscale Vision Transformers for Classification and Detection
<https://arxiv.org/abs/2112.01526>`__ and `Multiscale Vision Transformers
<https://arxiv.org/abs/2104.11227>`__ papers.


Model builders
--------------

The following model builders can be used to instantiate a MViT v1 or v2 model, with or
without pre-trained weights. All the model builders internally rely on the
``torchvision.models.video.MViT`` base class. Please refer to the `source
code
<https://github.com/pytorch/vision/blob/main/torchvision/models/video/mvit.py>`_ for
more details about this class.

.. autosummary::
    :toctree: generated/
    :template: function.rst

    mvit_v1_b
    mvit_v2_s
