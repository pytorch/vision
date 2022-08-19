Video S3D
=========

.. currentmodule:: torchvision.models.video

The S3D model is based on the
`Rethinking Spatiotemporal Feature Learning: Speed-Accuracy Trade-offs in Video Classification
<https://arxiv.org/abs/1712.04851>`__ paper.


Model builders
--------------

The following model builders can be used to instantiate an S3D model, with or
without pre-trained weights. All the model builders internally rely on the
``torchvision.models.video.S3D`` base class. Please refer to the `source
code
<https://github.com/pytorch/vision/blob/main/torchvision/models/video/s3d.py>`_ for
more details about this class.

.. autosummary::
    :toctree: generated/
    :template: function.rst

    s3d
