Inception V3
============

.. currentmodule:: torchvision.models

The InceptionV3 model is based on the `Rethinking the Inception Architecture for
Computer Vision <https://arxiv.org/abs/1512.00567>`__ paper.


Model builders
--------------

The following model builders can be used to instantiate an InceptionV3 model, with or
without pre-trained weights. All the model builders internally rely on the
``torchvision.models.inception.Inception3`` base class. Please refer to the `source
code <https://github.com/pytorch/vision/blob/main/torchvision/models/inception.py>`_ for
more details about this class.

.. autosummary::
    :toctree: generated/
    :template: function.rst

    inception_v3
