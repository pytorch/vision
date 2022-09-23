LRASPP
======

.. currentmodule:: torchvision.models.segmentation

The LRASPP model is based on the `Searching for MobileNetV3 <https://arxiv.org/abs/1905.02244>`_ paper.

.. betastatus:: segmentation module

Model builders
--------------

The following model builders can be used to instantiate a FCN model, with or
without pre-trained weights. All the model builders internally rely on the
``torchvision.models.segmentation.LRASPP`` base class. Please refer to the `source
code
<https://github.com/pytorch/vision/blob/main/torchvision/models/segmentation/lraspp.py>`_ for
more details about this class.

.. autosummary::
    :toctree: generated/
    :template: function.rst

    lraspp_mobilenet_v3_large
