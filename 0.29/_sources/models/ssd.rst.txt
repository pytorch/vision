SSD
===

.. currentmodule:: torchvision.models.detection

The SSD model is based on the `SSD: Single Shot MultiBox Detector
<https://arxiv.org/abs/1512.02325>`__ paper.

.. betastatus:: detection module


Model builders
--------------

The following model builders can be used to instantiate a SSD model, with or
without pre-trained weights. All the model builders internally rely on the
``torchvision.models.detection.SSD`` base class. Please refer to the `source
code
<https://github.com/pytorch/vision/blob/main/torchvision/models/detection/ssd.py>`_ for
more details about this class.

.. autosummary::
    :toctree: generated/
    :template: function.rst

    ssd300_vgg16
