Mask R-CNN
==========

.. currentmodule:: torchvision.models.detection

The Mask R-CNN model is based on the `Mask R-CNN <https://arxiv.org/abs/1703.06870>`__
paper.

.. betastatus:: detection module


Model builders
--------------

The following model builders can be used to instantiate a Mask R-CNN model, with or
without pre-trained weights. All the model builders internally rely on the
``torchvision.models.detection.mask_rcnn.MaskRCNN`` base class. Please refer to the `source
code
<https://github.com/pytorch/vision/blob/main/torchvision/models/detection/mask_rcnn.py>`_ for
more details about this class.

.. autosummary::
    :toctree: generated/
    :template: function.rst

    maskrcnn_resnet50_fpn
    maskrcnn_resnet50_fpn_v2
