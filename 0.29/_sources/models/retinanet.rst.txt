RetinaNet
=========

.. currentmodule:: torchvision.models.detection

The RetinaNet model is based on the `Focal Loss for Dense Object Detection
<https://arxiv.org/abs/1708.02002>`__ paper.

.. betastatus:: detection module

Model builders
--------------

The following model builders can be used to instantiate a RetinaNet model, with or
without pre-trained weights. All the model builders internally rely on the
``torchvision.models.detection.retinanet.RetinaNet`` base class. Please refer to the `source code
<https://github.com/pytorch/vision/blob/main/torchvision/models/detection/retinanet.py>`_ for
more details about this class.

.. autosummary::
    :toctree: generated/
    :template: function.rst

    retinanet_resnet50_fpn
    retinanet_resnet50_fpn_v2
