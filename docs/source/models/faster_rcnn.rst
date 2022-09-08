Faster R-CNN
============

.. currentmodule:: torchvision.models.detection


The Faster R-CNN model is based on the `Faster R-CNN: Towards Real-Time Object Detection 
with Region Proposal Networks <https://arxiv.org/abs/1506.01497>`__
paper.

.. betastatus:: detection module

Model builders
--------------

The following model builders can be used to instantiate a Faster R-CNN model, with or
without pre-trained weights. All the model builders internally rely on the
``torchvision.models.detection.faster_rcnn.FasterRCNN`` base class. Please refer to the `source
code
<https://github.com/pytorch/vision/blob/main/torchvision/models/detection/faster_rcnn.py>`_ for
more details about this class.

.. autosummary::
    :toctree: generated/
    :template: function.rst

    fasterrcnn_resnet50_fpn
    fasterrcnn_resnet50_fpn_v2
    fasterrcnn_mobilenet_v3_large_fpn
    fasterrcnn_mobilenet_v3_large_320_fpn
    
