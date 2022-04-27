Faster R-CNN
==========

.. currentmodule:: torchvision.models.detection

The Mask R-CNN model is based on the `Mask R-CNN <https://arxiv.org/abs/1703.06870>`__
paper.


Model builders
--------------

The following model builders can be used to instantiate a Mask R-CNN model, with or
without pre-trained weights. All the model builders internally rely on the
``torchvision.models.detection.faster_rcnn.FasterRCNN`` base class. Please refer to the `source
code
<https://github.com/pytorch/vision/blob/main/torchvision/models/detection/faster_rcnn.py>`_ for
more details about this class.

.. autosummary::
    :toctree: generated/
    :template: function.rst

    fasterrcnn_resnet50_fpn
    fasterrcnn_mobilenet_v3_large_fpn
    fasterrcnn_mobilenet_v3_large_320_fpn
    
