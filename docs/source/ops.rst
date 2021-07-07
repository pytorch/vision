torchvision.ops
===============

.. currentmodule:: torchvision.ops

:mod:`torchvision.ops` implements operators that are specific for Computer Vision.

.. note::
  All operators have native support for TorchScript.


.. autofunction:: nms
.. autofunction:: batched_nms
.. autofunction:: remove_small_boxes
.. autofunction:: clip_boxes_to_image
.. autofunction:: box_convert
.. autofunction:: box_area
.. autofunction:: box_iou
.. autofunction:: generalized_box_iou
.. autofunction:: roi_align
.. autofunction:: ps_roi_align
.. autofunction:: roi_pool
.. autofunction:: ps_roi_pool
.. autofunction:: deform_conv2d
.. autofunction:: sigmoid_focal_loss

.. autoclass:: RoIAlign
.. autoclass:: PSRoIAlign
.. autoclass:: RoIPool
.. autoclass:: PSRoIPool
.. autoclass:: DeformConv2d
.. autoclass:: MultiScaleRoIAlign
.. autoclass:: FeaturePyramidNetwork
