torchvision.ops
===============

.. currentmodule:: torchvision.ops

:mod:`torchvision.ops` implements operators that are specific for Computer Vision.

.. note::
  All operators have native support for TorchScript.


.. autofunction:: batched_nms
.. autofunction:: box_area
.. autofunction:: box_convert
.. autofunction:: box_iou
.. autofunction:: clip_boxes_to_image
.. autofunction:: deform_conv2d
.. autofunction:: generalized_box_iou
.. autofunction:: masks_to_boxes
.. autofunction:: nms
.. autofunction:: ps_roi_align
.. autofunction:: ps_roi_pool
.. autofunction:: remove_small_boxes
.. autofunction:: roi_align
.. autofunction:: roi_pool
.. autofunction:: sigmoid_focal_loss
.. autofunction:: stochastic_depth

.. autoclass:: RoIAlign
.. autoclass:: PSRoIAlign
.. autoclass:: RoIPool
.. autoclass:: PSRoIPool
.. autoclass:: DeformConv2d
.. autoclass:: MultiScaleRoIAlign
.. autoclass:: FeaturePyramidNetwork
.. autoclass:: StochasticDepth
