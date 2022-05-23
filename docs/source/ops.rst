.. _ops:

Operators
=========

.. currentmodule:: torchvision.ops

:mod:`torchvision.ops` implements operators, losses and layers that are specific for Computer Vision.

.. note::
  All operators have native support for TorchScript.


Detection and Segmentation Operators
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The below operators perform pre-processing as well as post-processing required in object detection and segmentation models.

.. autosummary::
    :toctree: generated/
    :template: function.rst

    batched_nms
    masks_to_boxes
    nms
    roi_align
    roi_pool
    ps_roi_align
    ps_roi_pool

.. autosummary::
    :toctree: generated/
    :template: class.rst

    FeaturePyramidNetwork
    MultiScaleRoIAlign
    RoIAlign
    RoIPool
    PSRoIAlign
    PSRoIPool


Box Operators
~~~~~~~~~~~~~

These utility functions perform various operations on bounding boxes.

.. autosummary::
    :toctree: generated/
    :template: function.rst

    box_area
    box_convert
    box_iou
    clip_boxes_to_image
    complete_box_iou
    distance_box_iou
    generalized_box_iou
    remove_small_boxes

Losses
~~~~~~

The following vision-specific loss functions are implemented:

.. autosummary::
    :toctree: generated/
    :template: function.rst

    complete_box_iou_loss
    distance_box_iou_loss
    generalized_box_iou_loss
    sigmoid_focal_loss


Layers
~~~~~~

TorchVision provides commonly used building blocks as layers:

.. autosummary::
    :toctree: generated/
    :template: class.rst

    Conv2dNormActivation
    Conv3dNormActivation
    DeformConv2d
    DropBlock2d
    DropBlock3d
    FrozenBatchNorm2d
    MLP
    Permute
    SqueezeExcitation
    StochasticDepth

.. autosummary::
    :toctree: generated/
    :template: function.rst

    deform_conv2d
    drop_block2d
    drop_block3d
    stochastic_depth
