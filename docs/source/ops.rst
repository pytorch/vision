.. _ops:

Operators
=========

.. currentmodule:: torchvision.ops

:mod:`torchvision.ops` implements operators that are specific for Computer Vision.

.. note::
  All operators have native support for TorchScript.

.. autosummary::
    :toctree: generated/
    :template: function.rst

    batched_nms
    box_area
    box_convert
    box_iou
    clip_boxes_to_image
    deform_conv2d
    drop_block2d
    drop_block3d
    generalized_box_iou
    generalized_box_iou_loss
    masks_to_boxes
    nms
    ps_roi_align
    ps_roi_pool
    remove_small_boxes
    roi_align
    roi_pool
    sigmoid_focal_loss
    stochastic_depth

.. autosummary::
    :toctree: generated/
    :template: class.rst

    RoIAlign
    PSRoIAlign
    RoIPool
    PSRoIPool
    DeformConv2d
    MultiScaleRoIAlign
    FeaturePyramidNetwork
    StochasticDepth
    FrozenBatchNorm2d
    Conv2dNormActivation
    Conv3dNormActivation
    SqueezeExcitation
    DropBlock2d
    DropBlock3d
