.. _ops:

Operators
=========

.. currentmodule:: torchvision.ops

:mod:`torchvision.ops` implements operators that are specific for Computer Vision.

.. note::
  All operators have native support for TorchScript.


Detection Operators
~~~~~~~~~~~~~~~~~~~

.. autosummary::
    :toctree: generated/
    :template: function.rst

    batched_nms
    nms
    roi_align
    roi_pool

.. autosummary::
    :toctree: generated/
    :template: class.rst

    FeaturePyramidNetwork
    MultiScaleRoIAlign
    RoIAlign
    RoIPool


Box Operators
~~~~~~~~~~~~~

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

.. autosummary::
    :toctree: generated/
    :template: function.rst

    complete_box_iou_loss
    distance_box_iou_loss
    generalized_box_iou_loss
    sigmoid_focal_loss


Layers
~~~~~~

.. autosummary::
    :toctree: generated/
    :template: class.rst

    Conv2dNormActivation
    Conv3dNormActivation
    DeformConv2d
    DropBlock2d
    DropBlock3d
    FrozenBatchNorm2d
    SqueezeExcitation
    StochasticDepth

.. autosummary::
    :toctree: generated/
    :template: function.rst

    deform_conv2d
    drop_block2d
    drop_block3d
    stochastic_depth


Segmentation Operators
~~~~~~~~~~~~~~~~~~~~~~

.. autosummary::
    :toctree: generated/
    :template: function.rst

    masks_to_boxes
    ps_roi_align
    ps_roi_pool

.. autosummary::
    :toctree: generated/
    :template: class.rst

    PSRoIAlign
    PSRoIPool
