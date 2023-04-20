import io
import re
from collections import OrderedDict
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Tuple, Union
from warnings import warn

import numpy as np
import torch
import torch.nn as nn
from torch import Tensor

from ...ops import box_convert
from ..yolo import (
    Conv,
    CSPSPP,
    CSPStage,
    ELANStage,
    FastSPP,
    MaxPool,
    RouteLayer,
    ShortcutLayer,
    YOLOV4Backbone,
    YOLOV4TinyBackbone,
    YOLOV5Backbone,
    YOLOV7Backbone,
)
from .anchor_utils import global_xy
from .target_matching import HighestIoUMatching, IoUThresholdMatching, PRIOR_SHAPES, SimOTAMatching, SizeRatioMatching
from .yolo_loss import YOLOLoss

DARKNET_CONFIG = Dict[str, Any]
CREATE_LAYER_OUTPUT = Tuple[nn.Module, int]  # layer, num_outputs
PRED = Dict[str, Tensor]
PREDS = List[PRED]  # TorchScript doesn't allow a tuple
TARGET = Dict[str, Tensor]
TARGETS = List[TARGET]  # TorchScript doesn't allow a tuple
NETWORK_OUTPUT = Tuple[List[Tensor], List[Tensor], List[int]]  # detections, losses, hits


class DetectionLayer(nn.Module):
    """A YOLO detection layer.

    A YOLO model has usually 1 - 3 detection layers at different resolutions. The loss is summed from all of them.

    Args:
        num_classes: Number of different classes that this layer predicts.
        prior_shapes: A list of prior box dimensions for this layer, used for scaling the predicted dimensions. The list
            should contain [width, height] pairs in the network input resolution.
        matching_func: The matching algorithm to be used for assigning targets to anchors.
        loss_func: ``YOLOLoss`` object for calculating the losses.
        xy_scale: Eliminate "grid sensitivity" by scaling the box coordinates by this factor. Using a value > 1.0 helps
            to produce coordinate values close to one.
        input_is_normalized: The input is normalized by logistic activation in the previous layer. In this case the
            detection layer will not take the sigmoid of the coordinate and probability predictions, and the width and
            height are scaled up so that the maximum value is four times the anchor dimension. This is used by the
            Darknet configurations of Scaled-YOLOv4.
    """

    def __init__(
        self,
        num_classes: int,
        prior_shapes: PRIOR_SHAPES,
        matching_func: Callable,
        loss_func: YOLOLoss,
        xy_scale: float = 1.0,
        input_is_normalized: bool = False,
    ) -> None:
        super().__init__()

        self.num_classes = num_classes
        self.prior_shapes = prior_shapes
        self.matching_func = matching_func
        self.loss_func = loss_func
        self.xy_scale = xy_scale
        self.input_is_normalized = input_is_normalized

    def forward(self, x: Tensor, image_size: Tensor) -> Tuple[Tensor, PREDS]:
        """Runs a forward pass through this YOLO detection layer.

        Maps cell-local coordinates to global coordinates in the image space, scales the bounding boxes with the
        anchors, converts the center coordinates to corner coordinates, and maps probabilities to the `]0, 1[` range
        using sigmoid.

        If targets are given, computes also losses from the predictions and the targets. This layer is responsible only
        for the targets that best match one of the anchors assigned to this layer. Training losses will be saved to the
        ``losses`` attribute. ``hits`` attribute will be set to the number of targets that this layer was responsible
        for. ``losses`` is a tensor of three elements: the overlap, confidence, and classification loss.

        Args:
            x: The output from the previous layer. The size of this tensor has to be
                ``[batch_size, anchors_per_cell * (num_classes + 5), height, width]``.
            image_size: Image width and height in a vector (defines the scale of the predicted and target coordinates).

        Returns:
            The layer output, with normalized probabilities, in a tensor sized
            ``[batch_size, anchors_per_cell * height * width, num_classes + 5]`` and a list of dictionaries, containing
            the same predictions, but with unnormalized probabilities (for loss calculation).
        """
        batch_size, num_features, height, width = x.shape
        num_attrs = self.num_classes + 5
        anchors_per_cell = num_features // num_attrs
        if anchors_per_cell != len(self.prior_shapes):
            raise ValueError(
                "The model predicts {} bounding boxes per spatial location, but {} prior box dimensions are defined "
                "for this layer.".format(anchors_per_cell, len(self.prior_shapes))
            )

        # Reshape the output to have the bounding box attributes of each grid cell on its own row.
        x = x.permute(0, 2, 3, 1)  # [batch_size, height, width, anchors_per_cell * num_attrs]
        x = x.view(batch_size, height, width, anchors_per_cell, num_attrs)

        # Take the sigmoid of the bounding box coordinates, confidence score, and class probabilities, unless the input
        # is normalized by the previous layer activation. Confidence and class losses use the unnormalized values if
        # possible.
        norm_x = x if self.input_is_normalized else torch.sigmoid(x)
        xy = norm_x[..., :2]
        wh = x[..., 2:4]
        confidence = x[..., 4]
        classprob = x[..., 5:]
        norm_confidence = norm_x[..., 4]
        norm_classprob = norm_x[..., 5:]

        # Eliminate grid sensitivity. The previous layer should output extremely high values for the sigmoid to produce
        # x/y coordinates close to one. YOLOv4 solves this by scaling the x/y coordinates.
        xy = xy * self.xy_scale - 0.5 * (self.xy_scale - 1)

        image_xy = global_xy(xy, image_size)
        prior_shapes = torch.tensor(self.prior_shapes, dtype=wh.dtype, device=wh.device)
        if self.input_is_normalized:
            image_wh = 4 * torch.square(wh) * prior_shapes
        else:
            image_wh = torch.exp(wh) * prior_shapes
        box = torch.cat((image_xy, image_wh), -1)
        box = box_convert(box, in_fmt="cxcywh", out_fmt="xyxy")
        output = torch.cat((box, norm_confidence.unsqueeze(-1), norm_classprob), -1)
        output = output.reshape(batch_size, height * width * anchors_per_cell, num_attrs)

        # It's better to use binary_cross_entropy_with_logits() for loss computation, so we'll provide the unnormalized
        # confidence and classprob, when available.
        preds = [{"boxes": b, "confidences": c, "classprobs": p} for b, c, p in zip(box, confidence, classprob)]

        return output, preds

    def match_targets(
        self,
        preds: PREDS,
        return_preds: PREDS,
        targets: TARGETS,
        image_size: Tensor,
    ) -> Tuple[PRED, TARGET]:
        """Matches the predictions to targets.

        Args:
            preds: List of predictions for each image, as returned by the ``forward()`` method of this layer. These will
                be matched to the training targets.
            return_preds: List of predictions for each image. The matched predictions will be returned from this list.
                When calculating the auxiliary loss for deep supervision, predictions from a different layer are used
                for loss computation.
            targets: List of training targets for each image.
            image_size: Width and height in a vector that defines the scale of the target coordinates.

        Returns:
            Two dictionaries, the matched predictions and targets.
        """
        batch_size = len(preds)
        if (len(targets) != batch_size) or (len(return_preds) != batch_size):
            raise ValueError("Different batch size for predictions and targets.")

        # Creating lists that are concatenated in the end will confuse TorchScript compilation. Instead, we'll create
        # tensors and concatenate new matches immediately.
        pred_boxes = torch.empty((0, 4), device=return_preds[0]["boxes"].device)
        pred_confidences = torch.empty(0, device=return_preds[0]["confidences"].device)
        pred_bg_confidences = torch.empty(0, device=return_preds[0]["confidences"].device)
        pred_classprobs = torch.empty((0, self.num_classes), device=return_preds[0]["classprobs"].device)
        target_boxes = torch.empty((0, 4), device=targets[0]["boxes"].device)
        target_labels = torch.empty(0, dtype=torch.int64, device=targets[0]["labels"].device)

        for image_preds, image_return_preds, image_targets in zip(preds, return_preds, targets):
            if image_targets["boxes"].shape[0] > 0:
                pred_selector, background_selector, target_selector = self.matching_func(
                    image_preds, image_targets, image_size
                )
                pred_boxes = torch.cat((pred_boxes, image_return_preds["boxes"][pred_selector]))
                pred_confidences = torch.cat((pred_confidences, image_return_preds["confidences"][pred_selector]))
                pred_bg_confidences = torch.cat(
                    (pred_bg_confidences, image_return_preds["confidences"][background_selector])
                )
                pred_classprobs = torch.cat((pred_classprobs, image_return_preds["classprobs"][pred_selector]))
                target_boxes = torch.cat((target_boxes, image_targets["boxes"][target_selector]))
                target_labels = torch.cat((target_labels, image_targets["labels"][target_selector]))
            else:
                pred_bg_confidences = torch.cat((pred_bg_confidences, image_return_preds["confidences"].flatten()))

        matched_preds = {
            "boxes": pred_boxes,
            "confidences": pred_confidences,
            "bg_confidences": pred_bg_confidences,
            "classprobs": pred_classprobs,
        }
        matched_targets = {
            "boxes": target_boxes,
            "labels": target_labels,
        }
        return matched_preds, matched_targets

    def calculate_losses(
        self,
        preds: PREDS,
        targets: TARGETS,
        image_size: Tensor,
        loss_preds: Optional[PREDS] = None,
    ) -> Tuple[Tensor, int]:
        """Matches the predictions to targets and computes the losses.

        Args:
            preds: List of predictions for each image, as returned by ``forward()``. These will be matched to the
                training targets and used to compute the losses (unless another set of predictions for loss computation
                is given in ``loss_preds``).
            targets: List of training targets for each image.
            image_size: Width and height in a vector that defines the scale of the target coordinates.
            loss_preds: List of predictions for each image. If given, these will be used for loss computation, instead
                of the same predictions that were used for matching. This is needed for deep supervision in YOLOv7.

        Returns:
            A vector of the overlap, confidence, and classification loss, normalized by batch size, and the number of
            targets that were matched to this layer.
        """
        if loss_preds is None:
            loss_preds = preds

        matched_preds, matched_targets = self.match_targets(preds, loss_preds, targets, image_size)

        losses = self.loss_func.elementwise_sums(matched_preds, matched_targets, self.input_is_normalized, image_size)
        losses = torch.stack((losses.overlap, losses.confidence, losses.classification)) / len(preds)

        hits = len(matched_targets["boxes"])

        return losses, hits


def create_detection_layer(
    prior_shapes: PRIOR_SHAPES,
    prior_shape_idxs: List[int],
    matching_algorithm: Optional[str] = None,
    matching_threshold: Optional[float] = None,
    spatial_range: float = 5.0,
    size_range: float = 4.0,
    ignore_bg_threshold: float = 0.7,
    overlap_func: str = "ciou",
    predict_overlap: Optional[float] = None,
    label_smoothing: Optional[float] = None,
    overlap_loss_multiplier: float = 5.0,
    confidence_loss_multiplier: float = 1.0,
    class_loss_multiplier: float = 1.0,
    **kwargs: Any,
) -> DetectionLayer:
    """Creates a detection layer module and the required loss function and target matching objects.

    Args:
        prior_shapes: A list of all the prior box dimensions, used for scaling the predicted dimensions and possibly for
            matching the targets to the anchors. The list should contain [width, height] pairs in the network input
            resolution.
        prior_shape_idxs: List of indices to ``prior_shapes`` that is used to select the (usually 3) prior shapes that
            this layer uses.
        matching_algorithm: Which algorithm to use for matching targets to anchors. "simota" (the SimOTA matching rule
            from YOLOX), "size" (match those prior shapes, whose width and height relative to the target is below given
            ratio), "iou" (match all prior shapes that give a high enough IoU), or "maxiou" (match the prior shape that
            gives the highest IoU, default).
        matching_threshold: Threshold for "size" and "iou" matching algorithms.
        spatial_range: The "simota" matching algorithm will restrict to anchors that are within an `N x N` grid cell
            area centered at the target, where `N` is the value of this parameter.
        size_range: The "simota" matching algorithm will restrict to anchors whose dimensions are no more than `N` and
            no less than `1/N` times the target dimensions, where `N` is the value of this parameter.
        ignore_bg_threshold: If a predictor is not responsible for predicting any target, but the corresponding anchor
            has IoU with some target greater than this threshold, the predictor will not be taken into account when
            calculating the confidence loss.
        overlap_func: Which function to use for calculating the IoU between two sets of boxes. Valid values are "iou",
            "giou", "diou", and "ciou".
        predict_overlap: Balance between binary confidence targets and predicting the overlap. 0.0 means that the target
            confidence is 1 if there's an object, and 1.0 means that the target confidence is the output of
            ``overlap_func``.
        label_smoothing: The epsilon parameter (weight) for class label smoothing. 0.0 means no smoothing (binary
            targets), and 1.0 means that the target probabilities are always 0.5.
        overlap_loss_multiplier: Overlap loss will be scaled by this value.
        confidence_loss_multiplier: Confidence loss will be scaled by this value.
        class_loss_multiplier: Classification loss will be scaled by this value.
        num_classes: Number of different classes that this layer predicts.
        xy_scale: Eliminate "grid sensitivity" by scaling the box coordinates by this factor. Using a value > 1.0 helps
            to produce coordinate values close to one.
        input_is_normalized: The input is normalized by logistic activation in the previous layer. In this case the
            detection layer will not take the sigmoid of the coordinate and probability predictions, and the width and
            height are scaled up so that the maximum value is four times the anchor dimension. This is used by the
            Darknet configurations of Scaled-YOLOv4.
    """
    matching_func: Callable
    if matching_algorithm == "simota":
        loss_func = YOLOLoss(
            overlap_func, None, None, overlap_loss_multiplier, confidence_loss_multiplier, class_loss_multiplier
        )
        matching_func = SimOTAMatching(prior_shapes, prior_shape_idxs, loss_func, spatial_range, size_range)
    elif matching_algorithm == "size":
        if matching_threshold is None:
            raise ValueError("matching_threshold is required with size ratio matching.")
        matching_func = SizeRatioMatching(prior_shapes, prior_shape_idxs, matching_threshold, ignore_bg_threshold)
    elif matching_algorithm == "iou":
        if matching_threshold is None:
            raise ValueError("matching_threshold is required with IoU threshold matching.")
        matching_func = IoUThresholdMatching(prior_shapes, prior_shape_idxs, matching_threshold, ignore_bg_threshold)
    elif matching_algorithm == "maxiou" or matching_algorithm is None:
        matching_func = HighestIoUMatching(prior_shapes, prior_shape_idxs, ignore_bg_threshold)
    else:
        raise ValueError(f"Matching algorithm `{matching_algorithm}´ is unknown.")

    loss_func = YOLOLoss(
        overlap_func,
        predict_overlap,
        label_smoothing,
        overlap_loss_multiplier,
        confidence_loss_multiplier,
        class_loss_multiplier,
    )
    layer_shapes = [prior_shapes[i] for i in prior_shape_idxs]
    return DetectionLayer(prior_shapes=layer_shapes, matching_func=matching_func, loss_func=loss_func, **kwargs)


class DetectionStage(nn.Module):
    """This is a convenience class for running a detection layer.

    It might be cleaner to implement this as a function, but TorchScript allows only specific types in function
    arguments, not modules.
    """

    def __init__(self, **kwargs: Any) -> None:
        super().__init__()
        self.detection_layer = create_detection_layer(**kwargs)

    def forward(
        self,
        layer_input: Tensor,
        targets: Optional[TARGETS],
        image_size: Tensor,
        detections: List[Tensor],
        losses: List[Tensor],
        hits: List[int],
    ) -> None:
        """Runs the detection layer on the inputs and appends the output to the ``detections`` list.

        If ``targets`` is given, also calculates the losses and appends to the ``losses`` list.

        Args:
            layer_input: Input to the detection layer.
            targets: List of training targets for each image.
            image_size: Width and height in a vector that defines the scale of the target coordinates.
            detections: A list where a tensor containing the detections will be appended to.
            losses: A list where a tensor containing the losses will be appended to, if ``targets`` is given.
            hits: A list where the number of targets that matched this layer will be appended to, if ``targets`` is
                given.
        """
        output, preds = self.detection_layer(layer_input, image_size)
        detections.append(output)

        if targets is not None:
            layer_losses, layer_hits = self.detection_layer.calculate_losses(preds, targets, image_size)
            losses.append(layer_losses)
            hits.append(layer_hits)


class DetectionStageWithAux(nn.Module):
    """This class represents a combination of a lead and an auxiliary detection layer.

    Args:
        spatial_range: The "simota" matching algorithm will restrict to anchors that are within an `N x N` grid cell
            area centered at the target. This parameter specifies `N` for the lead head.
        aux_spatial_range: The "simota" matching algorithm will restrict to anchors that are within an `N x N` grid cell
            area centered at the target. This parameter specifies `N` for the auxiliary head.
        aux_weight: Weight for the loss from the auxiliary head.
    """

    def __init__(
        self, spatial_range: float = 5.0, aux_spatial_range: float = 3.0, aux_weight: float = 0.25, **kwargs: Any
    ) -> None:
        super().__init__()
        self.detection_layer = create_detection_layer(spatial_range=spatial_range, **kwargs)
        self.aux_detection_layer = create_detection_layer(spatial_range=aux_spatial_range, **kwargs)
        self.aux_weight = aux_weight

    def forward(
        self,
        layer_input: Tensor,
        aux_input: Tensor,
        targets: Optional[TARGETS],
        image_size: Tensor,
        detections: List[Tensor],
        losses: List[Tensor],
        hits: List[int],
    ) -> None:
        """Runs the detection layer and the auxiliary detection layer on their respective inputs and appends the
        outputs to the ``detections`` list.

        If ``targets`` is given, also calculates the losses and appends to the ``losses`` list.

        Args:
            layer_input: Input to the lead detection layer.
            aux_input: Input to the auxiliary detection layer.
            targets: List of training targets for each image.
            image_size: Width and height in a vector that defines the scale of the target coordinates.
            detections: A list where a tensor containing the detections will be appended to.
            losses: A list where a tensor containing the losses will be appended to, if ``targets`` is given.
            hits: A list where the number of targets that matched this layer will be appended to, if ``targets`` is
                given.
        """
        output, preds = self.detection_layer(layer_input, image_size)
        detections.append(output)

        if targets is not None:
            # Match lead head predictions to targets and calculate losses from lead head outputs.
            layer_losses, layer_hits = self.detection_layer.calculate_losses(preds, targets, image_size)
            losses.append(layer_losses)
            hits.append(layer_hits)

            # Match lead head predictions to targets and calculate losses from auxiliary head outputs.
            _, aux_preds = self.aux_detection_layer(aux_input, image_size)
            layer_losses, layer_hits = self.aux_detection_layer.calculate_losses(
                preds, targets, image_size, loss_preds=aux_preds
            )
            losses.append(layer_losses * self.aux_weight)
            hits.append(layer_hits)


@torch.jit.script
def get_image_size(images: Tensor) -> Tensor:
    """Get the image size from an input tensor.

    The function needs the ``@torch.jit.script`` decorator in order for ONNX generation to work. The tracing based
    generator will loose track of e.g. ``images.shape[1]`` and treat it as a Python variable and not a tensor. This will
    cause the dimension to be treated as a constant in the model, which prevents dynamic input sizes.

    Args:
        images: An image batch to take the width and height from.

    Returns:
        A tensor that contains the image width and height.
    """
    height = images.shape[2]
    width = images.shape[3]
    return torch.tensor([width, height], device=images.device)


class YOLOV4TinyNetwork(nn.Module):
    """The "tiny" network architecture from YOLOv4.

    Args:
        num_classes: Number of different classes that this model predicts.
        backbone: A backbone network that returns the output from each stage.
        width: The number of channels in the narrowest convolutional layer. The wider convolutional layers will use a
            number of channels that is a multiple of this value.
        activation: Which layer activation to use. Can be "relu", "leaky", "mish", "silu" (or "swish"), "logistic",
            "linear", or "none".
        normalization: Which layer normalization to use. Can be "batchnorm", "groupnorm", or "none".
        prior_shapes: A list of prior box dimensions, used for scaling the predicted dimensions and possibly for
            matching the targets to the anchors. The list should contain [width, height] pairs in the network input
            resolution. There should be `3N` pairs, where `N` is the number of anchors per spatial location. They are
            assigned to the layers from the lowest (high-resolution) to the highest (low-resolution) layer, meaning that
            you typically want to sort the shapes from the smallest to the largest.
        matching_algorithm: Which algorithm to use for matching targets to anchors. "simota" (the SimOTA matching rule
            from YOLOX), "size" (match those prior shapes, whose width and height relative to the target is below given
            ratio), "iou" (match all prior shapes that give a high enough IoU), or "maxiou" (match the prior shape that
            gives the highest IoU, default).
        matching_threshold: Threshold for "size" and "iou" matching algorithms.
        spatial_range: The "simota" matching algorithm will restrict to anchors that are within an `N x N` grid cell
            area centered at the target, where `N` is the value of this parameter.
        size_range: The "simota" matching algorithm will restrict to anchors whose dimensions are no more than `N` and
            no less than `1/N` times the target dimensions, where `N` is the value of this parameter.
        ignore_bg_threshold: If a predictor is not responsible for predicting any target, but the prior shape has IoU
            with some target greater than this threshold, the predictor will not be taken into account when calculating
            the confidence loss.
        overlap_func: Which function to use for calculating the IoU between two sets of boxes. Valid values are "iou",
            "giou", "diou", and "ciou".
        predict_overlap: Balance between binary confidence targets and predicting the overlap. 0.0 means that the target
            confidence is 1 if there's an object, and 1.0 means that the target confidence is the output of
            ``overlap_func``.
        label_smoothing: The epsilon parameter (weight) for class label smoothing. 0.0 means no smoothing (binary
            targets), and 1.0 means that the target probabilities are always 0.5.
        overlap_loss_multiplier: Overlap loss will be scaled by this value.
        confidence_loss_multiplier: Confidence loss will be scaled by this value.
        class_loss_multiplier: Classification loss will be scaled by this value.
        xy_scale: Eliminate "grid sensitivity" by scaling the box coordinates by this factor. Using a value > 1.0 helps
            to produce coordinate values close to one.
    """

    def __init__(
        self,
        num_classes: int,
        backbone: Optional[nn.Module] = None,
        width: int = 32,
        activation: Optional[str] = "leaky",
        normalization: Optional[str] = "batchnorm",
        prior_shapes: Optional[PRIOR_SHAPES] = None,
        **kwargs: Any,
    ) -> None:
        super().__init__()

        # By default use the prior shapes that have been learned from the COCO data.
        if prior_shapes is None:
            prior_shapes = [
                [12, 16],
                [19, 36],
                [40, 28],
                [36, 75],
                [76, 55],
                [72, 146],
                [142, 110],
                [192, 243],
                [459, 401],
            ]
            anchors_per_cell = 3
        else:
            anchors_per_cell, modulo = divmod(len(prior_shapes), 3)
            if modulo != 0:
                raise ValueError("The number of provided prior shapes needs to be divisible by 3.")
        num_outputs = (5 + num_classes) * anchors_per_cell

        def conv(in_channels: int, out_channels: int, kernel_size: int = 1) -> nn.Module:
            return Conv(in_channels, out_channels, kernel_size, stride=1, activation=activation, norm=normalization)

        def upsample(in_channels: int, out_channels: int) -> nn.Module:
            channels = conv(in_channels, out_channels)
            upsample = nn.Upsample(scale_factor=2, mode="nearest")
            return nn.Sequential(OrderedDict([("channels", channels), ("upsample", upsample)]))

        def outputs(in_channels: int) -> nn.Module:
            return nn.Conv2d(in_channels, num_outputs, kernel_size=1, stride=1, bias=True)

        def detect(prior_shape_idxs: Sequence[int]) -> DetectionStage:
            assert prior_shapes is not None
            return DetectionStage(
                prior_shapes=prior_shapes,
                prior_shape_idxs=list(prior_shape_idxs),
                num_classes=num_classes,
                input_is_normalized=False,
                **kwargs,
            )

        self.backbone = backbone or YOLOV4TinyBackbone(width=width, activation=activation, normalization=normalization)

        self.fpn5 = conv(width * 16, width * 8)
        self.out5 = nn.Sequential(
            OrderedDict(
                [
                    ("channels", conv(width * 8, width * 16)),
                    (f"outputs_{num_outputs}", outputs(width * 16)),
                ]
            )
        )
        self.upsample5 = upsample(width * 8, width * 4)

        self.fpn4 = conv(width * 12, width * 8, kernel_size=3)
        self.out4 = nn.Sequential(OrderedDict([(f"outputs_{num_outputs}", outputs(width * 8))]))
        self.upsample4 = upsample(width * 8, width * 2)

        self.fpn3 = conv(width * 6, width * 4, kernel_size=3)
        self.out3 = nn.Sequential(OrderedDict([(f"outputs_{num_outputs}", outputs(width * 4))]))

        self.detect3 = detect([0, 1, 2])
        self.detect4 = detect([3, 4, 5])
        self.detect5 = detect([6, 7, 8])

    def forward(self, x: Tensor, targets: Optional[TARGETS] = None) -> NETWORK_OUTPUT:
        detections: List[Tensor] = []  # Outputs from detection layers
        losses: List[Tensor] = []  # Losses from detection layers
        hits: List[int] = []  # Number of targets each detection layer was responsible for

        image_size = get_image_size(x)

        c3, c4, c5 = self.backbone(x)[-3:]

        p5 = self.fpn5(c5)
        x = torch.cat((self.upsample5(p5), c4), dim=1)
        p4 = self.fpn4(x)
        x = torch.cat((self.upsample4(p4), c3), dim=1)
        p3 = self.fpn3(x)

        self.detect5(self.out5(p5), targets, image_size, detections, losses, hits)
        self.detect4(self.out4(p4), targets, image_size, detections, losses, hits)
        self.detect3(self.out3(p3), targets, image_size, detections, losses, hits)
        return detections, losses, hits


class YOLOV4Network(nn.Module):
    """Network architecture that corresponds approximately to the Cross Stage Partial Network from YOLOv4.

    Args:
        num_classes: Number of different classes that this model predicts.
        backbone: A backbone network that returns the output from each stage.
        widths: Number of channels at each network stage.
        activation: Which layer activation to use. Can be "relu", "leaky", "mish", "silu" (or "swish"), "logistic",
            "linear", or "none".
        normalization: Which layer normalization to use. Can be "batchnorm", "groupnorm", or "none".
        prior_shapes: A list of prior box dimensions, used for scaling the predicted dimensions and possibly for
            matching the targets to the anchors. The list should contain [width, height] pairs in the network input
            resolution. There should be `3N` pairs, where `N` is the number of anchors per spatial location. They are
            assigned to the layers from the lowest (high-resolution) to the highest (low-resolution) layer, meaning that
            you typically want to sort the shapes from the smallest to the largest.
        matching_algorithm: Which algorithm to use for matching targets to anchors. "simota" (the SimOTA matching rule
            from YOLOX), "size" (match those prior shapes, whose width and height relative to the target is below given
            ratio), "iou" (match all prior shapes that give a high enough IoU), or "maxiou" (match the prior shape that
            gives the highest IoU, default).
        matching_threshold: Threshold for "size" and "iou" matching algorithms.
        spatial_range: The "simota" matching algorithm will restrict to anchors that are within an `N x N` grid cell
            area centered at the target, where `N` is the value of this parameter.
        size_range: The "simota" matching algorithm will restrict to anchors whose dimensions are no more than `N` and
            no less than `1/N` times the target dimensions, where `N` is the value of this parameter.
        ignore_bg_threshold: If a predictor is not responsible for predicting any target, but the prior shape has IoU
            with some target greater than this threshold, the predictor will not be taken into account when calculating
            the confidence loss.
        overlap_func: Which function to use for calculating the IoU between two sets of boxes. Valid values are "iou",
            "giou", "diou", and "ciou".
        predict_overlap: Balance between binary confidence targets and predicting the overlap. 0.0 means that the target
            confidence is 1 if there's an object, and 1.0 means that the target confidence is the output of
            ``overlap_func``.
        label_smoothing: The epsilon parameter (weight) for class label smoothing. 0.0 means no smoothing (binary
            targets), and 1.0 means that the target probabilities are always 0.5.
        overlap_loss_multiplier: Overlap loss will be scaled by this value.
        confidence_loss_multiplier: Confidence loss will be scaled by this value.
        class_loss_multiplier: Classification loss will be scaled by this value.
        xy_scale: Eliminate "grid sensitivity" by scaling the box coordinates by this factor. Using a value > 1.0 helps
            to produce coordinate values close to one.
    """

    def __init__(
        self,
        num_classes: int,
        backbone: Optional[nn.Module] = None,
        widths: Sequence[int] = (32, 64, 128, 256, 512, 1024),
        activation: Optional[str] = "silu",
        normalization: Optional[str] = "batchnorm",
        prior_shapes: Optional[PRIOR_SHAPES] = None,
        **kwargs: Any,
    ) -> None:
        super().__init__()

        # By default use the prior shapes that have been learned from the COCO data.
        if prior_shapes is None:
            prior_shapes = [
                [12, 16],
                [19, 36],
                [40, 28],
                [36, 75],
                [76, 55],
                [72, 146],
                [142, 110],
                [192, 243],
                [459, 401],
            ]
            anchors_per_cell = 3
        else:
            anchors_per_cell, modulo = divmod(len(prior_shapes), 3)
            if modulo != 0:
                raise ValueError("The number of provided prior shapes needs to be divisible by 3.")
        num_outputs = (5 + num_classes) * anchors_per_cell

        def spp(in_channels: int, out_channels: int) -> nn.Module:
            return CSPSPP(in_channels, out_channels, activation=activation, norm=normalization)

        def conv(in_channels: int, out_channels: int) -> nn.Module:
            return Conv(in_channels, out_channels, kernel_size=1, stride=1, activation=activation, norm=normalization)

        def csp(in_channels: int, out_channels: int) -> nn.Module:
            return CSPStage(
                in_channels,
                out_channels,
                depth=2,
                shortcut=False,
                norm=normalization,
                activation=activation,
            )

        def out(in_channels: int) -> nn.Module:
            conv = Conv(in_channels, in_channels, kernel_size=3, stride=1, activation=activation, norm=normalization)
            outputs = nn.Conv2d(in_channels, num_outputs, kernel_size=1)
            return nn.Sequential(OrderedDict([("conv", conv), (f"outputs_{num_outputs}", outputs)]))

        def upsample(in_channels: int, out_channels: int) -> nn.Module:
            channels = conv(in_channels, out_channels)
            upsample = nn.Upsample(scale_factor=2, mode="nearest")
            return nn.Sequential(OrderedDict([("channels", channels), ("upsample", upsample)]))

        def downsample(in_channels: int, out_channels: int) -> nn.Module:
            return Conv(in_channels, out_channels, kernel_size=3, stride=2, activation=activation, norm=normalization)

        def detect(prior_shape_idxs: Sequence[int]) -> DetectionStage:
            assert prior_shapes is not None
            return DetectionStage(
                prior_shapes=prior_shapes,
                prior_shape_idxs=list(prior_shape_idxs),
                num_classes=num_classes,
                input_is_normalized=False,
                **kwargs,
            )

        if backbone is not None:
            self.backbone = backbone
        else:
            self.backbone = YOLOV4Backbone(widths=widths, activation=activation, normalization=normalization)

        w3 = widths[-3]
        w4 = widths[-2]
        w5 = widths[-1]

        self.spp = spp(w5, w5)

        self.pre4 = conv(w4, w4 // 2)
        self.upsample5 = upsample(w5, w4 // 2)
        self.fpn4 = csp(w4, w4)

        self.pre3 = conv(w3, w3 // 2)
        self.upsample4 = upsample(w4, w3 // 2)
        self.fpn3 = csp(w3, w3)

        self.downsample3 = downsample(w3, w3)
        self.pan4 = csp(w3 + w4, w4)

        self.downsample4 = downsample(w4, w4)
        self.pan5 = csp(w4 + w5, w5)

        self.out3 = out(w3)
        self.out4 = out(w4)
        self.out5 = out(w5)

        self.detect3 = detect(range(0, anchors_per_cell))
        self.detect4 = detect(range(anchors_per_cell, anchors_per_cell * 2))
        self.detect5 = detect(range(anchors_per_cell * 2, anchors_per_cell * 3))

    def forward(self, x: Tensor, targets: Optional[TARGETS] = None) -> NETWORK_OUTPUT:
        detections: List[Tensor] = []  # Outputs from detection layers
        losses: List[Tensor] = []  # Losses from detection layers
        hits: List[int] = []  # Number of targets each detection layer was responsible for

        image_size = get_image_size(x)

        c3, c4, x = self.backbone(x)[-3:]
        c5 = self.spp(x)

        x = torch.cat((self.upsample5(c5), self.pre4(c4)), dim=1)
        p4 = self.fpn4(x)
        x = torch.cat((self.upsample4(p4), self.pre3(c3)), dim=1)
        n3 = self.fpn3(x)
        x = torch.cat((self.downsample3(n3), p4), dim=1)
        n4 = self.pan4(x)
        x = torch.cat((self.downsample4(n4), c5), dim=1)
        n5 = self.pan5(x)

        self.detect3(self.out3(n3), targets, image_size, detections, losses, hits)
        self.detect4(self.out4(n4), targets, image_size, detections, losses, hits)
        self.detect5(self.out5(n5), targets, image_size, detections, losses, hits)
        return detections, losses, hits


class YOLOV4P6Network(nn.Module):
    """Network architecture that corresponds approximately to the variant of YOLOv4 with four detection layers.

    Args:
        num_classes: Number of different classes that this model predicts.
        backbone: A backbone network that returns the output from each stage.
        widths: Number of channels at each network stage.
        activation: Which layer activation to use. Can be "relu", "leaky", "mish", "silu" (or "swish"), "logistic",
            "linear", or "none".
        normalization: Which layer normalization to use. Can be "batchnorm", "groupnorm", or "none".
        prior_shapes: A list of prior box dimensions, used for scaling the predicted dimensions and possibly for
            matching the targets to the anchors. The list should contain [width, height] pairs in the network input
            resolution. There should be `4N` pairs, where `N` is the number of anchors per spatial location. They are
            assigned to the layers from the lowest (high-resolution) to the highest (low-resolution) layer, meaning that
            you typically want to sort the shapes from the smallest to the largest.
        matching_algorithm: Which algorithm to use for matching targets to anchors. "simota" (the SimOTA matching rule
            from YOLOX), "size" (match those prior shapes, whose width and height relative to the target is below given
            ratio), "iou" (match all prior shapes that give a high enough IoU), or "maxiou" (match the prior shape that
            gives the highest IoU, default).
        matching_threshold: Threshold for "size" and "iou" matching algorithms.
        spatial_range: The "simota" matching algorithm will restrict to anchors that are within an `N x N` grid cell
            area centered at the target, where `N` is the value of this parameter.
        size_range: The "simota" matching algorithm will restrict to anchors whose dimensions are no more than `N` and
            no less than `1/N` times the target dimensions, where `N` is the value of this parameter.
        ignore_bg_threshold: If a predictor is not responsible for predicting any target, but the prior shape has IoU
            with some target greater than this threshold, the predictor will not be taken into account when calculating
            the confidence loss.
        overlap_func: Which function to use for calculating the IoU between two sets of boxes. Valid values are "iou",
            "giou", "diou", and "ciou".
        predict_overlap: Balance between binary confidence targets and predicting the overlap. 0.0 means that the target
            confidence is 1 if there's an object, and 1.0 means that the target confidence is the output of
            ``overlap_func``.
        label_smoothing: The epsilon parameter (weight) for class label smoothing. 0.0 means no smoothing (binary
            targets), and 1.0 means that the target probabilities are always 0.5.
        overlap_loss_multiplier: Overlap loss will be scaled by this value.
        confidence_loss_multiplier: Confidence loss will be scaled by this value.
        class_loss_multiplier: Classification loss will be scaled by this value.
        xy_scale: Eliminate "grid sensitivity" by scaling the box coordinates by this factor. Using a value > 1.0 helps
            to produce coordinate values close to one.
    """

    def __init__(
        self,
        num_classes: int,
        backbone: Optional[nn.Module] = None,
        widths: Sequence[int] = (32, 64, 128, 256, 512, 1024, 1024),
        activation: Optional[str] = "silu",
        normalization: Optional[str] = "batchnorm",
        prior_shapes: Optional[PRIOR_SHAPES] = None,
        **kwargs: Any,
    ) -> None:
        super().__init__()

        # By default use the prior shapes that have been learned from the COCO data.
        if prior_shapes is None:
            prior_shapes = [
                [13, 17],
                [31, 25],
                [24, 51],
                [61, 45],
                [61, 45],
                [48, 102],
                [119, 96],
                [97, 189],
                [97, 189],
                [217, 184],
                [171, 384],
                [324, 451],
                [324, 451],
                [545, 357],
                [616, 618],
                [1024, 1024],
            ]
            anchors_per_cell = 4
        else:
            anchors_per_cell, modulo = divmod(len(prior_shapes), 4)
            if modulo != 0:
                raise ValueError("The number of provided prior shapes needs to be divisible by 4.")
        num_outputs = (5 + num_classes) * anchors_per_cell

        def spp(in_channels: int, out_channels: int) -> nn.Module:
            return CSPSPP(in_channels, out_channels, activation=activation, norm=normalization)

        def conv(in_channels: int, out_channels: int) -> nn.Module:
            return Conv(in_channels, out_channels, kernel_size=1, stride=1, activation=activation, norm=normalization)

        def csp(in_channels: int, out_channels: int) -> nn.Module:
            return CSPStage(
                in_channels,
                out_channels,
                depth=2,
                shortcut=False,
                norm=normalization,
                activation=activation,
            )

        def out(in_channels: int) -> nn.Module:
            conv = Conv(in_channels, in_channels, kernel_size=3, stride=1, activation=activation, norm=normalization)
            outputs = nn.Conv2d(in_channels, num_outputs, kernel_size=1)
            return nn.Sequential(OrderedDict([("conv", conv), (f"outputs_{num_outputs}", outputs)]))

        def upsample(in_channels: int, out_channels: int) -> nn.Module:
            channels = conv(in_channels, out_channels)
            upsample = nn.Upsample(scale_factor=2, mode="nearest")
            return nn.Sequential(OrderedDict([("channels", channels), ("upsample", upsample)]))

        def downsample(in_channels: int, out_channels: int) -> nn.Module:
            return Conv(in_channels, out_channels, kernel_size=3, stride=2, activation=activation, norm=normalization)

        def detect(prior_shape_idxs: Sequence[int]) -> DetectionStage:
            assert prior_shapes is not None
            return DetectionStage(
                prior_shapes=prior_shapes,
                prior_shape_idxs=list(prior_shape_idxs),
                num_classes=num_classes,
                input_is_normalized=False,
                **kwargs,
            )

        if backbone is not None:
            self.backbone = backbone
        else:
            self.backbone = YOLOV4Backbone(
                widths=widths, depths=(1, 1, 3, 15, 15, 7, 7), activation=activation, normalization=normalization
            )

        w3 = widths[-4]
        w4 = widths[-3]
        w5 = widths[-2]
        w6 = widths[-1]

        self.spp = spp(w6, w6)

        self.pre5 = conv(w5, w5 // 2)
        self.upsample6 = upsample(w6, w5 // 2)
        self.fpn5 = csp(w5, w5)

        self.pre4 = conv(w4, w4 // 2)
        self.upsample5 = upsample(w5, w4 // 2)
        self.fpn4 = csp(w4, w4)

        self.pre3 = conv(w3, w3 // 2)
        self.upsample4 = upsample(w4, w3 // 2)
        self.fpn3 = csp(w3, w3)

        self.downsample3 = downsample(w3, w3)
        self.pan4 = csp(w3 + w4, w4)

        self.downsample4 = downsample(w4, w4)
        self.pan5 = csp(w4 + w5, w5)

        self.downsample5 = downsample(w5, w5)
        self.pan6 = csp(w5 + w6, w6)

        self.out3 = out(w3)
        self.out4 = out(w4)
        self.out5 = out(w5)
        self.out6 = out(w6)

        self.detect3 = detect(range(0, anchors_per_cell))
        self.detect4 = detect(range(anchors_per_cell, anchors_per_cell * 2))
        self.detect5 = detect(range(anchors_per_cell * 2, anchors_per_cell * 3))
        self.detect6 = detect(range(anchors_per_cell * 3, anchors_per_cell * 4))

    def forward(self, x: Tensor, targets: Optional[TARGETS] = None) -> NETWORK_OUTPUT:
        detections: List[Tensor] = []  # Outputs from detection layers
        losses: List[Tensor] = []  # Losses from detection layers
        hits: List[int] = []  # Number of targets each detection layer was responsible for

        image_size = get_image_size(x)

        c3, c4, c5, x = self.backbone(x)[-4:]
        c6 = self.spp(x)

        x = torch.cat((self.upsample6(c6), self.pre5(c5)), dim=1)
        p5 = self.fpn5(x)
        x = torch.cat((self.upsample5(p5), self.pre4(c4)), dim=1)
        p4 = self.fpn4(x)
        x = torch.cat((self.upsample4(p4), self.pre3(c3)), dim=1)
        n3 = self.fpn3(x)
        x = torch.cat((self.downsample3(n3), p4), dim=1)
        n4 = self.pan4(x)
        x = torch.cat((self.downsample4(n4), p5), dim=1)
        n5 = self.pan5(x)
        x = torch.cat((self.downsample5(n5), c6), dim=1)
        n6 = self.pan6(x)

        self.detect3(self.out3(n3), targets, image_size, detections, losses, hits)
        self.detect4(self.out4(n4), targets, image_size, detections, losses, hits)
        self.detect5(self.out5(n5), targets, image_size, detections, losses, hits)
        self.detect6(self.out6(n6), targets, image_size, detections, losses, hits)
        return detections, losses, hits


class YOLOV5Network(nn.Module):
    """The YOLOv5 network architecture. Different variants (n/s/m/l/x) can be achieved by adjusting the ``depth``
    and ``width`` parameters.

    Args:
        num_classes: Number of different classes that this model predicts.
        backbone: A backbone network that returns the output from each stage.
        width: Number of channels in the narrowest convolutional layer. The wider convolutional layers will use a number
            of channels that is a multiple of this value. The values used by the different variants are 16 (yolov5n), 32
            (yolov5s), 48 (yolov5m), 64 (yolov5l), and 80 (yolov5x).
        depth: Repeat the bottleneck layers this many times. Can be used to make the network deeper. The values used by
            the different variants are 1 (yolov5n, yolov5s), 2 (yolov5m), 3 (yolov5l), and 4 (yolov5x).
        activation: Which layer activation to use. Can be "relu", "leaky", "mish", "silu" (or "swish"), "logistic",
            "linear", or "none".
        normalization: Which layer normalization to use. Can be "batchnorm", "groupnorm", or "none".
        prior_shapes: A list of prior box dimensions, used for scaling the predicted dimensions and possibly for
            matching the targets to the anchors. The list should contain [width, height] pairs in the network input
            resolution. There should be `3N` pairs, where `N` is the number of anchors per spatial location. They are
            assigned to the layers from the lowest (high-resolution) to the highest (low-resolution) layer, meaning that
            you typically want to sort the shapes from the smallest to the largest.
        matching_algorithm: Which algorithm to use for matching targets to anchors. "simota" (the SimOTA matching rule
            from YOLOX), "size" (match those prior shapes, whose width and height relative to the target is below given
            ratio), "iou" (match all prior shapes that give a high enough IoU), or "maxiou" (match the prior shape that
            gives the highest IoU, default).
        matching_threshold: Threshold for "size" and "iou" matching algorithms.
        spatial_range: The "simota" matching algorithm will restrict to anchors that are within an `N x N` grid cell
            area centered at the target, where `N` is the value of this parameter.
        size_range: The "simota" matching algorithm will restrict to anchors whose dimensions are no more than `N` and
            no less than `1/N` times the target dimensions, where `N` is the value of this parameter.
        ignore_bg_threshold: If a predictor is not responsible for predicting any target, but the prior shape has IoU
            with some target greater than this threshold, the predictor will not be taken into account when calculating
            the confidence loss.
        overlap_func: Which function to use for calculating the IoU between two sets of boxes. Valid values are "iou",
            "giou", "diou", and "ciou".
        predict_overlap: Balance between binary confidence targets and predicting the overlap. 0.0 means that the target
            confidence is 1 if there's an object, and 1.0 means that the target confidence is the output of
            ``overlap_func``.
        label_smoothing: The epsilon parameter (weight) for class label smoothing. 0.0 means no smoothing (binary
            targets), and 1.0 means that the target probabilities are always 0.5.
        overlap_loss_multiplier: Overlap loss will be scaled by this value.
        confidence_loss_multiplier: Confidence loss will be scaled by this value.
        class_loss_multiplier: Classification loss will be scaled by this value.
        xy_scale: Eliminate "grid sensitivity" by scaling the box coordinates by this factor. Using a value > 1.0 helps
            to produce coordinate values close to one.
    """

    def __init__(
        self,
        num_classes: int,
        backbone: Optional[nn.Module] = None,
        width: int = 64,
        depth: int = 3,
        activation: Optional[str] = "silu",
        normalization: Optional[str] = "batchnorm",
        prior_shapes: Optional[PRIOR_SHAPES] = None,
        **kwargs: Any,
    ) -> None:
        super().__init__()

        # By default use the prior shapes that have been learned from the COCO data.
        if prior_shapes is None:
            prior_shapes = [
                [12, 16],
                [19, 36],
                [40, 28],
                [36, 75],
                [76, 55],
                [72, 146],
                [142, 110],
                [192, 243],
                [459, 401],
            ]
            anchors_per_cell = 3
        else:
            anchors_per_cell, modulo = divmod(len(prior_shapes), 3)
            if modulo != 0:
                raise ValueError("The number of provided prior shapes needs to be divisible by 3.")
        num_outputs = (5 + num_classes) * anchors_per_cell

        def spp(in_channels: int, out_channels: int) -> nn.Module:
            return FastSPP(in_channels, out_channels, activation=activation, norm=normalization)

        def downsample(in_channels: int, out_channels: int) -> nn.Module:
            return Conv(in_channels, out_channels, kernel_size=3, stride=2, activation=activation, norm=normalization)

        def conv(in_channels: int, out_channels: int) -> nn.Module:
            return Conv(in_channels, out_channels, kernel_size=1, stride=1, activation=activation, norm=normalization)

        def out(in_channels: int) -> nn.Module:
            outputs = nn.Conv2d(in_channels, num_outputs, kernel_size=1)
            return nn.Sequential(OrderedDict([(f"outputs_{num_outputs}", outputs)]))

        def csp(in_channels: int, out_channels: int) -> nn.Module:
            return CSPStage(
                in_channels,
                out_channels,
                depth=depth,
                shortcut=False,
                norm=normalization,
                activation=activation,
            )

        def detect(prior_shape_idxs: Sequence[int]) -> DetectionStage:
            assert prior_shapes is not None
            return DetectionStage(
                prior_shapes=prior_shapes,
                prior_shape_idxs=list(prior_shape_idxs),
                num_classes=num_classes,
                input_is_normalized=False,
                **kwargs,
            )

        self.backbone = backbone or YOLOV5Backbone(
            depth=depth, width=width, activation=activation, normalization=normalization
        )

        self.spp = spp(width * 16, width * 16)

        self.pan3 = csp(width * 8, width * 4)
        self.out3 = out(width * 4)

        self.fpn4 = nn.Sequential(
            OrderedDict(
                [
                    ("csp", csp(width * 16, width * 8)),
                    ("conv", conv(width * 8, width * 4)),
                ]
            )
        )
        self.pan4 = csp(width * 8, width * 8)
        self.out4 = out(width * 8)

        self.fpn5 = conv(width * 16, width * 8)
        self.pan5 = csp(width * 16, width * 16)
        self.out5 = out(width * 16)

        self.upsample = nn.Upsample(scale_factor=2, mode="nearest")

        self.downsample3 = downsample(width * 4, width * 4)
        self.downsample4 = downsample(width * 8, width * 8)

        self.detect3 = detect(range(0, anchors_per_cell))
        self.detect4 = detect(range(anchors_per_cell, anchors_per_cell * 2))
        self.detect5 = detect(range(anchors_per_cell * 2, anchors_per_cell * 3))

    def forward(self, x: Tensor, targets: Optional[TARGETS] = None) -> NETWORK_OUTPUT:
        detections: List[Tensor] = []  # Outputs from detection layers
        losses: List[Tensor] = []  # Losses from detection layers
        hits: List[int] = []  # Number of targets each detection layer was responsible for

        image_size = get_image_size(x)

        c3, c4, x = self.backbone(x)[-3:]
        c5 = self.spp(x)

        p5 = self.fpn5(c5)
        x = torch.cat((self.upsample(p5), c4), dim=1)
        p4 = self.fpn4(x)
        x = torch.cat((self.upsample(p4), c3), dim=1)

        n3 = self.pan3(x)
        x = torch.cat((self.downsample3(n3), p4), dim=1)
        n4 = self.pan4(x)
        x = torch.cat((self.downsample4(n4), p5), dim=1)
        n5 = self.pan5(x)

        self.detect3(self.out3(n3), targets, image_size, detections, losses, hits)
        self.detect4(self.out4(n4), targets, image_size, detections, losses, hits)
        self.detect5(self.out5(n5), targets, image_size, detections, losses, hits)
        return detections, losses, hits


class YOLOV7Network(nn.Module):
    """Network architecture that corresponds to the W6 variant of YOLOv7 with four detection layers.

    Args:
        num_classes: Number of different classes that this model predicts.
        backbone: A backbone network that returns the output from each stage.
        widths: Number of channels at each network stage.
        activation: Which layer activation to use. Can be "relu", "leaky", "mish", "silu" (or "swish"), "logistic",
            "linear", or "none".
        normalization: Which layer normalization to use. Can be "batchnorm", "groupnorm", or "none".
        prior_shapes: A list of prior box dimensions, used for scaling the predicted dimensions and possibly for
            matching the targets to the anchors. The list should contain [width, height] pairs in the network input
            resolution. There should be `4N` pairs, where `N` is the number of anchors per spatial location. They are
            assigned to the layers from the lowest (high-resolution) to the highest (low-resolution) layer, meaning that
            you typically want to sort the shapes from the smallest to the largest.
        matching_algorithm: Which algorithm to use for matching targets to anchors. "simota" (the SimOTA matching rule
            from YOLOX), "size" (match those prior shapes, whose width and height relative to the target is below given
            ratio), "iou" (match all prior shapes that give a high enough IoU), or "maxiou" (match the prior shape that
            gives the highest IoU, default).
        matching_threshold: Threshold for "size" and "iou" matching algorithms.
        spatial_range: The "simota" matching algorithm will restrict to anchors that are within an `N x N` grid cell
            area centered at the target. This parameter specifies `N` for the lead head.
        aux_spatial_range: The "simota" matching algorithm will restrict to anchors that are within an `N x N` grid cell
            area centered at the target. This parameter specifies `N` for the auxiliary head.
        size_range: The "simota" matching algorithm will restrict to anchors whose dimensions are no more than `N` and
            no less than `1/N` times the target dimensions, where `N` is the value of this parameter.
        ignore_bg_threshold: If a predictor is not responsible for predicting any target, but the prior shape has IoU
            with some target greater than this threshold, the predictor will not be taken into account when calculating
            the confidence loss.
        overlap_func: Which function to use for calculating the IoU between two sets of boxes. Valid values are "iou",
            "giou", "diou", and "ciou".
        predict_overlap: Balance between binary confidence targets and predicting the overlap. 0.0 means that the target
            confidence is 1 if there's an object, and 1.0 means that the target confidence is the output of
            ``overlap_func``.
        label_smoothing: The epsilon parameter (weight) for class label smoothing. 0.0 means no smoothing (binary
            targets), and 1.0 means that the target probabilities are always 0.5.
        overlap_loss_multiplier: Overlap loss will be scaled by this value.
        confidence_loss_multiplier: Confidence loss will be scaled by this value.
        class_loss_multiplier: Classification loss will be scaled by this value.
        xy_scale: Eliminate "grid sensitivity" by scaling the box coordinates by this factor. Using a value > 1.0 helps
            to produce coordinate values close to one.
        aux_weight: Weight for the loss from the auxiliary heads.
    """

    def __init__(
        self,
        num_classes: int,
        backbone: Optional[nn.Module] = None,
        widths: Sequence[int] = (64, 128, 256, 512, 768, 1024),
        activation: Optional[str] = "silu",
        normalization: Optional[str] = "batchnorm",
        prior_shapes: Optional[PRIOR_SHAPES] = None,
        **kwargs: Any,
    ) -> None:
        super().__init__()

        # By default use the prior shapes that have been learned from the COCO data.
        if prior_shapes is None:
            prior_shapes = [
                [13, 17],
                [31, 25],
                [24, 51],
                [61, 45],
                [61, 45],
                [48, 102],
                [119, 96],
                [97, 189],
                [97, 189],
                [217, 184],
                [171, 384],
                [324, 451],
                [324, 451],
                [545, 357],
                [616, 618],
                [1024, 1024],
            ]
            anchors_per_cell = 4
        else:
            anchors_per_cell, modulo = divmod(len(prior_shapes), 4)
            if modulo != 0:
                raise ValueError("The number of provided prior shapes needs to be divisible by 4.")
        num_outputs = (5 + num_classes) * anchors_per_cell

        def spp(in_channels: int, out_channels: int) -> nn.Module:
            return CSPSPP(in_channels, out_channels, activation=activation, norm=normalization)

        def conv(in_channels: int, out_channels: int) -> nn.Module:
            return Conv(in_channels, out_channels, kernel_size=1, stride=1, activation=activation, norm=normalization)

        def elan(in_channels: int, out_channels: int) -> nn.Module:
            return ELANStage(
                in_channels,
                out_channels,
                split_channels=out_channels,
                depth=4,
                block_depth=1,
                norm=normalization,
                activation=activation,
            )

        def out(in_channels: int, hidden_channels: int) -> nn.Module:
            conv = Conv(
                in_channels, hidden_channels, kernel_size=3, stride=1, activation=activation, norm=normalization
            )
            outputs = nn.Conv2d(hidden_channels, num_outputs, kernel_size=1)
            return nn.Sequential(OrderedDict([("conv", conv), (f"outputs_{num_outputs}", outputs)]))

        def upsample(in_channels: int, out_channels: int) -> nn.Module:
            channels = conv(in_channels, out_channels)
            upsample = nn.Upsample(scale_factor=2, mode="nearest")
            return nn.Sequential(OrderedDict([("channels", channels), ("upsample", upsample)]))

        def downsample(in_channels: int, out_channels: int) -> nn.Module:
            return Conv(in_channels, out_channels, kernel_size=3, stride=2, activation=activation, norm=normalization)

        def detect(prior_shape_idxs: Sequence[int]) -> DetectionStageWithAux:
            assert prior_shapes is not None
            return DetectionStageWithAux(
                prior_shapes=prior_shapes,
                prior_shape_idxs=list(prior_shape_idxs),
                num_classes=num_classes,
                input_is_normalized=False,
                **kwargs,
            )

        if backbone is not None:
            self.backbone = backbone
        else:
            self.backbone = YOLOV7Backbone(
                widths=widths, depth=2, block_depth=2, activation=activation, normalization=normalization
            )

        w3 = widths[-4]
        w4 = widths[-3]
        w5 = widths[-2]
        w6 = widths[-1]

        self.spp = spp(w6, w6 // 2)

        self.pre5 = conv(w5, w5 // 2)
        self.upsample6 = upsample(w6 // 2, w5 // 2)
        self.fpn5 = elan(w5, w5 // 2)

        self.pre4 = conv(w4, w4 // 2)
        self.upsample5 = upsample(w5 // 2, w4 // 2)
        self.fpn4 = elan(w4, w4 // 2)

        self.pre3 = conv(w3, w3 // 2)
        self.upsample4 = upsample(w4 // 2, w3 // 2)
        self.fpn3 = elan(w3, w3 // 2)

        self.downsample3 = downsample(w3 // 2, w4 // 2)
        self.pan4 = elan(w4, w4 // 2)

        self.downsample4 = downsample(w4 // 2, w5 // 2)
        self.pan5 = elan(w5, w5 // 2)

        self.downsample5 = downsample(w5 // 2, w6 // 2)
        self.pan6 = elan(w6, w6 // 2)

        self.out3 = out(w3 // 2, w3)
        self.aux_out3 = out(w3 // 2, w3 + (w3 // 4))
        self.out4 = out(w4 // 2, w4)
        self.aux_out4 = out(w4 // 2, w4 + (w4 // 4))
        self.out5 = out(w5 // 2, w5)
        self.aux_out5 = out(w5 // 2, w5 + (w5 // 4))
        self.out6 = out(w6 // 2, w6)
        self.aux_out6 = out(w6 // 2, w6 + (w6 // 4))

        self.detect3 = detect(range(0, anchors_per_cell))
        self.detect4 = detect(range(anchors_per_cell, anchors_per_cell * 2))
        self.detect5 = detect(range(anchors_per_cell * 2, anchors_per_cell * 3))
        self.detect6 = detect(range(anchors_per_cell * 3, anchors_per_cell * 4))

    def forward(self, x: Tensor, targets: Optional[TARGETS] = None) -> NETWORK_OUTPUT:
        detections: List[Tensor] = []  # Outputs from detection layers
        losses: List[Tensor] = []  # Losses from detection layers
        hits: List[int] = []  # Number of targets each detection layer was responsible for

        image_size = get_image_size(x)

        c3, c4, c5, x = self.backbone(x)[-4:]
        c6 = self.spp(x)

        x = torch.cat((self.upsample6(c6), self.pre5(c5)), dim=1)
        p5 = self.fpn5(x)
        x = torch.cat((self.upsample5(p5), self.pre4(c4)), dim=1)
        p4 = self.fpn4(x)
        x = torch.cat((self.upsample4(p4), self.pre3(c3)), dim=1)
        n3 = self.fpn3(x)
        x = torch.cat((self.downsample3(n3), p4), dim=1)
        n4 = self.pan4(x)
        x = torch.cat((self.downsample4(n4), p5), dim=1)
        n5 = self.pan5(x)
        x = torch.cat((self.downsample5(n5), c6), dim=1)
        n6 = self.pan6(x)

        self.detect3(self.out3(n3), self.aux_out3(n3), targets, image_size, detections, losses, hits)
        self.detect4(self.out4(n4), self.aux_out4(p4), targets, image_size, detections, losses, hits)
        self.detect5(self.out5(n5), self.aux_out5(p5), targets, image_size, detections, losses, hits)
        self.detect6(self.out6(n6), self.aux_out6(c6), targets, image_size, detections, losses, hits)
        return detections, losses, hits


class YOLOXHead(nn.Module):
    """A module that produces features for YOLO detection layer, decoupling the classification and localization
    features.

    Args:
        in_channels: Number of input channels that the module expects.
        hidden_channels: Number of output channels in the hidden layers.
        anchors_per_cell: Number of detections made at each spatial location of the feature map.
        num_classes: Number of different classes that this model predicts.
        activation: Which layer activation to use. Can be "relu", "leaky", "mish", "silu" (or "swish"), "logistic",
            "linear", or "none".
        norm: Which layer normalization to use. Can be "batchnorm", "groupnorm", or "none".
    """

    def __init__(
        self,
        in_channels: int,
        hidden_channels: int,
        anchors_per_cell: int,
        num_classes: int,
        activation: Optional[str] = "silu",
        norm: Optional[str] = "batchnorm",
    ) -> None:
        super().__init__()

        def conv(in_channels: int, out_channels: int, kernel_size: int = 1) -> nn.Module:
            return Conv(in_channels, out_channels, kernel_size, stride=1, activation=activation, norm=norm)

        def linear(in_channels: int, out_channels: int) -> nn.Module:
            return nn.Conv2d(in_channels, out_channels, kernel_size=1)

        def features(num_channels: int) -> nn.Module:
            return nn.Sequential(
                conv(num_channels, num_channels, kernel_size=3),
                conv(num_channels, num_channels, kernel_size=3),
            )

        def classprob(num_channels: int) -> nn.Module:
            num_outputs = anchors_per_cell * num_classes
            outputs = linear(num_channels, num_outputs)
            return nn.Sequential(OrderedDict([("convs", features(num_channels)), (f"outputs_{num_outputs}", outputs)]))

        self.stem = conv(in_channels, hidden_channels)
        self.feat = features(hidden_channels)
        self.box = linear(hidden_channels, anchors_per_cell * 4)
        self.confidence = linear(hidden_channels, anchors_per_cell)
        self.classprob = classprob(hidden_channels)

    def forward(self, x: Tensor) -> Tensor:
        x = self.stem(x)
        features = self.feat(x)
        box = self.box(features)
        confidence = self.confidence(features)
        classprob = self.classprob(x)
        return torch.cat((box, confidence, classprob), dim=1)


class YOLOXNetwork(nn.Module):
    """The YOLOX network architecture. Different variants (nano/tiny/s/m/l/x) can be achieved by adjusting the
    ``depth`` and ``width`` parameters.

    Args:
        num_classes: Number of different classes that this model predicts.
        backbone: A backbone network that returns the output from each stage.
        width: Number of channels in the narrowest convolutional layer. The wider convolutional layers will use a number
            of channels that is a multiple of this value. The values used by the different variants are 24 (yolox-tiny),
            32 (yolox-s), 48 (yolox-m), and 64 (yolox-l).
        depth: Repeat the bottleneck layers this many times. Can be used to make the network deeper. The values used by
            the different variants are 1 (yolox-tiny, yolox-s), 2 (yolox-m), and 3 (yolox-l).
        activation: Which layer activation to use. Can be "relu", "leaky", "mish", "silu" (or "swish"), "logistic",
            "linear", or "none".
        normalization: Which layer normalization to use. Can be "batchnorm", "groupnorm", or "none".
        prior_shapes: A list of prior box dimensions, used for scaling the predicted dimensions and possibly for
            matching the targets to the anchors. The list should contain [width, height] pairs in the network input
            resolution. There should be `3N` pairs, where `N` is the number of anchors per spatial location. They are
            assigned to the layers from the lowest (high-resolution) to the highest (low-resolution) layer, meaning that
            you typically want to sort the shapes from the smallest to the largest.
        matching_algorithm: Which algorithm to use for matching targets to anchors. "simota" (the SimOTA matching rule
            from YOLOX), "size" (match those prior shapes, whose width and height relative to the target is below given
            ratio), "iou" (match all prior shapes that give a high enough IoU), or "maxiou" (match the prior shape that
            gives the highest IoU, default).
        matching_threshold: Threshold for "size" and "iou" matching algorithms.
        spatial_range: The "simota" matching algorithm will restrict to anchors that are within an `N x N` grid cell
            area centered at the target, where `N` is the value of this parameter.
        size_range: The "simota" matching algorithm will restrict to anchors whose dimensions are no more than `N` and
            no less than `1/N` times the target dimensions, where `N` is the value of this parameter.
        ignore_bg_threshold: If a predictor is not responsible for predicting any target, but the prior shape has IoU
            with some target greater than this threshold, the predictor will not be taken into account when calculating
            the confidence loss.
        overlap_func: Which function to use for calculating the IoU between two sets of boxes. Valid values are "iou",
            "giou", "diou", and "ciou".
        predict_overlap: Balance between binary confidence targets and predicting the overlap. 0.0 means that the target
            confidence is 1 if there's an object, and 1.0 means that the target confidence is the output of
            ``overlap_func``.
        label_smoothing: The epsilon parameter (weight) for class label smoothing. 0.0 means no smoothing (binary
            targets), and 1.0 means that the target probabilities are always 0.5.
        overlap_loss_multiplier: Overlap loss will be scaled by this value.
        confidence_loss_multiplier: Confidence loss will be scaled by this value.
        class_loss_multiplier: Classification loss will be scaled by this value.
        xy_scale: Eliminate "grid sensitivity" by scaling the box coordinates by this factor. Using a value > 1.0 helps
            to produce coordinate values close to one.
    """

    def __init__(
        self,
        num_classes: int,
        backbone: Optional[nn.Module] = None,
        width: int = 64,
        depth: int = 3,
        activation: Optional[str] = "silu",
        normalization: Optional[str] = "batchnorm",
        prior_shapes: Optional[PRIOR_SHAPES] = None,
        **kwargs: Any,
    ) -> None:
        super().__init__()

        # By default use one anchor per cell and the stride as the prior size.
        if prior_shapes is None:
            prior_shapes = [[8, 8], [16, 16], [32, 32]]
            anchors_per_cell = 1
        else:
            anchors_per_cell, modulo = divmod(len(prior_shapes), 3)
            if modulo != 0:
                raise ValueError("The number of provided prior shapes needs to be divisible by 3.")

        def spp(in_channels: int, out_channels: int) -> nn.Module:
            return FastSPP(in_channels, out_channels, activation=activation, norm=normalization)

        def downsample(in_channels: int, out_channels: int) -> nn.Module:
            return Conv(in_channels, out_channels, kernel_size=3, stride=2, activation=activation, norm=normalization)

        def conv(in_channels: int, out_channels: int, kernel_size: int = 1) -> nn.Module:
            return Conv(in_channels, out_channels, kernel_size, stride=1, activation=activation, norm=normalization)

        def csp(in_channels: int, out_channels: int) -> nn.Module:
            return CSPStage(
                in_channels,
                out_channels,
                depth=depth,
                shortcut=False,
                norm=normalization,
                activation=activation,
            )

        def head(in_channels: int, hidden_channels: int) -> YOLOXHead:
            return YOLOXHead(
                in_channels,
                hidden_channels,
                anchors_per_cell,
                num_classes,
                activation=activation,
                norm=normalization,
            )

        def detect(prior_shape_idxs: Sequence[int]) -> DetectionStage:
            assert prior_shapes is not None
            return DetectionStage(
                prior_shapes=prior_shapes,
                prior_shape_idxs=list(prior_shape_idxs),
                num_classes=num_classes,
                input_is_normalized=False,
                **kwargs,
            )

        self.backbone = backbone or YOLOV5Backbone(
            depth=depth, width=width, activation=activation, normalization=normalization
        )

        self.spp = spp(width * 16, width * 16)

        self.pan3 = csp(width * 8, width * 4)
        self.out3 = head(width * 4, width * 4)

        self.fpn4 = nn.Sequential(
            OrderedDict(
                [
                    ("csp", csp(width * 16, width * 8)),
                    ("conv", conv(width * 8, width * 4)),
                ]
            )
        )
        self.pan4 = csp(width * 8, width * 8)
        self.out4 = head(width * 8, width * 4)

        self.fpn5 = conv(width * 16, width * 8)
        self.pan5 = csp(width * 16, width * 16)
        self.out5 = head(width * 16, width * 4)

        self.upsample = nn.Upsample(scale_factor=2, mode="nearest")

        self.downsample3 = downsample(width * 4, width * 4)
        self.downsample4 = downsample(width * 8, width * 8)

        self.detect3 = detect(range(0, anchors_per_cell))
        self.detect4 = detect(range(anchors_per_cell, anchors_per_cell * 2))
        self.detect5 = detect(range(anchors_per_cell * 2, anchors_per_cell * 3))

    def forward(self, x: Tensor, targets: Optional[TARGETS] = None) -> NETWORK_OUTPUT:
        detections: List[Tensor] = []  # Outputs from detection layers
        losses: List[Tensor] = []  # Losses from detection layers
        hits: List[int] = []  # Number of targets each detection layer was responsible for

        image_size = get_image_size(x)

        c3, c4, x = self.backbone(x)[-3:]
        c5 = self.spp(x)

        p5 = self.fpn5(c5)
        x = torch.cat((self.upsample(p5), c4), dim=1)
        p4 = self.fpn4(x)
        x = torch.cat((self.upsample(p4), c3), dim=1)

        n3 = self.pan3(x)
        x = torch.cat((self.downsample3(n3), p4), dim=1)
        n4 = self.pan4(x)
        x = torch.cat((self.downsample4(n4), p5), dim=1)
        n5 = self.pan5(x)

        self.detect3(self.out3(n3), targets, image_size, detections, losses, hits)
        self.detect4(self.out4(n4), targets, image_size, detections, losses, hits)
        self.detect5(self.out5(n5), targets, image_size, detections, losses, hits)
        return detections, losses, hits


class DarknetNetwork(nn.Module):
    """This class can be used to parse the configuration files of the Darknet YOLOv4 implementation.

    Iterates through the layers from the configuration and creates corresponding PyTorch modules. If ``weights_path`` is
    given and points to a Darknet model file, loads the convolutional layer weights from the file.

    Args:
        config_path: Path to a Darknet configuration file that defines the network architecture.
        weights_path: Path to a Darknet model file. If given, the model weights will be read from this file.
        in_channels: Number of channels in the input image.
        matching_algorithm: Which algorithm to use for matching targets to anchors. "simota" (the SimOTA matching rule
            from YOLOX), "size" (match those prior shapes, whose width and height relative to the target is below given
            ratio), "iou" (match all prior shapes that give a high enough IoU), or "maxiou" (match the prior shape that
            gives the highest IoU, default).
        matching_threshold: Threshold for "size" and "iou" matching algorithms.
        spatial_range: The "simota" matching algorithm will restrict to anchors that are within an `N x N` grid cell
            area centered at the target, where `N` is the value of this parameter.
        size_range: The "simota" matching algorithm will restrict to anchors whose dimensions are no more than `N` and
            no less than `1/N` times the target dimensions, where `N` is the value of this parameter.
        ignore_bg_threshold: If a predictor is not responsible for predicting any target, but the corresponding anchor
            has IoU with some target greater than this threshold, the predictor will not be taken into account when
            calculating the confidence loss.
        overlap_func: Which function to use for calculating the IoU between two sets of boxes. Valid values are "iou",
            "giou", "diou", and "ciou".
        predict_overlap: Balance between binary confidence targets and predicting the overlap. 0.0 means that the target
            confidence is 1 if there's an object, and 1.0 means that the target confidence is the output of
            ``overlap_func``.
        label_smoothing: The epsilon parameter (weight) for class label smoothing. 0.0 means no smoothing (binary
            targets), and 1.0 means that the target probabilities are always 0.5.
        overlap_loss_multiplier: Overlap loss will be scaled by this value.
        confidence_loss_multiplier: Confidence loss will be scaled by this value.
        class_loss_multiplier: Classification loss will be scaled by this value.
    """

    def __init__(
        self, config_path: str, weights_path: Optional[str] = None, in_channels: Optional[int] = None, **kwargs: Any
    ) -> None:
        super().__init__()

        with open(config_path) as config_file:
            sections = self._read_config(config_file)

        if len(sections) < 2:
            raise ValueError("The model configuration file should include at least two sections.")

        self.__dict__.update(sections[0])
        global_config = sections[0]
        layer_configs = sections[1:]

        if in_channels is None:
            in_channels = global_config.get("channels", 3)
            assert isinstance(in_channels, int)

        self.layers = nn.ModuleList()
        # num_inputs will contain the number of channels in the input of every layer up to the current layer. It is
        # initialized with the number of channels in the input image.
        num_inputs = [in_channels]
        for layer_config in layer_configs:
            config = {**global_config, **layer_config}
            layer, num_outputs = _create_layer(config, num_inputs, **kwargs)
            self.layers.append(layer)
            num_inputs.append(num_outputs)

        if weights_path is not None:
            with open(weights_path) as weight_file:
                self.load_weights(weight_file)

    def forward(self, x: Tensor, targets: Optional[TARGETS] = None) -> NETWORK_OUTPUT:
        outputs: List[Tensor] = []  # Outputs from all layers
        detections: List[Tensor] = []  # Outputs from detection layers
        losses: List[Tensor] = []  # Losses from detection layers
        hits: List[int] = []  # Number of targets each detection layer was responsible for

        image_size = get_image_size(x)

        for layer in self.layers:
            if isinstance(layer, (RouteLayer, ShortcutLayer)):
                x = layer(outputs)
            elif isinstance(layer, DetectionLayer):
                x, preds = layer(x, image_size)
                detections.append(x)
                if targets is not None:
                    layer_losses, layer_hits = layer.calculate_losses(preds, targets, image_size)
                    losses.append(layer_losses)
                    hits.append(layer_hits)
            else:
                x = layer(x)

            outputs.append(x)

        return detections, losses, hits

    def load_weights(self, weight_file: io.IOBase) -> None:
        """Loads weights to layer modules from a pretrained Darknet model.

        One may want to continue training from pretrained weights, on a dataset with a different number of object
        categories. The number of kernels in the convolutional layers just before each detection layer depends on the
        number of output classes. The Darknet solution is to truncate the weight file and stop reading weights at the
        first incompatible layer. For this reason the function silently leaves the rest of the layers unchanged, when
        the weight file ends.

        Args:
            weight_file: A file-like object containing model weights in the Darknet binary format.
        """
        if not isinstance(weight_file, io.IOBase):
            raise ValueError("weight_file must be a file-like object.")

        version = np.fromfile(weight_file, count=3, dtype=np.int32)
        images_seen = np.fromfile(weight_file, count=1, dtype=np.int64)
        print(
            f"Loading weights from Darknet model version {version[0]}.{version[1]}.{version[2]} "
            f"that has been trained on {images_seen[0]} images."
        )

        def read(tensor: Tensor) -> int:
            """Reads the contents of ``tensor`` from the current position of ``weight_file``.

            Returns the number of elements read. If there's no more data in ``weight_file``, returns 0.
            """
            np_array = np.fromfile(weight_file, count=tensor.numel(), dtype=np.float32)
            num_elements = np_array.size
            if num_elements > 0:
                source = torch.from_numpy(np_array).view_as(tensor)
                with torch.no_grad():
                    tensor.copy_(source)
            return num_elements

        for layer in self.layers:
            # Weights are loaded only to convolutional layers
            if not isinstance(layer, Conv):
                continue

            # If convolution is followed by batch normalization, read the batch normalization parameters. Otherwise we
            # read the convolution bias.
            if isinstance(layer.norm, nn.Identity):
                assert layer.conv.bias is not None
                read(layer.conv.bias)
            else:
                assert isinstance(layer.norm, nn.BatchNorm2d)
                assert layer.norm.running_mean is not None
                assert layer.norm.running_var is not None
                read(layer.norm.bias)
                read(layer.norm.weight)
                read(layer.norm.running_mean)
                read(layer.norm.running_var)

            read_count = read(layer.conv.weight)
            if read_count == 0:
                return

    def _read_config(self, config_file: Iterable[str]) -> List[Dict[str, Any]]:
        """Reads a Darnet network configuration file and returns a list of configuration sections.

        Args:
            config_file: The configuration file to read.

        Returns:
            A list of configuration sections.
        """
        section_re = re.compile(r"\[([^]]+)\]")
        list_variables = ("layers", "anchors", "mask", "scales")
        variable_types = {
            "activation": str,
            "anchors": int,
            "angle": float,
            "batch": int,
            "batch_normalize": bool,
            "beta_nms": float,
            "burn_in": int,
            "channels": int,
            "classes": int,
            "cls_normalizer": float,
            "decay": float,
            "exposure": float,
            "filters": int,
            "from": int,
            "groups": int,
            "group_id": int,
            "height": int,
            "hue": float,
            "ignore_thresh": float,
            "iou_loss": str,
            "iou_normalizer": float,
            "iou_thresh": float,
            "jitter": float,
            "layers": int,
            "learning_rate": float,
            "mask": int,
            "max_batches": int,
            "max_delta": float,
            "momentum": float,
            "mosaic": bool,
            "new_coords": int,
            "nms_kind": str,
            "num": int,
            "obj_normalizer": float,
            "pad": bool,
            "policy": str,
            "random": bool,
            "resize": float,
            "saturation": float,
            "scales": float,
            "scale_x_y": float,
            "size": int,
            "steps": str,
            "stride": int,
            "subdivisions": int,
            "truth_thresh": float,
            "width": int,
        }

        section = None
        sections = []

        def convert(key: str, value: str) -> Union[str, int, float, List[Union[str, int, float]]]:
            """Converts a value to the correct type based on key."""
            if key not in variable_types:
                warn("Unknown YOLO configuration variable: " + key)
                return value
            if key in list_variables:
                return [variable_types[key](v) for v in value.split(",")]
            else:
                return variable_types[key](value)

        for line in config_file:
            line = line.strip()
            if (not line) or (line[0] == "#"):
                continue

            section_match = section_re.match(line)
            if section_match:
                if section is not None:
                    sections.append(section)
                section = {"type": section_match.group(1)}
            else:
                if section is None:
                    raise RuntimeError("Darknet network configuration file does not start with a section header.")
                key, value = line.split("=")
                key = key.rstrip()
                value = value.lstrip()
                section[key] = convert(key, value)
        if section is not None:
            sections.append(section)

        return sections


def _create_layer(config: DARKNET_CONFIG, num_inputs: List[int], **kwargs: Any) -> CREATE_LAYER_OUTPUT:
    """Calls one of the ``_create_<layertype>(config, num_inputs)`` functions to create a PyTorch module from the
    layer config.

    Args:
        config: Dictionary of configuration options for this layer.
        num_inputs: Number of channels in the input of every layer up to this layer.

    Returns:
        module (:class:`~torch.nn.Module`), num_outputs (int): The created PyTorch module and the number of channels in
        its output.
    """
    create_func: Dict[str, Callable[..., CREATE_LAYER_OUTPUT]] = {
        "convolutional": _create_convolutional,
        "maxpool": _create_maxpool,
        "route": _create_route,
        "shortcut": _create_shortcut,
        "upsample": _create_upsample,
        "yolo": _create_yolo,
    }
    return create_func[config["type"]](config, num_inputs, **kwargs)


def _create_convolutional(config: DARKNET_CONFIG, num_inputs: List[int], **kwargs: Any) -> CREATE_LAYER_OUTPUT:
    """Creates a convolutional layer.

    Args:
        config: Dictionary of configuration options for this layer.
        num_inputs: Number of channels in the input of every layer up to this layer.

    Returns:
        module (:class:`~torch.nn.Module`), num_outputs (int): The created PyTorch module and the number of channels in
        its output.
    """
    batch_normalize = config.get("batch_normalize", False)
    padding = (config["size"] - 1) // 2 if config["pad"] else 0

    layer = Conv(
        num_inputs[-1],
        config["filters"],
        kernel_size=config["size"],
        stride=config["stride"],
        padding=padding,
        bias=not batch_normalize,
        activation=config["activation"],
        norm="batchnorm" if batch_normalize else None,
    )
    return layer, config["filters"]


def _create_maxpool(config: DARKNET_CONFIG, num_inputs: List[int], **kwargs: Any) -> CREATE_LAYER_OUTPUT:
    """Creates a max pooling layer.

    Padding is added so that the output resolution will be the input resolution divided by stride, rounded upwards.

    Args:
        config: Dictionary of configuration options for this layer.
        num_inputs: Number of channels in the input of every layer up to this layer.

    Returns:
        module (:class:`~torch.nn.Module`), num_outputs (int): The created PyTorch module and the number of channels in
        its output.
    """
    layer = MaxPool(config["size"], config["stride"])
    return layer, num_inputs[-1]


def _create_route(config: DARKNET_CONFIG, num_inputs: List[int], **kwargs: Any) -> CREATE_LAYER_OUTPUT:
    """Creates a routing layer.

    A routing layer concatenates the output (or part of it) from the layers specified by the "layers" configuration
    option.

    Args:
        config: Dictionary of configuration options for this layer.
        num_inputs: Number of channels in the input of every layer up to this layer.

    Returns:
        module (:class:`~torch.nn.Module`), num_outputs (int): The created PyTorch module and the number of channels in
        its output.
    """
    num_chunks = config.get("groups", 1)
    chunk_idx = config.get("group_id", 0)

    # 0 is the first layer, -1 is the previous layer
    last = len(num_inputs) - 1
    source_layers = [layer if layer >= 0 else last + layer for layer in config["layers"]]

    layer = RouteLayer(source_layers, num_chunks, chunk_idx)

    # The number of outputs of a source layer is the number of inputs of the next layer.
    num_outputs = sum(num_inputs[layer + 1] // num_chunks for layer in source_layers)

    return layer, num_outputs


def _create_shortcut(config: DARKNET_CONFIG, num_inputs: List[int], **kwargs: Any) -> CREATE_LAYER_OUTPUT:
    """Creates a shortcut layer.

    A shortcut layer adds a residual connection from the layer specified by the "from" configuration option.

    Args:
        config: Dictionary of configuration options for this layer.
        num_inputs: Number of channels in the input of every layer up to this layer.

    Returns:
        module (:class:`~torch.nn.Module`), num_outputs (int): The created PyTorch module and the number of channels in
        its output.
    """
    layer = ShortcutLayer(config["from"])
    return layer, num_inputs[-1]


def _create_upsample(config: DARKNET_CONFIG, num_inputs: List[int], **kwargs: Any) -> CREATE_LAYER_OUTPUT:
    """Creates a layer that upsamples the data.

    Args:
        config: Dictionary of configuration options for this layer.
        num_inputs: Number of channels in the input of every layer up to this layer.

    Returns:
        module (:class:`~torch.nn.Module`), num_outputs (int): The created PyTorch module and the number of channels in
        its output.
    """
    layer = nn.Upsample(scale_factor=config["stride"], mode="nearest")
    return layer, num_inputs[-1]


def _create_yolo(
    config: DARKNET_CONFIG,
    num_inputs: List[int],
    prior_shapes: Optional[PRIOR_SHAPES] = None,
    matching_algorithm: Optional[str] = None,
    matching_threshold: Optional[float] = None,
    spatial_range: float = 5.0,
    size_range: float = 4.0,
    ignore_bg_threshold: Optional[float] = None,
    overlap_func: Optional[str] = None,
    predict_overlap: Optional[float] = None,
    label_smoothing: Optional[float] = None,
    overlap_loss_multiplier: Optional[float] = None,
    confidence_loss_multiplier: Optional[float] = None,
    class_loss_multiplier: Optional[float] = None,
    **kwargs: Any,
) -> CREATE_LAYER_OUTPUT:
    """Creates a YOLO detection layer.

    Args:
        config: Dictionary of configuration options for this layer.
        num_inputs: Number of channels in the input of every layer up to this layer. Not used by the detection layer.
        prior_shapes: A list of prior box dimensions, used for scaling the predicted dimensions and possibly for
            matching the targets to the anchors. The list should contain [width, height] pairs in the network input
            resolution. There should be `M x N` pairs, where `M` is the number of detection layers and `N` is the number
            of anchors per spatial location. They are assigned to the layers from the lowest (high-resolution) to the
            highest (low-resolution) layer, meaning that you typically want to sort the shapes from the smallest to the
            largest.
        matching_algorithm: Which algorithm to use for matching targets to anchors. "simota" (the SimOTA matching rule
            from YOLOX), "size" (match those prior shapes, whose width and height relative to the target is below given
            ratio), "iou" (match all prior shapes that give a high enough IoU), or "maxiou" (match the prior shape that
            gives the highest IoU, default).
        matching_threshold: Threshold for "size" and "iou" matching algorithms.
        spatial_range: The "simota" matching algorithm will restrict to anchors that are within an `N x N` grid cell
            area centered at the target, where `N` is the value of this parameter.
        size_range: The "simota" matching algorithm will restrict to anchors whose dimensions are no more than `N` and
            no less than `1/N` times the target dimensions, where `N` is the value of this parameter.
        ignore_bg_threshold: If a predictor is not responsible for predicting any target, but the corresponding anchor
            has IoU with some target greater than this threshold, the predictor will not be taken into account when
            calculating the confidence loss.
        overlap_func: Which function to use for calculating the IoU between two sets of boxes. Valid values are "iou",
            "giou", "diou", and "ciou".
        predict_overlap: Balance between binary confidence targets and predicting the overlap. 0.0 means that the target
            confidence is 1 if there's an object, and 1.0 means that the target confidence is the output of
            ``overlap_func``.
        label_smoothing: The epsilon parameter (weight) for class label smoothing. 0.0 means no smoothing (binary
            targets), and 1.0 means that the target probabilities are always 0.5.
        overlap_loss_multiplier: Overlap loss will be scaled by this value.
        confidence_loss_multiplier: Confidence loss will be scaled by this value.
        class_loss_multiplier: Classification loss will be scaled by this value.

    Returns:
        module (:class:`~torch.nn.Module`), num_outputs (int): The created PyTorch module and the number of channels in
        its output (always 0 for a detection layer).
    """
    if prior_shapes is None:
        # The "anchors" list alternates width and height.
        dims = config["anchors"]
        prior_shapes = [[dims[i], dims[i + 1]] for i in range(0, len(dims), 2)]
    if ignore_bg_threshold is None:
        ignore_bg_threshold = config.get("ignore_thresh", 1.0)
        assert isinstance(ignore_bg_threshold, float)
    if overlap_func is None:
        overlap_func = config.get("iou_loss", "iou")
        assert isinstance(overlap_func, str)
    if overlap_loss_multiplier is None:
        overlap_loss_multiplier = config.get("iou_normalizer", 1.0)
        assert isinstance(overlap_loss_multiplier, float)
    if confidence_loss_multiplier is None:
        confidence_loss_multiplier = config.get("obj_normalizer", 1.0)
        assert isinstance(confidence_loss_multiplier, float)
    if class_loss_multiplier is None:
        class_loss_multiplier = config.get("cls_normalizer", 1.0)
        assert isinstance(class_loss_multiplier, float)

    layer = create_detection_layer(
        num_classes=config["classes"],
        prior_shapes=prior_shapes,
        prior_shape_idxs=config["mask"],
        matching_algorithm=matching_algorithm,
        matching_threshold=matching_threshold,
        spatial_range=spatial_range,
        size_range=size_range,
        ignore_bg_threshold=ignore_bg_threshold,
        overlap_func=overlap_func,
        predict_overlap=predict_overlap,
        label_smoothing=label_smoothing,
        overlap_loss_multiplier=overlap_loss_multiplier,
        confidence_loss_multiplier=confidence_loss_multiplier,
        class_loss_multiplier=class_loss_multiplier,
        xy_scale=config.get("scale_x_y", 1.0),
        input_is_normalized=config.get("new_coords", 0) > 0,
    )
    return layer, 0
