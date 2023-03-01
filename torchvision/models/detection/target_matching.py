from abc import ABC, abstractmethod
from typing import Dict, List, Sequence, Tuple, Union

import torch
from torch import Tensor

from ...ops import box_convert, box_iou
from .anchor_utils import grid_centers
from .yolo_loss import YOLOLoss


def aligned_iou(dims1: Tensor, dims2: Tensor) -> Tensor:
    """Calculates a matrix of intersections over union from box dimensions, assuming that the boxes are located at
    the same coordinates.

    Args:
        dims1: Width and height of `N` boxes. Tensor of size ``[N, 2]``.
        dims2: Width and height of `M` boxes. Tensor of size ``[M, 2]``.

    Returns:
        Tensor of size ``[N, M]`` containing the pairwise IoU values for every element in ``dims1`` and ``dims2``
    """
    area1 = dims1[:, 0] * dims1[:, 1]  # [N]
    area2 = dims2[:, 0] * dims2[:, 1]  # [M]

    inter_wh = torch.min(dims1[:, None, :], dims2)  # [N, M, 2]
    inter = inter_wh[:, :, 0] * inter_wh[:, :, 1]  # [N, M]
    union = area1[:, None] + area2 - inter  # [N, M]

    return inter / union


def iou_below(pred_boxes: Tensor, target_boxes: Tensor, threshold: float) -> Tensor:
    """Creates a binary mask whose value will be ``True``, unless the predicted box overlaps any target
    significantly (IoU greater than ``threshold``).

    Args:
        pred_boxes: The predicted corner coordinates. Tensor of size ``[height, width, boxes_per_cell, 4]``.
        target_boxes: Corner coordinates of the target boxes. Tensor of size ``[height, width, boxes_per_cell, 4]``.

    Returns:
        A boolean tensor sized ``[height, width, boxes_per_cell]``, with ``False`` where the predicted box overlaps a
        target significantly and ``True`` elsewhere.
    """
    shape = pred_boxes.shape[:-1]
    pred_boxes = pred_boxes.view(-1, 4)
    ious = box_iou(pred_boxes, target_boxes)
    best_iou = ious.max(-1).values
    below_threshold = best_iou <= threshold
    return below_threshold.view(shape)


def is_inside_box(points: Tensor, boxes: Tensor) -> Tensor:
    """Get pairwise truth values of whether the point is inside the box.

    Args:
        points: point (x, y) coordinates, [points, 2]
        boxes: box (x1, y1, x2, y2) coordinates, [boxes, 4]

    Returns:
        A tensor shaped ``[boxes, points]`` containing pairwise truth values of whether the points are inside the boxes.
    """
    points = points.unsqueeze(0)  # [1, points, 2]
    boxes = boxes.unsqueeze(1)  # [boxes, 1, 4]
    lt = points - boxes[..., :2]  # [boxes, points, 2]
    rb = boxes[..., 2:] - points  # [boxes, points, 2]
    deltas = torch.cat((lt, rb), -1)  # [boxes, points, 4]
    return deltas.min(-1).values > 0.0  # [boxes, points]


class ShapeMatching(ABC):
    """Selects which anchors are used to predict each target, by comparing the shape of the target box to a set of
    prior shapes.

    Most YOLO variants match targets to anchors based on prior shapes that are assigned to the anchors in the model
    configuration. The subclasses of ``ShapeMatching`` implement matching rules that compare the width and height of
    the targets to each prior shape (regardless of the location where the target is). When the model includes multiple
    detection layers, different shapes are defined for each layer. Usually there are three detection layers and three
    prior shapes per layer.

    Args:
        ignore_bg_threshold: If a predictor is not responsible for predicting any target, but the prior shape has IoU
            with some target greater than this threshold, the predictor will not be taken into account when calculating
            the confidence loss.
    """

    def __init__(self, ignore_bg_threshold: float = 0.7) -> None:
        self.ignore_bg_threshold = ignore_bg_threshold

    def __call__(
        self,
        preds: Dict[str, Tensor],
        targets: Dict[str, Tensor],
        image_size: Tensor,
    ) -> Tuple[List[Tensor], Tensor, Tensor]:
        """For each target, selects predictions from the same grid cell, where the center of the target box is.

        Typically there are three predictions per grid cell. Subclasses implement ``match()``, which selects the
        predictions within the grid cell.

        Args:
            preds: Predictions for a single image.
            targets: Training targets for a single image.
            image_size: Input image width and height.

        Returns:
            The indices of the matched predictions, background mask, and a mask for selecting the matched targets.
        """
        height, width = preds["boxes"].shape[:2]
        device = preds["boxes"].device

        # A multiplier for scaling image coordinates to feature map coordinates
        grid_size = torch.tensor([width, height], device=device)
        image_to_grid = torch.true_divide(grid_size, image_size)

        # Bounding box center coordinates are converted to the feature map dimensions so that the whole number tells the
        # cell index and the fractional part tells the location inside the cell.
        xywh = box_convert(targets["boxes"], in_fmt="xyxy", out_fmt="cxcywh")
        grid_xy = xywh[:, :2] * image_to_grid
        cell_i = grid_xy[:, 0].to(torch.int64).clamp(0, width - 1)
        cell_j = grid_xy[:, 1].to(torch.int64).clamp(0, height - 1)

        target_selector, anchor_selector = self.match(xywh[:, 2:])
        cell_i = cell_i[target_selector]
        cell_j = cell_j[target_selector]

        # Background mask is used to select anchors that are not responsible for predicting any object, for
        # calculating the part of the confidence loss with zero as the target confidence. It is set to False, if a
        # predicted box overlaps any target significantly, or if a prediction is matched to a target.
        background_mask = iou_below(preds["boxes"], targets["boxes"], self.ignore_bg_threshold)
        background_mask[cell_j, cell_i, anchor_selector] = False

        pred_selector = [cell_j, cell_i, anchor_selector]

        return pred_selector, background_mask, target_selector

    @abstractmethod
    def match(self, wh: Tensor) -> Union[Tuple[Tensor, Tensor], Tensor]:
        """Selects anchors for each target based on the predicted shapes. The subclasses implement this method.

        Args:
            wh: A matrix of predicted width and height values.

        Returns:
            matched_targets, matched_anchors: Two vectors or a `2xN` matrix. The first vector is used to select the
            targets that this layer matched and the second one lists the matching anchors within the grid cell.
        """
        pass


class HighestIoUMatching(ShapeMatching):
    """For each target, select the prior shape that gives the highest IoU.

    This is the original YOLO matching rule.

    Args:
        prior_shapes: A list of all the prior box dimensions. The list should contain (width, height) tuples in the
            network input resolution.
        prior_shape_idxs: List of indices to ``prior_shapes`` that is used to select the (usually 3) prior shapes that
            this layer uses.
        ignore_bg_threshold: If a predictor is not responsible for predicting any target, but the prior shape has IoU
            with some target greater than this threshold, the predictor will not be taken into account when calculating
            the confidence loss.
    """

    def __init__(
        self, prior_shapes: Sequence[Tuple[int, int]], prior_shape_idxs: Sequence[int], ignore_bg_threshold: float = 0.7
    ) -> None:
        super().__init__(ignore_bg_threshold)
        self.prior_shapes = prior_shapes
        # anchor_map maps the anchor indices to anchors in this layer, or to -1 if it's not an anchor of this layer.
        # This layer ignores the target if all the selected anchors are in another layer.
        self.anchor_map = [
            prior_shape_idxs.index(idx) if idx in prior_shape_idxs else -1 for idx in range(len(prior_shapes))
        ]

    def match(self, wh: Tensor) -> Union[Tuple[Tensor, Tensor], Tensor]:
        prior_wh = torch.tensor(self.prior_shapes, dtype=wh.dtype, device=wh.device)
        anchor_map = torch.tensor(self.anchor_map, dtype=torch.int64, device=wh.device)

        ious = aligned_iou(wh, prior_wh)
        highest_iou_anchors = ious.max(1).indices
        highest_iou_anchors = anchor_map[highest_iou_anchors]
        matched_targets = highest_iou_anchors >= 0
        matched_anchors = highest_iou_anchors[matched_targets]
        return matched_targets, matched_anchors


class IoUThresholdMatching(ShapeMatching):
    """For each target, select all prior shapes that give a high enough IoU.

    Args:
        prior_shapes: A list of all the prior box dimensions. The list should contain (width, height) tuples in the
            network input resolution.
        prior_shape_idxs: List of indices to ``prior_shapes`` that is used to select the (usually 3) prior shapes that
            this layer uses.
        threshold: IoU treshold for matching.
        ignore_bg_threshold: If a predictor is not responsible for predicting any target, but the corresponding anchor
            has IoU with some target greater than this threshold, the predictor will not be taken into account when
            calculating the confidence loss.
    """

    def __init__(
        self,
        prior_shapes: Sequence[Tuple[int, int]],
        prior_shape_idxs: Sequence[int],
        threshold: float,
        ignore_bg_threshold: float = 0.7,
    ) -> None:
        super().__init__(ignore_bg_threshold)
        self.prior_shapes = [prior_shapes[idx] for idx in prior_shape_idxs]
        self.threshold = threshold

    def match(self, wh: Tensor) -> Union[Tuple[Tensor, Tensor], Tensor]:
        prior_wh = torch.tensor(self.prior_shapes, dtype=wh.dtype, device=wh.device)

        ious = aligned_iou(wh, prior_wh)
        above_threshold = (ious > self.threshold).nonzero()
        return above_threshold.T


class SizeRatioMatching(ShapeMatching):
    """For each target, select those prior shapes, whose width and height relative to the target is below given
    ratio.

    This is the matching rule used by Ultralytics YOLOv5 implementation.

    Args:
        prior_shapes: A list of all the prior box dimensions. The list should contain (width, height) tuples in the
            network input resolution.
        prior_shape_idxs: List of indices to ``prior_shapes`` that is used to select the (usually 3) prior shapes that
            this layer uses.
        threshold: Size ratio threshold for matching.
        ignore_bg_threshold: If a predictor is not responsible for predicting any target, but the corresponding anchor
            has IoU with some target greater than this threshold, the predictor will not be taken into account when
            calculating the confidence loss.
    """

    def __init__(
        self,
        prior_shapes: Sequence[Tuple[int, int]],
        prior_shape_idxs: Sequence[int],
        threshold: float,
        ignore_bg_threshold: float = 0.7,
    ) -> None:
        super().__init__(ignore_bg_threshold)
        self.prior_shapes = [prior_shapes[idx] for idx in prior_shape_idxs]
        self.threshold = threshold

    def match(self, wh: Tensor) -> Union[Tuple[Tensor, Tensor], Tensor]:
        prior_wh = torch.tensor(self.prior_shapes, dtype=wh.dtype, device=wh.device)

        wh_ratio = wh[:, None, :] / prior_wh[None, :, :]  # [num_targets, num_anchors, 2]
        wh_ratio = torch.max(wh_ratio, 1.0 / wh_ratio)
        wh_ratio = wh_ratio.max(2).values  # [num_targets, num_anchors]
        below_threshold = (wh_ratio < self.threshold).nonzero()
        return below_threshold.T


def _sim_ota_match(costs: Tensor, ious: Tensor) -> Tuple[Tensor, Tensor]:
    """Implements the SimOTA matching rule.

    The number of units supplied by each supplier (training target) needs to be decided in the Optimal Transport
    problem. "Dynamic k Estimation" uses the sum of the top 10 IoU values (casted to int) between the target and the
    predicted boxes.

    Args:
        costs: Sum of losses for (prediction, target) pairs: ``[targets, predictions]``
        ious: IoUs for (prediction, target) pairs: ``[targets, predictions]``

    Returns:
        A mask of predictions that were matched, and the indices of the matched targets. The latter contains as many
        elements as there are ``True`` values in the mask.
    """
    matching_matrix = torch.zeros_like(costs, dtype=torch.bool)

    if ious.numel() > 0:
        # For each target, define k as the sum of the 10 highest IoUs.
        top10_iou = torch.topk(ious, min(10, ious.shape[1])).values.sum(1)
        ks = torch.clip(top10_iou.int(), min=1)

        # For each target, select k predictions with lowest cost.
        for target_idx, (cost, k) in enumerate(zip(costs, ks)):
            prediction_idx = torch.topk(cost, k, largest=False).indices
            matching_matrix[target_idx, prediction_idx] = True

        # If there's more than one match for some prediction, match it with the best target. Now we consider all
        # targets, regardless of whether they were originally matched with the prediction or not.
        more_than_one_match = matching_matrix.sum(0) > 1
        best_targets = costs[:, more_than_one_match].argmin(0)
        matching_matrix[:, more_than_one_match] = False
        matching_matrix[best_targets, more_than_one_match] = True

    # For those predictions that were matched, get the index of the target.
    pred_mask = matching_matrix.sum(0) > 0
    target_selector = matching_matrix[:, pred_mask].int().argmax(0)
    return pred_mask, target_selector


class SimOTAMatching:
    """Selects which anchors are used to predict each target using the SimOTA matching rule.

    This is the matching rule used by YOLOX.

    Args:
        loss_func: A ``LossFunction`` object that can be used to calculate the pairwise costs.
        range: For each target, restrict to the anchors that are within an `N x N` grid cell are centered at the target,
            where `N` is the value of this parameter.
    """

    def __init__(self, loss_func: YOLOLoss, range: float = 5.0) -> None:
        self.loss_func = loss_func
        self.range = range

    def __call__(
        self,
        preds: Dict[str, Tensor],
        targets: Dict[str, Tensor],
        image_size: Tensor,
    ) -> Tuple[Tensor, Tensor, Tensor]:
        """For each target, selects predictions using the SimOTA matching rule.

        Args:
            preds: Predictions for a single image.
            targets: Training targets for a single image.
            image_size: Input image width and height.

        Returns:
            A mask of predictions that were matched, background mask (inverse of the first mask), and the indices of the
            matched targets. The last tensor contains as many elements as there are ``True`` values in the first mask.
        """
        height, width, boxes_per_cell, num_classes = preds["classprobs"].shape
        device = preds["boxes"].device

        # A multiplier for scaling feature map coordinates to image coordinates
        grid_size = torch.tensor([width, height], device=device)
        grid_to_image = torch.true_divide(image_size, grid_size)

        # Create a matrix for selecting the anchors that are inside the target bounding boxes.
        centers = grid_centers(grid_size).view(-1, 2) * grid_to_image
        inside_matrix = is_inside_box(centers, targets["boxes"])

        # Set the width and height of all target bounding boxes to self.range grid cells and create a matrix for
        # selecting the anchors that are now inside the boxes. If a small target has no anchors inside its bounding
        # box, it will be matched to one of these anchors, but a high penalty will ensure that anchors that are inside
        # the bounding box will be preferred.
        xywh = box_convert(targets["boxes"], in_fmt="xyxy", out_fmt="cxcywh")
        xy = xywh[:, :2]
        wh = self.range * grid_to_image * torch.ones_like(xy)
        xywh = torch.cat((xy, wh), -1)
        boxes = box_convert(xywh, in_fmt="cxcywh", out_fmt="xyxy")
        close_matrix = is_inside_box(centers, boxes)

        # In the first step we restrict ourselves to the grid cells whose center is inside or close enough to one or
        # more targets. The prediction grids are flattened and masked using a [height * width] boolean vector.
        mask = (inside_matrix | close_matrix).sum(0) > 0
        shape = (height * width, boxes_per_cell)
        fg_preds = {
            "boxes": preds["boxes"].view(*shape, 4)[mask].view(-1, 4),
            "confidences": preds["confidences"].view(shape)[mask].view(-1),
            "classprobs": preds["classprobs"].view(*shape, num_classes)[mask].view(-1, num_classes),
        }

        losses, ious = self.loss_func.pairwise(fg_preds, targets, input_is_normalized=False)
        costs = losses.overlap + losses.confidence + losses.classification
        costs += 100000.0 * ~inside_matrix[:, mask].repeat_interleave(boxes_per_cell, 1)
        pred_mask, target_selector = _sim_ota_match(costs, ious)

        # Add the anchor dimension to the mask and replace True values with the results of the actual SimOTA matching.
        mask = mask.view(height, width).unsqueeze(-1).repeat(1, 1, boxes_per_cell)
        mask[mask.nonzero().T.tolist()] = pred_mask

        background_mask = torch.logical_not(mask)

        return mask, background_mask, target_selector
