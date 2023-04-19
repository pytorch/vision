from typing import Dict, List, Tuple

import torch
from torch import Tensor

from ...ops import box_convert
from .anchor_utils import grid_centers
from .box_utils import aligned_iou, box_size_ratio, iou_below, is_inside_box
from .yolo_loss import YOLOLoss

PRIOR_SHAPES = List[List[int]]  # TorchScript doesn't allow a list of tuples.


def target_boxes_to_grid(preds: Tensor, targets: Tensor, image_size: Tensor) -> Tensor:
    """Scales target bounding boxes to feature map coordinates.

    It would be better to implement this in a super class, but TorchScript doesn't allow class inheritance.

    Args:
        preds: Predicted bounding boxes for a single image.
        targets: Target bounding boxes for a single image.
        image_size: Input image width and height.

    Returns:
        A tensor containing target x, y, width, and height in the feature map coordinates.
    """
    height, width = preds.shape[:2]

    # A multiplier for scaling image coordinates to feature map coordinates
    grid_size = torch.tensor([width, height], device=image_size.device)
    image_to_grid = torch.true_divide(grid_size, image_size)

    # Bounding box center coordinates are converted to the feature map dimensions so that the whole number tells the
    # cell index and the fractional part tells the location inside the cell.
    xywh = box_convert(targets, in_fmt="xyxy", out_fmt="cxcywh")
    grid_xy = xywh[:, :2] * image_to_grid
    cell_i = grid_xy[:, 0].to(torch.int64).clamp(0, width - 1)
    cell_j = grid_xy[:, 1].to(torch.int64).clamp(0, height - 1)

    return torch.cat((cell_i.unsqueeze(1), cell_j.unsqueeze(1), xywh[:, 2:]), 1)


class HighestIoUMatching:
    """For each target, select the prior shape that gives the highest IoU.

    This is the original YOLO matching rule.

    Args:
        prior_shapes: A list of all the prior box dimensions. The list should contain [width, height] pairs in the
            network input resolution.
        prior_shape_idxs: List of indices to ``prior_shapes`` that is used to select the (usually 3) prior shapes that
            this layer uses.
        ignore_bg_threshold: If a predictor is not responsible for predicting any target, but the prior shape has IoU
            with some target greater than this threshold, the predictor will not be taken into account when calculating
            the confidence loss.
    """

    def __init__(
        self, prior_shapes: PRIOR_SHAPES, prior_shape_idxs: List[int], ignore_bg_threshold: float = 0.7
    ) -> None:
        self.prior_shapes = prior_shapes
        # anchor_map maps the anchor indices to anchors in this layer, or to -1 if it's not an anchor of this layer.
        # This layer ignores the target if all the selected anchors are in another layer.
        self.anchor_map = [
            prior_shape_idxs.index(idx) if idx in prior_shape_idxs else -1 for idx in range(len(prior_shapes))
        ]
        self.ignore_bg_threshold = ignore_bg_threshold

    def match(self, wh: Tensor) -> Tuple[Tensor, Tensor]:
        """Selects anchors for each target based on the predicted shapes. The subclasses implement this method.

        Args:
            wh: A matrix of predicted width and height values.

        Returns:
            matched_targets, matched_anchors: Two vectors. The first vector is used to select the targets that this
            layer matched and the second one lists the matching anchors within the grid cell.
        """
        prior_wh = torch.tensor(self.prior_shapes, dtype=wh.dtype, device=wh.device)
        anchor_map = torch.tensor(self.anchor_map, dtype=torch.int64, device=wh.device)

        ious = aligned_iou(wh, prior_wh)
        highest_iou_anchors = ious.max(1).indices
        highest_iou_anchors = anchor_map[highest_iou_anchors]
        matched_targets = highest_iou_anchors >= 0
        matched_anchors = highest_iou_anchors[matched_targets]
        return matched_targets, matched_anchors

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
        scaled_targets = target_boxes_to_grid(preds["boxes"], targets["boxes"], image_size)
        target_selector, anchor_selector = self.match(scaled_targets[:, 2:])

        scaled_targets = scaled_targets[target_selector]
        cell_i = scaled_targets[:, 0]
        cell_j = scaled_targets[:, 1]

        # Background mask is used to select anchors that are not responsible for predicting any object, for
        # calculating the part of the confidence loss with zero as the target confidence. It is set to False, if a
        # predicted box overlaps any target significantly, or if a prediction is matched to a target.
        background_mask = iou_below(preds["boxes"], targets["boxes"], self.ignore_bg_threshold)
        background_mask[cell_j, cell_i, anchor_selector] = False

        pred_selector = [cell_j, cell_i, anchor_selector]

        return pred_selector, background_mask, target_selector


class IoUThresholdMatching:
    """For each target, select all prior shapes that give a high enough IoU.

    Args:
        prior_shapes: A list of all the prior box dimensions. The list should contain [width, height] pairs in the
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
        prior_shapes: PRIOR_SHAPES,
        prior_shape_idxs: List[int],
        threshold: float,
        ignore_bg_threshold: float = 0.7,
    ) -> None:
        self.prior_shapes = [prior_shapes[idx] for idx in prior_shape_idxs]
        self.threshold = threshold
        self.ignore_bg_threshold = ignore_bg_threshold

    def match(self, wh: Tensor) -> Tuple[Tensor, Tensor]:
        """Selects anchors for each target based on the predicted shapes. The subclasses implement this method.

        Args:
            wh: A matrix of predicted width and height values.

        Returns:
            matched_targets, matched_anchors: Two vectors. The first vector is used to select the targets that this
            layer matched and the second one lists the matching anchors within the grid cell.
        """
        prior_wh = torch.tensor(self.prior_shapes, dtype=wh.dtype, device=wh.device)

        ious = aligned_iou(wh, prior_wh)
        above_threshold = (ious > self.threshold).nonzero()
        return above_threshold[:, 0], above_threshold[:, 1]

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
        scaled_targets = target_boxes_to_grid(preds["boxes"], targets["boxes"], image_size)
        target_selector, anchor_selector = self.match(scaled_targets[:, 2:])

        scaled_targets = scaled_targets[target_selector]
        cell_i = scaled_targets[:, 0]
        cell_j = scaled_targets[:, 1]

        # Background mask is used to select anchors that are not responsible for predicting any object, for
        # calculating the part of the confidence loss with zero as the target confidence. It is set to False, if a
        # predicted box overlaps any target significantly, or if a prediction is matched to a target.
        background_mask = iou_below(preds["boxes"], targets["boxes"], self.ignore_bg_threshold)
        background_mask[cell_j, cell_i, anchor_selector] = False

        pred_selector = [cell_j, cell_i, anchor_selector]

        return pred_selector, background_mask, target_selector


class SizeRatioMatching:
    """For each target, select those prior shapes, whose width and height relative to the target is below given
    ratio.

    This is the matching rule used by Ultralytics YOLOv5 implementation.

    Args:
        prior_shapes: A list of all the prior box dimensions. The list should contain [width, height] pairs in the
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
        prior_shapes: PRIOR_SHAPES,
        prior_shape_idxs: List[int],
        threshold: float,
        ignore_bg_threshold: float = 0.7,
    ) -> None:
        self.prior_shapes = [prior_shapes[idx] for idx in prior_shape_idxs]
        self.threshold = threshold
        self.ignore_bg_threshold = ignore_bg_threshold

    def match(self, wh: Tensor) -> Tuple[Tensor, Tensor]:
        """Selects anchors for each target based on the predicted shapes. The subclasses implement this method.

        Args:
            wh: A matrix of predicted width and height values.

        Returns:
            matched_targets, matched_anchors: Two vectors. The first vector is used to select the targets that this
            layer matched and the second one lists the matching anchors within the grid cell.
        """
        prior_wh = torch.tensor(self.prior_shapes, dtype=wh.dtype, device=wh.device)
        below_threshold = (box_size_ratio(wh, prior_wh) < self.threshold).nonzero()
        return below_threshold[:, 0], below_threshold[:, 1]

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
        scaled_targets = target_boxes_to_grid(preds["boxes"], targets["boxes"], image_size)
        target_selector, anchor_selector = self.match(scaled_targets[:, 2:])

        scaled_targets = scaled_targets[target_selector]
        cell_i = scaled_targets[:, 0]
        cell_j = scaled_targets[:, 1]

        # Background mask is used to select anchors that are not responsible for predicting any object, for
        # calculating the part of the confidence loss with zero as the target confidence. It is set to False, if a
        # predicted box overlaps any target significantly, or if a prediction is matched to a target.
        background_mask = iou_below(preds["boxes"], targets["boxes"], self.ignore_bg_threshold)
        background_mask[cell_j, cell_i, anchor_selector] = False

        pred_selector = [cell_j, cell_i, anchor_selector]

        return pred_selector, background_mask, target_selector


def _sim_ota_match(costs: Tensor, ious: Tensor) -> Tuple[Tensor, Tensor]:
    """Implements the SimOTA matching rule.

    The number of units supplied by each supplier (training target) needs to be decided in the Optimal Transport
    problem. "Dynamic k Estimation" uses the sum of the top 10 IoU values (casted to int) between the target and the
    predicted boxes.

    Args:
        costs: A ``[predictions, targets]`` matrix of losses.
        ious: A ``[predictions, targets]`` matrix of IoUs.

    Returns:
        A mask of predictions that were matched, and the indices of the matched targets. The latter contains as many
        elements as there are ``True`` values in the mask.
    """
    num_preds, num_targets = ious.shape

    matching_matrix = torch.zeros_like(costs, dtype=torch.bool)

    if ious.numel() > 0:
        # For each target, define k as the sum of the 10 highest IoUs.
        top10_iou = torch.topk(ious, min(10, num_preds), dim=0).values.sum(0)
        ks = torch.clip(top10_iou.int(), min=1)
        assert len(ks) == num_targets

        # For each target, select k predictions with the lowest cost.
        for target_idx, (target_costs, k) in enumerate(zip(costs.T, ks)):
            pred_idx = torch.topk(target_costs, k, largest=False).indices
            matching_matrix[pred_idx, target_idx] = True

        # If there's more than one match for some prediction, match it with the best target. Now we consider all
        # targets, regardless of whether they were originally matched with the prediction or not.
        more_than_one_match = matching_matrix.sum(1) > 1
        best_targets = costs[more_than_one_match, :].argmin(1)
        matching_matrix[more_than_one_match, :] = False
        matching_matrix[more_than_one_match, best_targets] = True

    # For those predictions that were matched, get the index of the target.
    pred_mask = matching_matrix.sum(1) > 0
    target_selector = matching_matrix[pred_mask, :].int().argmax(1)
    return pred_mask, target_selector


class SimOTAMatching:
    """Selects which anchors are used to predict each target using the SimOTA matching rule.

    This is the matching rule used by YOLOX.

    Args:
        prior_shapes: A list of all the prior box dimensions. The list should contain [width, height] pairs in the
            network input resolution.
        prior_shape_idxs: List of indices to ``prior_shapes`` that is used to select the (usually 3) prior shapes that
            this layer uses.
        loss_func: A ``YOLOLoss`` object that can be used to calculate the pairwise costs.
        spatial_range: For each target, restrict to the anchors that are within an `N × N` grid cell are centered at the
            target, where `N` is the value of this parameter.
        size_range: For each target, restrict to the anchors whose prior dimensions are not larger than the target
            dimensions multiplied by this value and not smaller than the target dimensions divided by this value.
    """

    def __init__(
        self,
        prior_shapes: PRIOR_SHAPES,
        prior_shape_idxs: List[int],
        loss_func: YOLOLoss,
        spatial_range: float,
        size_range: float,
    ) -> None:
        self.prior_shapes = [prior_shapes[idx] for idx in prior_shape_idxs]
        self.loss_func = loss_func
        self.spatial_range = spatial_range
        self.size_range = size_range

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
        height, width, boxes_per_cell, _ = preds["boxes"].shape
        prior_mask, anchor_inside_target = self._get_prior_mask(targets, image_size, width, height, boxes_per_cell)
        prior_preds = {
            "boxes": preds["boxes"][prior_mask],
            "confidences": preds["confidences"][prior_mask],
            "classprobs": preds["classprobs"][prior_mask],
        }

        losses, ious = self.loss_func.pairwise(prior_preds, targets, input_is_normalized=False)
        costs = losses.overlap + losses.confidence + losses.classification
        costs += 100000.0 * ~anchor_inside_target
        pred_mask, target_selector = _sim_ota_match(costs, ious)

        # Add the anchor dimension to the mask and replace True values with the results of the actual SimOTA matching.
        pred_selector = prior_mask.nonzero().T.tolist()
        prior_mask[pred_selector] = pred_mask

        background_mask = torch.logical_not(prior_mask)

        return prior_mask, background_mask, target_selector

    def _get_prior_mask(
        self,
        targets: Dict[str, Tensor],
        image_size: Tensor,
        grid_width: int,
        grid_height: int,
        boxes_per_cell: int,
    ) -> Tuple[Tensor, Tensor]:
        """Creates a mask for selecting the "center prior" anchors.

        In the first step we restrict ourselves to the grid cells whose center is inside or close enough to one or more
        targets.

        Args:
            targets: Training targets for a single image.
            image_size: Input image width and height.
            grid_width: Width of the feature grid.
            grid_height: Height of the feature grid.
            boxes_per_cell: Number of boxes that will be predicted per feature grid cell.

        Returns:
            Two masks, a ``[grid_height, grid_width, boxes_per_cell]`` mask for selecting anchors that are close and
            similar in shape to a target, and an ``[anchors, targets]`` matrix that indicates which targets are inside
            those anchors.
        """
        # A multiplier for scaling feature map coordinates to image coordinates
        grid_size = torch.tensor([grid_width, grid_height], device=targets["boxes"].device)
        grid_to_image = torch.true_divide(image_size, grid_size)

        # Get target center coordinates and dimensions.
        xywh = box_convert(targets["boxes"], in_fmt="xyxy", out_fmt="cxcywh")
        xy = xywh[:, :2]
        wh = xywh[:, 2:]

        # Create a [boxes_per_cell, targets] tensor for selecting prior shapes that are close enough to the target
        # dimensions.
        prior_wh = torch.tensor(self.prior_shapes, device=targets["boxes"].device)
        shape_selector = box_size_ratio(prior_wh, wh) < self.size_range

        # Create a [grid_cells, targets] tensor for selecting spatial locations that are inside target bounding boxes.
        centers = grid_centers(grid_size).view(-1, 2) * grid_to_image
        inside_selector = is_inside_box(centers, targets["boxes"])

        # Combine the above selectors into a [grid_cells, boxes_per_cell, targets] tensor for selecting anchors that are
        # inside target bounding boxes and close enough shape.
        inside_selector = inside_selector[:, None, :].repeat(1, boxes_per_cell, 1)
        inside_selector = torch.logical_and(inside_selector, shape_selector)

        # Set the width and height of all target bounding boxes to self.range grid cells and create a selector for
        # anchors that are now inside the boxes. If a small target has no anchors inside its bounding box, it will be
        # matched to one of these anchors, but a high penalty will ensure that anchors that are inside the bounding box
        # will be preferred.
        wh = self.spatial_range * grid_to_image * torch.ones_like(xy)
        xywh = torch.cat((xy, wh), -1)
        boxes = box_convert(xywh, in_fmt="cxcywh", out_fmt="xyxy")
        close_selector = is_inside_box(centers, boxes)

        # Create a [grid_cells, boxes_per_cell, targets] tensor for selecting anchors that are spatially close to a
        # target and whose shape is close enough to the target.
        close_selector = close_selector[:, None, :].repeat(1, boxes_per_cell, 1)
        close_selector = torch.logical_and(close_selector, shape_selector)

        mask = torch.logical_or(inside_selector, close_selector).sum(-1) > 0
        mask = mask.view(grid_height, grid_width, boxes_per_cell)
        inside_selector = inside_selector.view(grid_height, grid_width, boxes_per_cell, -1)
        return mask, inside_selector[mask]
