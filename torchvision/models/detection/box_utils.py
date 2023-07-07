import torch
from torch import Tensor

from ...ops import box_iou


def aligned_iou(wh1: Tensor, wh2: Tensor) -> Tensor:
    """Calculates a matrix of intersections over union from box dimensions, assuming that the boxes are located at
    the same coordinates.

    Args:
        wh1: An ``[N, 2]`` matrix of box shapes (width and height).
        wh2: An ``[M, 2]`` matrix of box shapes (width and height).

    Returns:
        An ``[N, M]`` matrix of pairwise IoU values for every element in ``wh1`` and ``wh2``
    """
    area1 = wh1[:, 0] * wh1[:, 1]  # [N]
    area2 = wh2[:, 0] * wh2[:, 1]  # [M]

    inter_wh = torch.min(wh1[:, None, :], wh2)  # [N, M, 2]
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
        points: Point (x, y) coordinates, a tensor shaped ``[points, 2]``.
        boxes: Box (x1, y1, x2, y2) coordinates, a tensor shaped ``[boxes, 4]``.

    Returns:
        A tensor shaped ``[points, boxes]`` containing pairwise truth values of whether the points are inside the boxes.
    """
    lt = points[:, None, :] - boxes[None, :, :2]  # [boxes, points, 2]
    rb = boxes[None, :, 2:] - points[:, None, :]  # [boxes, points, 2]
    deltas = torch.cat((lt, rb), -1)  # [points, boxes, 4]
    return deltas.min(-1).values > 0.0  # [points, boxes]


def box_size_ratio(wh1: Tensor, wh2: Tensor) -> Tensor:
    """Compares the dimensions of the boxes pairwise.

    For each pair of boxes, calculates the largest ratio that can be obtained by dividing the widths with each other or
    dividing the heights with each other.

    Args:
        wh1: An ``[N, 2]`` matrix of box shapes (width and height).
        wh2: An ``[M, 2]`` matrix of box shapes (width and height).

    Returns:
        An ``[N, M]`` matrix of ratios of width or height dimensions, whichever is larger.
    """
    wh_ratio = wh1[:, None, :] / wh2[None, :, :]  # [M, N, 2]
    wh_ratio = torch.max(wh_ratio, 1.0 / wh_ratio)
    wh_ratio = wh_ratio.max(2).values  # [M, N]
    return wh_ratio
