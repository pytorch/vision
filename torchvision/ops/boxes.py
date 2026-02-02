import torch
import torchvision
from torch import Tensor
from torchvision.extension import _assert_has_ops, _has_ops

from ..utils import _log_api_usage_once
from ._box_convert import (
    _box_cxcywh_to_xyxy,
    _box_cxcywhr_to_xywhr,
    _box_xywh_to_xyxy,
    _box_xywhr_to_cxcywhr,
    _box_xywhr_to_xyxyxyxy,
    _box_xyxy_to_cxcywh,
    _box_xyxy_to_xywh,
    _box_xyxyxyxy_to_xywhr,
)
from ._utils import _upcast


def nms(boxes: Tensor, scores: Tensor, iou_threshold: float) -> Tensor:
    """
    Performs non-maximum suppression (NMS) on the boxes according
    to their intersection-over-union (IoU).

    NMS iteratively removes lower scoring boxes which have an
    IoU greater than ``iou_threshold`` with another (higher scoring)
    box.

    If multiple boxes have the exact same score and satisfy the IoU
    criterion with respect to a reference box, the selected box is
    not guaranteed to be the same between CPU and GPU. This is similar
    to the behavior of argsort in PyTorch when repeated values are present.

    Args:
        boxes (Tensor[N, 4])): boxes to perform NMS on. They
            are expected to be in ``(x1, y1, x2, y2)`` format with ``0 <= x1 < x2`` and
            ``0 <= y1 < y2``.
        scores (Tensor[N]): scores for each one of the boxes
        iou_threshold (float): discards all overlapping boxes with IoU > iou_threshold

    Returns:
        Tensor: int64 tensor with the indices of the elements that have been kept
        by NMS, sorted in decreasing order of scores
    """
    if not torch.jit.is_scripting() and not torch.jit.is_tracing():
        _log_api_usage_once(nms)
    _assert_has_ops()
    return torch.ops.torchvision.nms(boxes, scores, iou_threshold)


def batched_nms(
    boxes: Tensor,
    scores: Tensor,
    idxs: Tensor,
    iou_threshold: float,
) -> Tensor:
    """
    Performs non-maximum suppression in a batched fashion.

    Each index value correspond to a category, and NMS
    will not be applied between elements of different categories.

    Args:
        boxes (Tensor[N, 4]): boxes where NMS will be performed. They
            are expected to be in ``(x1, y1, x2, y2)`` format with ``0 <= x1 < x2`` and
            ``0 <= y1 < y2``.
        scores (Tensor[N]): scores for each one of the boxes
        idxs (Tensor[N]): indices of the categories for each one of the boxes.
        iou_threshold (float): discards all overlapping boxes with IoU > iou_threshold

    Returns:
        Tensor: int64 tensor with the indices of the elements that have been kept by NMS, sorted
        in decreasing order of scores
    """
    if not torch.jit.is_scripting() and not torch.jit.is_tracing():
        _log_api_usage_once(batched_nms)
    # Benchmarks that drove the following thresholds are at
    # https://github.com/pytorch/vision/issues/1311#issuecomment-781329339
    # and https://github.com/pytorch/vision/pull/8925
    if boxes.numel() > (4000 if boxes.device.type == "cpu" else 100_000) and not torchvision._is_tracing():
        return _batched_nms_vanilla(boxes, scores, idxs, iou_threshold)
    else:
        return _batched_nms_coordinate_trick(boxes, scores, idxs, iou_threshold)


@torch.jit._script_if_tracing
def _batched_nms_coordinate_trick(
    boxes: Tensor,
    scores: Tensor,
    idxs: Tensor,
    iou_threshold: float,
) -> Tensor:
    # strategy: in order to perform NMS independently per class,
    # we add an offset to all the boxes. The offset is dependent
    # only on the class idx, and is large enough so that boxes
    # from different classes do not overlap
    if boxes.numel() == 0:
        return torch.empty((0,), dtype=torch.int64, device=boxes.device)
    max_coordinate = boxes.max()
    offsets = idxs.to(boxes) * (max_coordinate + torch.tensor(1).to(boxes))
    boxes_for_nms = boxes + offsets[:, None]
    keep = nms(boxes_for_nms, scores, iou_threshold)
    return keep


@torch.jit._script_if_tracing
def _batched_nms_vanilla(
    boxes: Tensor,
    scores: Tensor,
    idxs: Tensor,
    iou_threshold: float,
) -> Tensor:
    # Based on Detectron2 implementation, just manually call nms() on each class independently
    keep_mask = torch.zeros_like(scores, dtype=torch.bool)
    for class_id in torch.unique(idxs):
        curr_indices = torch.where(idxs == class_id)[0]
        curr_keep_indices = nms(boxes[curr_indices], scores[curr_indices], iou_threshold)
        keep_mask[curr_indices[curr_keep_indices]] = True
    keep_indices = torch.where(keep_mask)[0]
    return keep_indices[scores[keep_indices].sort(descending=True)[1]]


def remove_small_boxes(boxes: Tensor, min_size: float) -> Tensor:
    """
    Remove every box from ``boxes`` which contains at least one side length
    that is smaller than ``min_size``.

    .. note::
        For sanitizing a :class:`~torchvision.tv_tensors.BoundingBoxes` object, consider using
        the transform :func:`~torchvision.transforms.v2.SanitizeBoundingBoxes` instead.

    Args:
        boxes (Tensor[..., 4]): boxes in ``(x1, y1, x2, y2)`` format
            with ``0 <= x1 < x2`` and ``0 <= y1 < y2``.
        min_size (float): minimum size

    Returns:
        Tensor[K]: indices of the boxes that have both sides
        larger than ``min_size``
    """
    if not torch.jit.is_scripting() and not torch.jit.is_tracing():
        _log_api_usage_once(remove_small_boxes)
    ws, hs = boxes[..., 2] - boxes[..., 0], boxes[..., 3] - boxes[..., 1]
    keep = (ws >= min_size) & (hs >= min_size)
    keep = torch.where(keep)[0]
    return keep


def clip_boxes_to_image(boxes: Tensor, size: tuple[int, int]) -> Tensor:
    """
    Clip boxes so that they lie inside an image of size ``size``.

    .. note::
        For clipping a :class:`~torchvision.tv_tensors.BoundingBoxes` object, consider using
        the transform :func:`~torchvision.transforms.v2.ClampBoundingBoxes` instead.

    Args:
        boxes (Tensor[..., 4]): boxes in ``(x1, y1, x2, y2)`` format
            with ``0 <= x1 < x2`` and ``0 <= y1 < y2``.
        size (Tuple[height, width]): size of the image

    Returns:
        Tensor[..., 4]: clipped boxes
    """
    if not torch.jit.is_scripting() and not torch.jit.is_tracing():
        _log_api_usage_once(clip_boxes_to_image)
    dim = boxes.dim()
    boxes_x = boxes[..., 0::2]
    boxes_y = boxes[..., 1::2]
    height, width = size

    if torchvision._is_tracing():
        boxes_x = torch.max(boxes_x, torch.tensor(0, dtype=boxes.dtype, device=boxes.device))
        boxes_x = torch.min(boxes_x, torch.tensor(width, dtype=boxes.dtype, device=boxes.device))
        boxes_y = torch.max(boxes_y, torch.tensor(0, dtype=boxes.dtype, device=boxes.device))
        boxes_y = torch.min(boxes_y, torch.tensor(height, dtype=boxes.dtype, device=boxes.device))
    else:
        boxes_x = boxes_x.clamp(min=0, max=width)
        boxes_y = boxes_y.clamp(min=0, max=height)

    clipped_boxes = torch.stack((boxes_x, boxes_y), dim=dim)
    return clipped_boxes.reshape(boxes.shape)


def box_convert(boxes: Tensor, in_fmt: str, out_fmt: str) -> Tensor:
    """
    Converts :class:`torch.Tensor` boxes from a given ``in_fmt`` to ``out_fmt``.

    .. note::
        For converting a :class:`torch.Tensor` or a :class:`~torchvision.tv_tensors.BoundingBoxes` object
        between different formats,
        consider using :func:`~torchvision.transforms.v2.functional.convert_bounding_box_format` instead.
        Or see the corresponding transform :func:`~torchvision.transforms.v2.ConvertBoundingBoxFormat`.

    Supported ``in_fmt`` and ``out_fmt`` strings are:

    ``'xyxy'``: boxes are represented via corners, x1, y1 being top left and x2, y2 being bottom right.
    This is the format that torchvision utilities expect.

    ``'xywh'``: boxes are represented via corner, width and height, x1, y2 being top left, w, h being width and height.

    ``'cxcywh'``: boxes are represented via centre, width and height, cx, cy being center of box, w, h
    being width and height.

    ``'xywhr'``: boxes are represented via corner, width and height, x1, y2 being top left, w, h being width and height.
    r is rotation angle w.r.t to the box center by :math:`|r|` degrees counter clock wise in the image plan

    ``'cxcywhr'``: boxes are represented via centre, width and height, cx, cy being center of box, w, h
    being width and height.
    r is rotation angle w.r.t to the box center by :math:`|r|` degrees counter clock wise in the image plan

    ``'xyxyxyxy'``: boxes are represented via corners, x1, y1 being top left, x2, y2 top right,
    x3, y3 bottom right, and x4, y4 bottom left.

    Args:
        boxes (Tensor[..., K]): boxes which will be converted. K is the number of coordinates (4 for unrotated bounding boxes, 5 or 8 for rotated bounding boxes). Supports any number of leading batch dimensions.
        in_fmt (str): Input format of given boxes. Supported formats are ['xyxy', 'xywh', 'cxcywh', 'xywhr', 'cxcywhr', 'xyxyxyxy'].
        out_fmt (str): Output format of given boxes. Supported formats are ['xyxy', 'xywh', 'cxcywh', 'xywhr', 'cxcywhr', 'xyxyxyxy']

    Returns:
        Tensor[..., K]: Boxes into converted format.
    """
    if not torch.jit.is_scripting() and not torch.jit.is_tracing():
        _log_api_usage_once(box_convert)
    allowed_fmts = (
        "xyxy",
        "xywh",
        "cxcywh",
        "xywhr",
        "cxcywhr",
        "xyxyxyxy",
    )
    if in_fmt not in allowed_fmts or out_fmt not in allowed_fmts:
        raise ValueError(f"Unsupported Bounding Box Conversions for given in_fmt {in_fmt} and out_fmt {out_fmt}")

    if in_fmt == out_fmt:
        return boxes.clone()
    e = (in_fmt, out_fmt)
    if e == ("xywh", "xyxy"):
        boxes = _box_xywh_to_xyxy(boxes)
    elif e == ("cxcywh", "xyxy"):
        boxes = _box_cxcywh_to_xyxy(boxes)
    elif e == ("xyxy", "xywh"):
        boxes = _box_xyxy_to_xywh(boxes)
    elif e == ("xyxy", "cxcywh"):
        boxes = _box_xyxy_to_cxcywh(boxes)
    elif e == ("xywh", "cxcywh"):
        boxes = _box_xywh_to_xyxy(boxes)
        boxes = _box_xyxy_to_cxcywh(boxes)
    elif e == ("cxcywh", "xywh"):
        boxes = _box_cxcywh_to_xyxy(boxes)
        boxes = _box_xyxy_to_xywh(boxes)
    elif e == ("cxcywhr", "xywhr"):
        boxes = _box_cxcywhr_to_xywhr(boxes)
    elif e == ("xywhr", "cxcywhr"):
        boxes = _box_xywhr_to_cxcywhr(boxes)
    elif e == ("cxcywhr", "xyxyxyxy"):
        boxes = _box_cxcywhr_to_xywhr(boxes).to(boxes.dtype)
        boxes = _box_xywhr_to_xyxyxyxy(boxes)
    elif e == ("xyxyxyxy", "cxcywhr"):
        boxes = _box_xyxyxyxy_to_xywhr(boxes).to(boxes.dtype)
        boxes = _box_xywhr_to_cxcywhr(boxes)
    elif e == ("xywhr", "xyxyxyxy"):
        boxes = _box_xywhr_to_xyxyxyxy(boxes)
    elif e == ("xyxyxyxy", "xywhr"):
        boxes = _box_xyxyxyxy_to_xywhr(boxes)
    else:
        raise NotImplementedError(f"Unsupported Bounding Box Conversions for given in_fmt {e[0]} and out_fmt {e[1]}")

    return boxes


def box_area(boxes: Tensor, fmt: str = "xyxy") -> Tensor:
    """
    Computes the area of a set of bounding boxes from a given format.

    Args:
        boxes (Tensor[..., 4]): boxes for which the area will be computed.
        fmt (str): Format of the input boxes.
            Default is "xyxy" to preserve backward compatibility.
            Supported formats are "xyxy", "xywh", and "cxcywh".

    Returns:
        Tensor[N]: Tensor containing the area for each box.
    """
    if not torch.jit.is_scripting() and not torch.jit.is_tracing():
        _log_api_usage_once(box_area)
    allowed_fmts = (
        "xyxy",
        "xywh",
        "cxcywh",
    )
    if fmt not in allowed_fmts:
        raise ValueError(f"Unsupported Bounding Box area for given format {fmt}")
    boxes = _upcast(boxes)
    if fmt == "xyxy":
        area = (boxes[..., 2] - boxes[..., 0]) * (boxes[..., 3] - boxes[..., 1])
    else:
        # For formats with width and height, area = width * height
        # Supported: cxcywh, xywh
        area = boxes[..., 2] * boxes[..., 3]

    return area


# implementation from https://github.com/kuangliu/torchcv/blob/master/torchcv/utils/box.py
# with slight modifications
def _box_inter_union(boxes1: Tensor, boxes2: Tensor, fmt: str = "xyxy") -> tuple[Tensor, Tensor]:
    area1 = box_area(boxes1, fmt=fmt)
    area2 = box_area(boxes2, fmt=fmt)

    allowed_fmts = (
        "xyxy",
        "xywh",
        "cxcywh",
    )
    if fmt not in allowed_fmts:
        raise ValueError(f"Unsupported Box IoU Calculation for given fmt {fmt}.")

    if fmt == "xyxy":
        lt = torch.max(boxes1[..., None, :2], boxes2[..., None, :, :2])  # [...,N,M,2]
        rb = torch.min(boxes1[..., None, 2:], boxes2[..., None, :, 2:])  # [...,N,M,2]
    elif fmt == "xywh":
        lt = torch.max(boxes1[..., None, :2], boxes2[..., None, :, :2])  # [...,N,M,2]
        rb = torch.min(
            boxes1[..., None, :2] + boxes1[..., None, 2:], boxes2[..., None, :, :2] + boxes2[..., None, :, 2:]
        )  # [...,N,M,2]
    else:  # fmt == "cxcywh":
        lt = torch.max(
            boxes1[..., None, :2] - boxes1[..., None, 2:] / 2, boxes2[..., None, :, :2] - boxes2[..., None, :, 2:] / 2
        )  # [N,M,2]
        rb = torch.min(
            boxes1[..., None, :2] + boxes1[..., None, 2:] / 2, boxes2[..., None, :, :2] + boxes2[..., None, :, 2:] / 2
        )  # [N,M,2]

    wh = _upcast(rb - lt).clamp(min=0)  # [N,M,2]
    inter = wh[..., 0] * wh[..., 1]  # [N,M]

    union = area1[..., None] + area2[..., None, :] - inter

    return inter, union


def box_iou(boxes1: Tensor, boxes2: Tensor, fmt: str = "xyxy") -> Tensor:
    """
    Return intersection-over-union (Jaccard index) between two sets of boxes from a given format.

    Args:
        boxes1 (Tensor[..., N, 4]): first set of boxes
        boxes2 (Tensor[..., M, 4]): second set of boxes
        fmt (str): Format of the input boxes.
            Default is "xyxy" to preserve backward compatibility.
            Supported formats are "xyxy", "xywh", and "cxcywh".

    Returns:
        Tensor[..., N, M]: the NxM matrix containing the pairwise IoU values for every element
        in boxes1 and boxes2
    """
    if not torch.jit.is_scripting() and not torch.jit.is_tracing():
        _log_api_usage_once(box_iou)
    allowed_fmts = (
        "xyxy",
        "xywh",
        "cxcywh",
    )
    if fmt not in allowed_fmts:
        raise ValueError(f"Unsupported Box IoU Calculation for given format {fmt}.")
    inter, union = _box_inter_union(boxes1, boxes2, fmt=fmt)
    iou = inter / union
    return iou


# Implementation adapted from https://github.com/facebookresearch/detr/blob/master/util/box_ops.py
def generalized_box_iou(boxes1: Tensor, boxes2: Tensor) -> Tensor:
    """
    Return generalized intersection-over-union (Jaccard index) between two sets of boxes.

    Both sets of boxes are expected to be in ``(x1, y1, x2, y2)`` format with
    ``0 <= x1 < x2`` and ``0 <= y1 < y2``.

    Args:
        boxes1 (Tensor[..., N, 4]): first set of boxes
        boxes2 (Tensor[..., M, 4]): second set of boxes

    Returns:
        Tensor[..., N, M]: the NxM matrix containing the pairwise generalized IoU values
        for every element in boxes1 and boxes2
    """
    if not torch.jit.is_scripting() and not torch.jit.is_tracing():
        _log_api_usage_once(generalized_box_iou)

    inter, union = _box_inter_union(boxes1, boxes2)
    iou = inter / union

    lti = torch.min(boxes1[..., None, :2], boxes2[..., None, :, :2])
    rbi = torch.max(boxes1[..., None, 2:], boxes2[..., None, :, 2:])

    whi = _upcast(rbi - lti).clamp(min=0)  # [N,M,2]
    areai = whi[..., 0] * whi[..., 1]

    return iou - (areai - union) / areai


def complete_box_iou(boxes1: Tensor, boxes2: Tensor, eps: float = 1e-7) -> Tensor:
    """
    Return complete intersection-over-union (Jaccard index) between two sets of boxes.
    Both sets of boxes are expected to be in ``(x1, y1, x2, y2)`` format with
    ``0 <= x1 < x2`` and ``0 <= y1 < y2``.
    Args:
        boxes1 (Tensor[..., N, 4]): first set of boxes
        boxes2 (Tensor[..., M, 4]): second set of boxes
        eps (float, optional): small number to prevent division by zero. Default: 1e-7
    Returns:
        Tensor[..., N, M]: the NxM matrix containing the pairwise complete IoU values
        for every element in boxes1 and boxes2
    """
    if not torch.jit.is_scripting() and not torch.jit.is_tracing():
        _log_api_usage_once(complete_box_iou)

    boxes1 = _upcast(boxes1)
    boxes2 = _upcast(boxes2)

    diou, iou = _box_diou_iou(boxes1, boxes2, eps)

    w_pred = boxes1[..., None, 2] - boxes1[..., None, 0]
    h_pred = boxes1[..., None, 3] - boxes1[..., None, 1]

    w_gt = boxes2[..., None, :, 2] - boxes2[..., None, :, 0]
    h_gt = boxes2[..., None, :, 3] - boxes2[..., None, :, 1]

    v = (4 / (torch.pi**2)) * torch.pow(torch.atan(w_pred / h_pred) - torch.atan(w_gt / h_gt), 2)
    with torch.no_grad():
        alpha = v / (1 - iou + v + eps)
    return diou - alpha * v


def distance_box_iou(boxes1: Tensor, boxes2: Tensor, eps: float = 1e-7) -> Tensor:
    """
    Return distance intersection-over-union (Jaccard index) between two sets of boxes.

    Both sets of boxes are expected to be in ``(x1, y1, x2, y2)`` format with
    ``0 <= x1 < x2`` and ``0 <= y1 < y2``.

    Args:
        boxes1 (Tensor[..., N, 4]): first set of boxes
        boxes2 (Tensor[..., M, 4]): second set of boxes
        eps (float, optional): small number to prevent division by zero. Default: 1e-7

    Returns:
        Tensor[..., N, M]: the NxM matrix containing the pairwise distance IoU values
        for every element in boxes1 and boxes2
    """
    if not torch.jit.is_scripting() and not torch.jit.is_tracing():
        _log_api_usage_once(distance_box_iou)

    boxes1 = _upcast(boxes1)
    boxes2 = _upcast(boxes2)
    diou, _ = _box_diou_iou(boxes1, boxes2, eps=eps)
    return diou


def _box_diou_iou(boxes1: Tensor, boxes2: Tensor, eps: float = 1e-7) -> tuple[Tensor, Tensor]:

    iou = box_iou(boxes1, boxes2)
    lti = torch.min(boxes1[..., None, :2], boxes2[..., None, :, :2])
    rbi = torch.max(boxes1[..., None, 2:], boxes2[..., None, :, 2:])
    whi = _upcast(rbi - lti).clamp(min=0)  # [N,M,2]
    diagonal_distance_squared = (whi[..., 0] ** 2) + (whi[..., 1] ** 2) + eps
    # centers of boxes
    x_p = (boxes1[..., 0] + boxes1[..., 2]) / 2
    y_p = (boxes1[..., 1] + boxes1[..., 3]) / 2
    x_g = (boxes2[..., 0] + boxes2[..., 2]) / 2
    y_g = (boxes2[..., 1] + boxes2[..., 3]) / 2
    # The distance between boxes' centers squared.
    centers_distance_squared = (_upcast(x_p[..., None] - x_g[..., None, :]) ** 2) + (
        _upcast(y_p[..., None] - y_g[..., None, :]) ** 2
    )
    # The distance IoU is the IoU penalized by a normalized
    # distance between boxes' centers squared.
    return iou - (centers_distance_squared / diagonal_distance_squared), iou


# =====================================================
# Rotated Box IoU Implementation (Batched Tensor Operations)
# Algorithm ported from Detectron2's box_iou_rotated_utils.h
# https://github.com/facebookresearch/detectron2/blob/main/detectron2/layers/csrc/box_iou_rotated/box_iou_rotated_utils.h
#
# This implementation uses batched tensor operations instead of Python for-loops
# to leverage GPU parallelism and improve performance.
# =====================================================


def _batched_rotated_box_to_corners(boxes: Tensor) -> Tensor:
    """
    Convert rotated boxes to corner vertices (batched).
    Matches Detectron2's exact corner ordering and rotation convention.

    Args:
        boxes: Tensor[N, 5] in (cx, cy, w, h, angle) format, angle in degrees

    Returns:
        Tensor[N, 4, 2]: Corner vertices for each box
            Order: top-right, top-left, bottom-left, bottom-right
    """
    x_ctr, y_ctr, w, h, angle = boxes.unbind(dim=-1)

    # Convert angle to radians and pre-multiply by 0.5 (Detectron2 convention)
    theta = angle * (torch.pi / 180.0)
    cos_theta = torch.cos(theta) * 0.5
    sin_theta = torch.sin(theta) * 0.5

    # Following Detectron2's exact corner formula
    corners = torch.stack(
        [
            # Corner 0: top-right
            x_ctr + sin_theta * h + cos_theta * w,
            y_ctr + cos_theta * h - sin_theta * w,
            # Corner 1: top-left
            x_ctr - sin_theta * h + cos_theta * w,
            y_ctr - cos_theta * h - sin_theta * w,
            # Corner 2: bottom-left
            x_ctr - sin_theta * h - cos_theta * w,
            y_ctr - cos_theta * h + sin_theta * w,
            # Corner 3: bottom-right
            x_ctr + sin_theta * h - cos_theta * w,
            y_ctr + cos_theta * h + sin_theta * w,
        ],
        dim=-1,
    )

    # Reshape to [N, 4, 2]
    shape = boxes.shape[:-1] + (4, 2)
    return corners.reshape(shape)


def _batched_line_segment_intersection(
    p1: Tensor, p2: Tensor, p3: Tensor, p4: Tensor, eps: float = 1e-8
) -> tuple[Tensor, Tensor]:
    """
    Compute line segment intersections for batched inputs.

    Line segment 1: p1 -> p2
    Line segment 2: p3 -> p4

    Args:
        p1, p2, p3, p4: Tensor[..., 2] - endpoints of line segments

    Returns:
        intersections: Tensor[..., 2] - intersection points
        valid: Tensor[...] - boolean mask for valid intersections
    """
    # Direction vectors
    d1 = p2 - p1  # [..., 2]
    d2 = p4 - p3  # [..., 2]

    # Cross product of directions: d1 x d2
    cross = d1[..., 0] * d2[..., 1] - d1[..., 1] * d2[..., 0]  # [...]

    # Check if lines are parallel
    parallel = torch.abs(cross) < eps

    # Avoid division by zero
    cross_safe = torch.where(parallel, torch.ones_like(cross), cross)

    # Vector from p1 to p3
    d3 = p3 - p1  # [..., 2]

    # Parameters for intersection
    # t = (d3 x d2) / (d1 x d2)
    # s = (d3 x d1) / (d1 x d2)
    t_num = d3[..., 0] * d2[..., 1] - d3[..., 1] * d2[..., 0]
    s_num = d3[..., 0] * d1[..., 1] - d3[..., 1] * d1[..., 0]

    t = t_num / cross_safe
    s = s_num / cross_safe

    # Check if intersection is within both segments [0, 1]
    valid = (~parallel) & (t >= 0) & (t <= 1) & (s >= 0) & (s <= 1)

    # Compute intersection point: p1 + t * d1
    intersection = p1 + t.unsqueeze(-1) * d1

    return intersection, valid


def _batched_point_in_box(points: Tensor, corners: Tensor, eps: float = 1e-5) -> Tensor:
    """
    Check if points are inside rotated rectangles using dot-product method.
    Matches Detectron2's point-in-box check.

    For a rectangle with corners [0,1,2,3], a point P is inside if:
    - 0 <= AP·AB <= AB·AB (projection onto edge 0->1)
    - 0 <= AP·AD <= AD·AD (projection onto edge 0->3)

    Args:
        points: Tensor[..., 2] - points to check
        corners: Tensor[..., 4, 2] - box corners (Detectron2 order: TR, TL, BL, BR)

    Returns:
        Tensor[...] - boolean mask, True if point is inside
    """
    # Edge vectors
    # AB: edge from corner 0 to corner 1
    AB = corners[..., 1, :] - corners[..., 0, :]  # [..., 2]
    # AD: edge from corner 0 to corner 3 (note: original uses -DA where DA = corners[3] - corners[0])
    AD = corners[..., 3, :] - corners[..., 0, :]  # [..., 2]

    # Dot products for normalization
    ABdotAB = AB[..., 0] * AB[..., 0] + AB[..., 1] * AB[..., 1]  # [...]
    ADdotAD = AD[..., 0] * AD[..., 0] + AD[..., 1] * AD[..., 1]  # [...]

    # Vector from corner 0 to point
    AP = points - corners[..., 0, :]  # [..., 2]

    # Projections
    APdotAB = AP[..., 0] * AB[..., 0] + AP[..., 1] * AB[..., 1]  # [...]
    APdotAD = AP[..., 0] * AD[..., 0] + AP[..., 1] * AD[..., 1]  # [...]

    # Point is inside if projections are within [0, edge_length^2]
    inside = (APdotAB >= -eps) & (APdotAB <= ABdotAB + eps) & (APdotAD >= -eps) & (APdotAD <= ADdotAD + eps)

    return inside


def _batched_get_intersection_points(
    corners1: Tensor, corners2: Tensor, eps: float = 1e-5
) -> tuple[Tensor, Tensor, Tensor]:
    """
    Find all intersection points between two sets of rotated boxes (batched).

    For N boxes in set 1 and M boxes in set 2, computes intersections for all N×M pairs.

    Args:
        corners1: Tensor[N, 4, 2] - corners of first set of boxes
        corners2: Tensor[M, 4, 2] - corners of second set of boxes

    Returns:
        intersections: Tensor[N, M, 24, 2] - intersection points (padded)
        valid_mask: Tensor[N, M, 24] - boolean mask for valid points
        counts: Tensor[N, M] - number of valid intersection points per pair
    """
    N = corners1.shape[0]
    M = corners2.shape[0]
    device = corners1.device
    dtype = corners1.dtype

    # Allocate output: max 24 intersection points per pair
    # 16 from edge-edge + 4 from box1 vertices in box2 + 4 from box2 vertices in box1
    intersections = torch.zeros(N, M, 24, 2, dtype=dtype, device=device)
    valid_mask = torch.zeros(N, M, 24, dtype=torch.bool, device=device)

    # Expand corners for broadcasting: [N, 1, 4, 2] and [1, M, 4, 2]
    c1 = corners1[:, None, :, :]  # [N, 1, 4, 2]
    c2 = corners2[None, :, :, :]  # [1, M, 4, 2]

    # --- Part 1: Edge-edge intersections (16 pairs per box pair) ---
    idx = 0
    for i in range(4):  # Edges of box1
        i_next = (i + 1) % 4
        p1 = c1[:, :, i, :]  # [N, M, 2]
        p2 = c1[:, :, i_next, :]

        for j in range(4):  # Edges of box2
            j_next = (j + 1) % 4
            p3 = c2[:, :, j, :]  # [N, M, 2]
            p4 = c2[:, :, j_next, :]

            intersection, valid = _batched_line_segment_intersection(p1, p2, p3, p4, eps)

            # Store intersection points and validity
            intersections[:, :, idx, :] = intersection
            valid_mask[:, :, idx] = valid
            idx += 1

    # --- Part 2: Vertices of box1 inside box2 (4 per pair) ---
    for i in range(4):
        vertex = c1[:, :, i, :]  # [N, M, 2]
        inside = _batched_point_in_box(vertex, c2.expand(N, M, 4, 2), eps)

        intersections[:, :, idx, :] = vertex
        valid_mask[:, :, idx] = inside
        idx += 1

    # --- Part 3: Vertices of box2 inside box1 (4 per pair) ---
    for i in range(4):
        vertex = c2[:, :, i, :]  # [N, M, 2]
        inside = _batched_point_in_box(vertex, c1.expand(N, M, 4, 2), eps)

        intersections[:, :, idx, :] = vertex
        valid_mask[:, :, idx] = inside
        idx += 1

    counts = valid_mask.sum(dim=-1).int()  # [N, M]

    return intersections, valid_mask, counts


def _batched_convex_hull_area(points: Tensor, valid_mask: Tensor, counts: Tensor, eps: float = 1e-8) -> Tensor:
    """
    Compute convex hull area for batched point sets.

    Uses a simplified approach:
    1. Find centroid of valid points
    2. Sort points by angle around centroid
    3. Compute polygon area using shoelace formula

    Args:
        points: Tensor[N, M, 24, 2] - candidate points (padded)
        valid_mask: Tensor[N, M, 24] - boolean mask for valid points
        counts: Tensor[N, M] - number of valid points per pair

    Returns:
        Tensor[N, M] - convex hull areas
    """
    N, M, max_pts, _ = points.shape
    device = points.device
    dtype = points.dtype

    # Handle case where counts <= 2 (no area)
    no_area_mask = counts <= 2  # [N, M]

    # Compute centroid of valid points
    valid_mask_float = valid_mask.unsqueeze(-1).to(dtype)  # [N, M, 24, 1]
    masked_points = points * valid_mask_float  # Zero out invalid points
    centroid = masked_points.sum(dim=2) / counts.unsqueeze(-1).clamp(min=1).to(dtype)  # [N, M, 2]

    # Compute angles from centroid
    relative = points - centroid.unsqueeze(2)  # [N, M, 24, 2]
    angles = torch.atan2(relative[..., 1], relative[..., 0])  # [N, M, 24]

    # Set invalid points to have large angle (will be sorted to end)
    angles = torch.where(valid_mask, angles, torch.full_like(angles, float("inf")))

    # Sort by angle
    sorted_indices = torch.argsort(angles, dim=-1)  # [N, M, 24]

    # Gather sorted points
    sorted_indices_expanded = sorted_indices.unsqueeze(-1).expand(-1, -1, -1, 2)
    sorted_points = torch.gather(points, dim=2, index=sorted_indices_expanded)  # [N, M, 24, 2]

    # Extract x and y coordinates
    x = sorted_points[..., 0]  # [N, M, 24]
    y = sorted_points[..., 1]  # [N, M, 24]

    # Shoelace formula: Area = 0.5 * |sum_{i=0}^{n-1} (x[i]*y[i+1] - x[i+1]*y[i])|
    # where indices wrap around (x[n] = x[0], y[n] = y[0])

    # Part 1: Non-wrap terms (indices 0 to counts-2)
    # cross[i] = x[i] * y[i+1] - x[i+1] * y[i]
    cross_no_wrap = x[:, :, :-1] * y[:, :, 1:] - x[:, :, 1:] * y[:, :, :-1]  # [N, M, 23]

    # Mask: only valid for indices 0 to counts-2
    k_indices = torch.arange(max_pts - 1, device=device).view(1, 1, -1)
    valid_no_wrap = k_indices < (counts.unsqueeze(-1) - 1)  # [N, M, 23]
    cross_no_wrap = torch.where(valid_no_wrap, cross_no_wrap, torch.zeros_like(cross_no_wrap))

    # Sum of non-wrap terms
    sum_no_wrap = cross_no_wrap.sum(dim=-1)  # [N, M]

    # Part 2: Wrap-around term (connecting last valid point back to first)
    # wrap_term = x[counts-1] * y[0] - x[0] * y[counts-1]
    last_idx = (counts - 1).clamp(min=0).long()  # [N, M]
    last_idx_expanded = last_idx.unsqueeze(-1)  # [N, M, 1]

    x_last = torch.gather(x, dim=2, index=last_idx_expanded).squeeze(-1)  # [N, M]
    y_last = torch.gather(y, dim=2, index=last_idx_expanded).squeeze(-1)  # [N, M]

    x_first = x[:, :, 0]  # [N, M]
    y_first = y[:, :, 0]  # [N, M]

    wrap_term = x_last * y_first - x_first * y_last  # [N, M]

    # Total area using shoelace formula
    total_cross = sum_no_wrap + wrap_term
    area = 0.5 * torch.abs(total_cross)  # [N, M]

    # Zero out areas where counts <= 2
    area = torch.where(no_area_mask, torch.zeros_like(area), area)

    return area


def _rotated_box_inter_union(boxes1: Tensor, boxes2: Tensor) -> tuple[Tensor, Tensor]:
    """
    Compute pairwise intersection and union areas for rotated boxes (batched).

    Args:
        boxes1 (Tensor[N, 5]): First set of rotated boxes (x_ctr, y_ctr, w, h, angle)
        boxes2 (Tensor[M, 5]): Second set of rotated boxes (x_ctr, y_ctr, w, h, angle)

    Returns:
        tuple[Tensor, Tensor]: (intersection [N, M], union [N, M])
    """
    # Compute areas
    area1 = boxes1[:, 2] * boxes1[:, 3]  # [N]
    area2 = boxes2[:, 2] * boxes2[:, 3]  # [M]

    # Handle zero area boxes
    zero_area1 = area1 < 1e-14  # [N]
    zero_area2 = area2 < 1e-14  # [M]
    zero_area_mask = zero_area1[:, None] | zero_area2[None, :]  # [N, M]

    # Convert to corners
    corners1 = _batched_rotated_box_to_corners(boxes1)  # [N, 4, 2]
    corners2 = _batched_rotated_box_to_corners(boxes2)  # [M, 4, 2]

    # Shift centers for numerical stability
    # Use mean of all box centers as reference
    all_centers = torch.cat([boxes1[:, :2], boxes2[:, :2]], dim=0)
    center_shift = all_centers.mean(dim=0)  # [2]

    corners1_shifted = corners1 - center_shift
    corners2_shifted = corners2 - center_shift

    # Find all intersection points
    intersections, valid_mask, counts = _batched_get_intersection_points(corners1_shifted, corners2_shifted)

    # Compute intersection area via convex hull
    inter = _batched_convex_hull_area(intersections, valid_mask, counts)

    # Zero out intersection for zero-area boxes
    inter = torch.where(zero_area_mask, torch.zeros_like(inter), inter)

    # Compute union: area1 + area2 - intersection
    union = area1[:, None] + area2[None, :] - inter

    return inter, union


def rotated_box_iou(boxes1: Tensor, boxes2: Tensor, fmt: str = "cxcywhr") -> Tensor:
    """
    Return intersection-over-union (Jaccard index) between two sets of rotated boxes.

    Args:
        boxes1 (Tensor[N, K]): First set of rotated boxes
        boxes2 (Tensor[M, K]): Second set of rotated boxes
        fmt (str): Format of the input boxes. Supported formats are:

            - ``'cxcywhr'`` (default): boxes are represented via center, width, height, and rotation angle.
              (cx, cy) is the center, (w, h) is width and height, r is rotation angle in degrees
              (counter-clockwise positive).

            - ``'xywhr'``: boxes are represented via corner, width, height, and rotation angle.
              (x1, y1) is the top-left corner, (w, h) is width and height, r is rotation angle in degrees.

            - ``'xyxyxyxy'``: boxes are represented via 4 corner coordinates.
              (x1, y1) is top-left, (x2, y2) is top-right, (x3, y3) is bottom-right, (x4, y4) is bottom-left.

    Returns:
        Tensor[N, M]: the NxM matrix containing the pairwise IoU values
            for every element in boxes1 and boxes2

    Example:
        >>> boxes1 = torch.tensor([[100, 100, 50, 30, 0], [200, 200, 60, 40, 45]], dtype=torch.float32)
        >>> boxes2 = torch.tensor([[100, 100, 50, 30, 0], [150, 150, 50, 30, 30]], dtype=torch.float32)
        >>> iou = rotated_box_iou(boxes1, boxes2)
        >>> iou.shape
        torch.Size([2, 2])
    """
    if not torch.jit.is_scripting() and not torch.jit.is_tracing():
        _log_api_usage_once(rotated_box_iou)

    allowed_fmts = ("cxcywhr", "xywhr", "xyxyxyxy")
    if fmt not in allowed_fmts:
        raise ValueError(f"Unsupported format '{fmt}'. Supported formats are {allowed_fmts}")

    # Convert to cxcywhr format for internal computation
    if fmt != "cxcywhr":
        boxes1 = box_convert(boxes1, in_fmt=fmt, out_fmt="cxcywhr")
        boxes2 = box_convert(boxes2, in_fmt=fmt, out_fmt="cxcywhr")

    # Try to use the optimized C++/CUDA implementation
    if _has_ops():
        _assert_has_ops()
        # C++ implementation only supports float32, so convert if needed
        original_dtype = boxes1.dtype
        if original_dtype != torch.float32:
            boxes1 = boxes1.float()
            boxes2 = boxes2.float()
        result = torch.ops.torchvision.box_iou_rotated(boxes1, boxes2)
        # Convert back to original dtype
        if original_dtype != torch.float32:
            result = result.to(original_dtype)
        return result

    # Fallback to batched tensor implementation
    inter, union = _rotated_box_inter_union(boxes1, boxes2)
    iou = inter / union
    return iou


def masks_to_boxes(masks: torch.Tensor) -> torch.Tensor:
    """
    Compute the bounding boxes around the provided masks.

    Returns a [N, 4] tensor containing bounding boxes. The boxes are in ``(x1, y1, x2, y2)`` format with
    ``0 <= x1 <= x2`` and ``0 <= y1 <= y2``.

    .. warning::

        In most cases the output will guarantee ``x1 < x2`` and ``y1 < y2``. But
        if the input is degenerate, e.g. if a mask is a single row or a single
        column, then the output may have x1 = x2 or y1 = y2.

    Args:
        masks (Tensor[N, H, W]): masks to transform where N is the number of masks
            and (H, W) are the spatial dimensions.

    Returns:
        Tensor[N, 4]: bounding boxes
    """
    if not torch.jit.is_scripting() and not torch.jit.is_tracing():
        _log_api_usage_once(masks_to_boxes)
    if masks.numel() == 0:
        return torch.zeros((0, 4), device=masks.device, dtype=torch.float)

    n = masks.shape[0]

    bounding_boxes = torch.zeros((n, 4), device=masks.device, dtype=torch.float)

    for index, mask in enumerate(masks):
        y, x = torch.where(mask != 0)

        bounding_boxes[index, 0] = torch.min(x)
        bounding_boxes[index, 1] = torch.min(y)
        bounding_boxes[index, 2] = torch.max(x)
        bounding_boxes[index, 3] = torch.max(y)

    return bounding_boxes
