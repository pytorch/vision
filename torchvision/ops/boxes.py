import torch
import torchvision
from torch import Tensor
from torchvision.extension import _assert_has_ops

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
# Rotated Box IoU Implementation
# Algorithm ported from Detectron2's box_iou_rotated_utils.h
# https://github.com/facebookresearch/detectron2/blob/main/detectron2/layers/csrc/box_iou_rotated/box_iou_rotated_utils.h
# =====================================================


def _rotated_box_to_corners(boxes: Tensor) -> Tensor:
    """
    Convert rotated boxes in (x_ctr, y_ctr, w, h, angle) format to corner vertices.

    Args:
        boxes (Tensor[..., 5]): Rotated boxes in (x_ctr, y_ctr, w, h, angle) format.
            Angle is in degrees, counter-clockwise positive.

    Returns:
        Tensor[..., 4, 2]: Corner vertices for each box, in order:
            top-right, top-left, bottom-left, bottom-right
    """
    x_ctr, y_ctr, w, h, angle = boxes.unbind(dim=-1)

    # Convert angle from degrees to radians
    theta = angle * (torch.pi / 180.0)
    cos_theta = torch.cos(theta) * 0.5
    sin_theta = torch.sin(theta) * 0.5

    # Compute the four corners
    # Following Detectron2's convention
    corners = torch.stack(
        [
            # Corner 0: top-right
            x_ctr + sin_theta * h + cos_theta * w,
            y_ctr + cos_theta * h - sin_theta * w,
            # Corner 1: top-left
            x_ctr - sin_theta * h + cos_theta * w,
            y_ctr - cos_theta * h - sin_theta * w,
            # Corner 2: bottom-left (opposite of corner 0)
            x_ctr - sin_theta * h - cos_theta * w,
            y_ctr - cos_theta * h + sin_theta * w,
            # Corner 3: bottom-right (opposite of corner 1)
            x_ctr + sin_theta * h - cos_theta * w,
            y_ctr + cos_theta * h + sin_theta * w,
        ],
        dim=-1,
    )

    # Reshape to [..., 4, 2] (4 corners, each with x,y coordinates)
    shape = boxes.shape[:-1] + (4, 2)
    return corners.reshape(shape)


def _cross_2d(a: Tensor, b: Tensor) -> Tensor:
    """
    Compute 2D cross product: a.x * b.y - a.y * b.x

    Args:
        a (Tensor[..., 2]): First 2D vector
        b (Tensor[..., 2]): Second 2D vector

    Returns:
        Tensor[...]: Cross product values
    """
    return a[..., 0] * b[..., 1] - a[..., 1] * b[..., 0]


def _dot_2d(a: Tensor, b: Tensor) -> Tensor:
    """
    Compute 2D dot product: a.x * b.x + a.y * b.y

    Args:
        a (Tensor[..., 2]): First 2D vector
        b (Tensor[..., 2]): Second 2D vector

    Returns:
        Tensor[...]: Dot product values
    """
    return a[..., 0] * b[..., 0] + a[..., 1] * b[..., 1]


def _get_intersection_points(pts1: Tensor, pts2: Tensor, eps: float = 1e-5) -> tuple[Tensor, Tensor]:
    """
    Find all intersection points between two rotated rectangles.

    This includes:
    1. Edge-edge intersections (up to 16)
    2. Vertices of rect1 inside rect2 (up to 4)
    3. Vertices of rect2 inside rect1 (up to 4)

    Total: up to 24 points (including duplicates)

    Args:
        pts1 (Tensor[4, 2]): Corner vertices of first rectangle
        pts2 (Tensor[4, 2]): Corner vertices of second rectangle
        eps (float): Epsilon for numerical comparisons

    Returns:
        tuple[Tensor, int]: (intersection_points [24, 2], num_valid_points)
    """
    # Initialize output array for up to 24 intersection points
    intersections = torch.zeros(24, 2, dtype=pts1.dtype, device=pts1.device)
    num = 0

    # Compute edge vectors for both rectangles
    # vec1[i] = pts1[(i+1)%4] - pts1[i]
    vec1 = torch.stack([pts1[(i + 1) % 4] - pts1[i] for i in range(4)])  # [4, 2]
    vec2 = torch.stack([pts2[(i + 1) % 4] - pts2[i] for i in range(4)])  # [4, 2]

    # Part 1: Find edge-edge intersections (16 pairs)
    for i in range(4):
        for j in range(4):
            # Solve for intersection using cross product method
            det = _cross_2d(vec2[j], vec1[i])

            # Skip parallel lines
            if torch.abs(det) <= 1e-14:
                continue

            vec12 = pts2[j] - pts1[i]
            t1 = _cross_2d(vec2[j], vec12) / det
            t2 = _cross_2d(vec1[i], vec12) / det

            # Check if intersection is within both line segments
            if t1 > -eps and t1 < 1.0 + eps and t2 > -eps and t2 < 1.0 + eps:
                intersection = pts1[i] + vec1[i] * t1
                intersections[num] = intersection
                num += 1

    # Part 2: Check vertices of rect1 inside rect2
    AB = vec2[0]  # Edge from pts2[0] to pts2[1]
    DA = vec2[3]  # Edge from pts2[3] to pts2[0]
    ABdotAB = _dot_2d(AB, AB)
    ADdotAD = _dot_2d(DA, DA)

    for i in range(4):
        AP = pts1[i] - pts2[0]
        APdotAB = _dot_2d(AP, AB)
        APdotAD = -_dot_2d(AP, DA)

        if APdotAB > -eps and APdotAD > -eps and APdotAB < ABdotAB + eps and APdotAD < ADdotAD + eps:
            intersections[num] = pts1[i]
            num += 1

    # Part 3: Check vertices of rect2 inside rect1
    AB = vec1[0]
    DA = vec1[3]
    ABdotAB = _dot_2d(AB, AB)
    ADdotAD = _dot_2d(DA, DA)

    for i in range(4):
        AP = pts2[i] - pts1[0]
        APdotAB = _dot_2d(AP, AB)
        APdotAD = -_dot_2d(AP, DA)

        if APdotAB > -eps and APdotAD > -eps and APdotAB < ABdotAB + eps and APdotAD < ADdotAD + eps:
            intersections[num] = pts2[i]
            num += 1

    return intersections, num


def _convex_hull_graham(points: Tensor, num_in: int, shift_to_zero: bool = False) -> tuple[Tensor, int]:
    """
    Compute the convex hull of a set of 2D points using Graham scan algorithm.

    Args:
        points (Tensor[24, 2]): Input points (only first num_in are valid)
        num_in (int): Number of valid input points
        shift_to_zero (bool): If True, return hull centered at origin

    Returns:
        tuple[Tensor, int]: (hull_points [24, 2], num_hull_points)
    """
    if num_in < 2:
        return points.clone(), num_in

    # Output array
    q = torch.zeros_like(points)

    # Step 1: Find point with minimum y (and minimum x if tied)
    t = 0
    for i in range(1, num_in):
        if points[i, 1] < points[t, 1] or (points[i, 1] == points[t, 1] and points[i, 0] < points[t, 0]):
            t = i

    start = points[t].clone()

    # Step 2: Subtract starting point from all points
    for i in range(num_in):
        q[i] = points[i] - start

    # Swap starting point to position 0
    tmp = q[0].clone()
    q[0] = q[t].clone()
    q[t] = tmp

    # Step 3: Sort points by angle (using cross product for comparison)
    # Compute distances for tie-breaking
    dist = torch.zeros(num_in, dtype=points.dtype, device=points.device)
    for i in range(num_in):
        dist[i] = _dot_2d(q[i], q[i])

    # Bubble sort by angle (simple but works for small num_in <= 24)
    for i in range(1, num_in - 1):
        for j in range(i + 1, num_in):
            cross_product = _cross_2d(q[i], q[j])
            if cross_product < -1e-6 or (torch.abs(cross_product) < 1e-6 and dist[i] > dist[j]):
                # Swap q[i] and q[j]
                q_tmp = q[i].clone()
                q[i] = q[j].clone()
                q[j] = q_tmp
                # Swap dist[i] and dist[j]
                dist_tmp = dist[i].clone()
                dist[i] = dist[j].clone()
                dist[j] = dist_tmp

    # Recompute distances after sort
    for i in range(num_in):
        dist[i] = _dot_2d(q[i], q[i])

    # Step 4: Find first non-overlapping point
    k = 1
    while k < num_in and dist[k] <= 1e-8:
        k += 1

    if k == num_in:
        # All points are the same
        q[0] = points[t]
        return q, 1

    q[1] = q[k].clone()
    m = 2  # Points in stack

    # Step 5: Graham scan
    for i in range(k + 1, num_in):
        while m > 1:
            q1 = q[i] - q[m - 2]
            q2 = q[m - 1] - q[m - 2]
            if q1[0] * q2[1] >= q2[0] * q1[1]:
                m -= 1
            else:
                break
        q[m] = q[i].clone()
        m += 1

    # Step 6: Shift back if needed
    if not shift_to_zero:
        for i in range(m):
            q[i] = q[i] + start

    return q, m


def _polygon_area(vertices: Tensor, num: int) -> Tensor:
    """
    Compute the area of a polygon using the triangle fan method.

    Args:
        vertices (Tensor[24, 2]): Polygon vertices (only first num are valid)
        num (int): Number of valid vertices

    Returns:
        Tensor: Polygon area (scalar)
    """
    if num <= 2:
        return torch.tensor(0.0, dtype=vertices.dtype, device=vertices.device)

    area = torch.tensor(0.0, dtype=vertices.dtype, device=vertices.device)
    for i in range(1, num - 1):
        area = area + torch.abs(_cross_2d(vertices[i] - vertices[0], vertices[i + 1] - vertices[0]))

    return area / 2.0


def _rotated_box_inter_union(boxes1: Tensor, boxes2: Tensor) -> tuple[Tensor, Tensor]:
    """
    Compute pairwise intersection and union areas for rotated boxes.

    Args:
        boxes1 (Tensor[N, 5]): First set of rotated boxes (x_ctr, y_ctr, w, h, angle)
        boxes2 (Tensor[M, 5]): Second set of rotated boxes (x_ctr, y_ctr, w, h, angle)

    Returns:
        tuple[Tensor, Tensor]: (intersection [N, M], union [N, M])
    """
    N = boxes1.shape[0]
    M = boxes2.shape[0]

    area1 = boxes1[:, 2] * boxes1[:, 3]  # [N]
    area2 = boxes2[:, 2] * boxes2[:, 3]  # [M]

    inter = torch.zeros(N, M, dtype=boxes1.dtype, device=boxes1.device)

    for i in range(N):
        for j in range(M):
            # Shift centers for numerical precision
            center_shift_x = (boxes1[i, 0] + boxes2[j, 0]) / 2.0
            center_shift_y = (boxes1[i, 1] + boxes2[j, 1]) / 2.0

            box1_shifted = boxes1[i].clone()
            box1_shifted[0] = boxes1[i, 0] - center_shift_x
            box1_shifted[1] = boxes1[i, 1] - center_shift_y

            box2_shifted = boxes2[j].clone()
            box2_shifted[0] = boxes2[j, 0] - center_shift_x
            box2_shifted[1] = boxes2[j, 1] - center_shift_y

            # Skip if either box has zero area
            if area1[i] < 1e-14 or area2[j] < 1e-14:
                continue

            # Convert to corners
            pts1 = _rotated_box_to_corners(box1_shifted)  # [4, 2]
            pts2 = _rotated_box_to_corners(box2_shifted)  # [4, 2]

            # Find intersection points
            intersections, num = _get_intersection_points(pts1, pts2)

            if num <= 2:
                continue

            # Compute convex hull
            hull, num_hull = _convex_hull_graham(intersections, num, shift_to_zero=True)

            # Compute intersection area
            inter[i, j] = _polygon_area(hull, num_hull)

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
