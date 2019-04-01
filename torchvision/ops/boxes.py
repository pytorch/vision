import torch
from torchvision import _C


def nms(boxes, scores, iou_threshold):
    """
    Performs non-maximum suppression (NMS) on the boxes according
    to their intersection-over-union (IoU).

    NMS iteratively removes lower scoring boxes which have an
    IoU greater than iou_threshold with another (higher scoring)
    box.

    Arguments:
        boxes (Tensor[N, 4]): boxes to perform NMS on
        scores (Tensor[N]): scores for each one of the boxes
        iou_threshold (float): discards all overlapping
            boxes with IoU < iou_threshold

    Returns:
        keep (Tensor): int64 tensor with the indices
            of the elements that have been kept
            by NMS
    """
    return _C.nms(boxes, scores, iou_threshold)


def box_area(left_top, right_bottom):
    """Compute the areas of rectangles given two corners.
    Args:
        left_top (N, 2): left top corner.
        right_bottom (N, 2): right bottom corner.
    Returns:
        area (N): return the area.
    """
    hw = torch.clamp(right_bottom - left_top, min=0.0)
    return hw[..., 0] * hw[..., 1]


def box_iou(boxes0, boxes1, eps=1e-5):
    """Return intersection-over-union (Jaccard index) of boxes.
    Args:
        boxes0 (N, 4): ground truth boxes.
        boxes1 (N or 1, 4): predicted boxes.
        eps: a small number to avoid 0 as denominator.
    Returns:
        iou (N): IoU values.
    """
    overlap_left_top = torch.max(boxes0[..., :2], boxes1[..., :2])
    overlap_right_bottom = torch.min(boxes0[..., 2:], boxes1[..., 2:])

    overlap_area = box_area(overlap_left_top, overlap_right_bottom)
    area0 = box_area(boxes0[..., :2], boxes0[..., 2:])
    area1 = box_area(boxes1[..., :2], boxes1[..., 2:])
    return overlap_area / (area0 + area1 - overlap_area + eps)
