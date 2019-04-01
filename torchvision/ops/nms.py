from torchvision import _C


def nms(boxes, scores, iou_threshold):
    """
    Performs non-maximum suppression.

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
