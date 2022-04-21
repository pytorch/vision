import torch

from .boxes import distance_box_iou


def distance_box_iou_loss(
    boxes1: torch.Tensor,
    boxes2: torch.Tensor,
    reduction: str = "none",
    eps: float = 1e-7,
) -> torch.Tensor:
    """
    Gradient-friendly IoU loss with an additional penalty that is non-zero when the
    distance between boxes' centers isn't zero. Indeed, for two exactly overlapping
    boxes, the distance IoU is the same as the IoU loss.
    This loss is symmetric, so the boxes1 and boxes2 arguments are interchangeable.

    Both sets of boxes are expected to be in ``(x1, y1, x2, y2)`` format with
    ``0 <= x1 < x2`` and ``0 <= y1 < y2``, and The two boxes should have the
    same dimensions.

    Args:
        boxes1 (Tensor[N, 4] or Tensor[4]): first set of boxes
        boxes2 (Tensor[N, 4] or Tensor[4]): second set of boxes
        reduction (string, optional): Specifies the reduction to apply to the output:
            ``'none'`` | ``'mean'`` | ``'sum'``. ``'none'``: No reduction will be
            applied to the output. ``'mean'``: The output will be averaged.
            ``'sum'``: The output will be summed. Default: ``'none'``
        eps (float, optional): small number to prevent division by zero. Default: 1e-7

    Returns:
        Tensor[]: Loss tensor with the reduction option applied.

    Reference:
        Zhaohui Zheng et. al: Distance Intersection over Union Loss:
        https://arxiv.org/abs/1911.08287
    """
    if boxes1.dim() == 1 and boxes2.dim() == 1:
        batch_boxes1 = boxes1.unsqueeze(0)
        batch_boxes2 = boxes2.unsqueeze(0)
        diou = distance_box_iou(batch_boxes1, batch_boxes2, eps)[0, 0]
    else:
        diou = distance_box_iou(boxes1, boxes2, eps)[0]
    loss = 1 - diou
    if reduction == "mean":
        loss = loss.mean() if loss.numel() > 0 else 0.0 * loss.sum()
    elif reduction == "sum":
        loss = loss.sum()
    # Cast the loss to the same dtype as the input boxes
    loss = loss.to(boxes1.dtype)
    return loss
