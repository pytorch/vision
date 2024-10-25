import torch
import triton

from torchvision.ops.triton.nms import triton_nms_IoU_kernel


@torch.library.register_kernel("torchvision::nms", "xpu")
def xpu_triton_nms(boxes: torch.Tensor, scores: torch.Tensor, threshold: float) -> torch.Tensor:
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
    num_boxes = boxes.shape[0]

    # Triton does not support argsort yet, thus it needs to fallback to ATen Calls
    order = torch.argsort(scores, descending=True)
    boxes = boxes[order]
    iou_keep_out_mask = torch.zeros(num_boxes, num_boxes, dtype=torch.bool, device=boxes.device)

    grid = lambda meta: (triton.cdiv(num_boxes, meta["BLOCK_SIZE"]), triton.cdiv(num_boxes, meta["BLOCK_SIZE"]))
    # TODO: We need to tune the config from different devices.
    triton_nms_IoU_kernel[grid](boxes, iou_keep_out_mask, threshold, num_boxes, BLOCK_SIZE=64, num_warps=8)

    # # TODO: Need to improve performance for this reduction
    picked = []
    remove_box = torch.zeros(num_boxes, dtype=torch.bool, device=boxes.device)
    for i in range(num_boxes):
        if not (remove_box[i]):
            picked.append(order[i])
            remove_box[i:] |= iou_keep_out_mask[i][i:]

    return torch.as_tensor(picked)
