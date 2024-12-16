import torch
import triton
from torchvision.ops.boxes import _nms_kernel_postprocess

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
    iou_keep_out_mask = torch.zeros(num_boxes, (num_boxes + 32 - 1) // 32, dtype=torch.int64, device=boxes.device)

    grid = lambda meta: (  # noqa: E731
        triton.cdiv(num_boxes, meta["BLOCK_SIZE"]),
        triton.cdiv(num_boxes, meta["BLOCK_SIZE"]),
    )

    # This triton kernel will calcualte the IoU matrix for all the input boxes (iou_keep_out_mask).
    # The iou_keep_out_mask is defined as a 32-bit long bitmask matrix. So the matrix shape is [N, N//32].
    # Each item [i, j] will be interpreted as whether we should keep box j when we choose box i.
    triton_nms_IoU_kernel[grid](
        boxes,
        iou_keep_out_mask,
        threshold,
        num_boxes,
        iou_keep_out_mask.stride(0),
        iou_keep_out_mask.stride(1),
        BLOCK_SIZE=64,
        num_warps=4,
    )

    # The postprocess will calculate the final indices of the boxes that should be kept.
    # It is a serialized process, and we choose to run it on CPU for more generalization.
    return _nms_kernel_postprocess(order.cpu(), iou_keep_out_mask.cpu(), num_boxes).to(order.device)
