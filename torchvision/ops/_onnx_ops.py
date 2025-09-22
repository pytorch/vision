from collections.abc import Callable
import sys
import warnings

import torch


_INT64_MAX = sys.maxsize


def nms(boxes, scores, iou_threshold: float):
    import onnxscript

    op = onnxscript.opset18
    # boxes: [num_batches, spatial_dimension, 4]
    boxes = op.Unsqueeze(boxes, [0])
    # scores: [num_batches, num_classes, spatial_dimension]
    scores = op.Unsqueeze(scores, [0, 1])
    # nms_out: [num_selected_indices, 3] where each column is [batch_index, class_index, box_index]
    nms_out = op.NonMaxSuppression(boxes, scores, _INT64_MAX, iou_threshold)
    return op.Reshape(op.Slice(nms_out, axes=[1], starts=[2], ends=[3]), [-1])


def _process_batch_indices_for_roi_align(rois):
    import onnxscript
    from onnxscript import INT64

    op = onnxscript.opset18
    # Extract batch indices from the first column (index 0) of rois
    indices = op.Slice(rois, axes=[1], starts=[0], ends=[1])
    indices = op.Squeeze(indices, axes=[1])
    return op.Cast(indices, to=INT64.dtype)


def _process_rois_for_roi_align(rois):
    import onnxscript

    op = onnxscript.opset18
    # Extract roi coordinates from columns 1, 2, 3, 4 (x1, y1, x2, y2)
    return op.Slice(rois, axes=[1], starts=[1], ends=[5])


def _process_sampling_ratio_for_roi_align(sampling_ratio: int):
    if sampling_ratio < 0:
        warnings.warn(
            "ONNX export for RoIAlign with a non-zero sampling_ratio is not supported. "
            "The model will be exported with a sampling_ratio of 0."
        )
        sampling_ratio = 0
    return sampling_ratio


def roi_align(
    input,
    rois,
    spatial_scale: float,
    pooled_height: int,
    pooled_width: int,
    sampling_ratio: int = -1,
    aligned: bool = False,
):
    import onnxscript

    op = onnxscript.opset18
    batch_indices = _process_batch_indices_for_roi_align(rois)
    rois_coords = _process_rois_for_roi_align(rois)
    coordinate_transformation_mode = "half_pixel" if aligned else "output_half_pixel"
    sampling_ratio = _process_sampling_ratio_for_roi_align(sampling_ratio)

    return op.RoiAlign(
        input,
        rois_coords,
        batch_indices,
        coordinate_transformation_mode=coordinate_transformation_mode,
        spatial_scale=spatial_scale,
        output_height=pooled_height,
        output_width=pooled_width,
        sampling_ratio=sampling_ratio,
    )


def roi_pool(input, rois, spatial_scale: float, pooled_height: int, pooled_width: int):
    import onnxscript

    op = onnxscript.opset18
    # MaxRoiPool expects rois in format [batch_index, x1, y1, x2, y2]
    return op.MaxRoiPool(
        input,
        rois,
        pooled_shape=(pooled_height, pooled_width),
        spatial_scale=spatial_scale,
    )


def onnx_translation_table() -> dict[torch._ops.OpOverload, Callable]:
    return {
        torch.ops.torchvision.nms.default: nms,
        torch.ops.torchvision.roi_align.default: roi_align,
        torch.ops.torchvision.roi_pool.default: roi_pool,
    }
