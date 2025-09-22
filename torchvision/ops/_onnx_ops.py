import sys
import warnings

import torchvision
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


def _process_batch_indices_for_roi_align(g, rois):
    indices = opset11.squeeze(
        g, opset11.select(g, rois, 1, g.op("Constant", value_t=torch.tensor([0], dtype=torch.long))), 1
    )
    return g.op("Cast", indices, to_i=torch.onnx.TensorProtoDataType.INT64)


def _process_rois_for_roi_align(g, rois):
    return opset11.select(g, rois, 1, g.op("Constant", value_t=torch.tensor([1, 2, 3, 4], dtype=torch.long)))


def _process_sampling_ratio_for_roi_align(g, sampling_ratio: int):
    if sampling_ratio < 0:
        warnings.warn(
            "ONNX export for RoIAlign with a non-zero sampling_ratio is not supported. "
            "The model will be exported with a sampling_ratio of 0."
        )
        sampling_ratio = 0
    return sampling_ratio


@parse_args("v", "v", "f", "i", "i", "i", "i")
def roi_align_opset16(g, input, rois, spatial_scale, pooled_height, pooled_width, sampling_ratio, aligned):
    batch_indices = _process_batch_indices_for_roi_align(g, rois)
    rois = _process_rois_for_roi_align(g, rois)
    coordinate_transformation_mode = "half_pixel" if aligned else "output_half_pixel"
    sampling_ratio = _process_sampling_ratio_for_roi_align(g, sampling_ratio)
    return g.op(
        "RoiAlign",
        input,
        rois,
        batch_indices,
        coordinate_transformation_mode_s=coordinate_transformation_mode,
        spatial_scale_f=spatial_scale,
        output_height_i=pooled_height,
        output_width_i=pooled_width,
        sampling_ratio_i=sampling_ratio,
    )


@parse_args("v", "v", "f", "i", "i")
def roi_pool(g, input, rois, spatial_scale, pooled_height, pooled_width):
    roi_pool = g.op(
        "MaxRoiPool", input, rois, pooled_shape_i=(pooled_height, pooled_width), spatial_scale_f=spatial_scale
    )
    return roi_pool, None


def onnx_translation_table():
    return {
        torchvision.ops.nms: nms,
        torchvision.ops.roi_align: roi_align,
        torchvision.ops.roi_pool: roi_pool,
    }
