import sys
import warnings

import torch

_onnx_opset_version = 11


def _register_custom_op():
    from torch.onnx.symbolic_helper import parse_args
    from torch.onnx.symbolic_opset11 import select, squeeze, unsqueeze
    from torch.onnx.symbolic_opset9 import _cast_Long

    @parse_args("v", "v", "f")
    def symbolic_multi_label_nms(g, boxes, scores, iou_threshold):
        boxes = unsqueeze(g, boxes, 0)
        scores = unsqueeze(g, unsqueeze(g, scores, 0), 0)
        max_output_per_class = g.op("Constant", value_t=torch.tensor([sys.maxsize], dtype=torch.long))
        iou_threshold = g.op("Constant", value_t=torch.tensor([iou_threshold], dtype=torch.float))
        nms_out = g.op("NonMaxSuppression", boxes, scores, max_output_per_class, iou_threshold)
        return squeeze(g, select(g, nms_out, 1, g.op("Constant", value_t=torch.tensor([2], dtype=torch.long))), 1)

    @parse_args("v", "v", "f", "i", "i", "i", "i")
    def roi_align(g, input, rois, spatial_scale, pooled_height, pooled_width, sampling_ratio, aligned):
        batch_indices = _cast_Long(
            g, squeeze(g, select(g, rois, 1, g.op("Constant", value_t=torch.tensor([0], dtype=torch.long))), 1), False
        )
        rois = select(g, rois, 1, g.op("Constant", value_t=torch.tensor([1, 2, 3, 4], dtype=torch.long)))
        # TODO: Remove this warning after ONNX opset 16 is supported.
        if aligned:
            warnings.warn(
                "ROIAlign with aligned=True is not supported in ONNX, but will be supported in opset 16. "
                "The workaround is that the user need apply the patch "
                "https://github.com/microsoft/onnxruntime/pull/8564 "
                "and build ONNXRuntime from source."
            )

        # ONNX doesn't support negative sampling_ratio
        if sampling_ratio < 0:
            warnings.warn(
                "ONNX doesn't support negative sampling ratio, therefore is is set to 0 in order to be exported."
            )
            sampling_ratio = 0
        return g.op(
            "RoiAlign",
            input,
            rois,
            batch_indices,
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

    from torch.onnx import register_custom_op_symbolic

    register_custom_op_symbolic("torchvision::nms", symbolic_multi_label_nms, _onnx_opset_version)
    register_custom_op_symbolic("torchvision::roi_align", roi_align, _onnx_opset_version)
    register_custom_op_symbolic("torchvision::roi_pool", roi_pool, _onnx_opset_version)
