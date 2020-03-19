import sys
import torch

_onnx_opset_version = 11


def _register_custom_op():
    from torch.onnx.symbolic_helper import parse_args, scalar_type_to_onnx, scalar_type_to_pytorch_type, \
        cast_pytorch_to_onnx
    from torch.onnx.symbolic_opset9 import select, unsqueeze, squeeze, _cast_Long, reshape

    @parse_args('v', 'v', 'f')
    def symbolic_multi_label_nms(g, boxes, scores, iou_threshold):
        boxes = unsqueeze(g, boxes, 0)
        scores = unsqueeze(g, unsqueeze(g, scores, 0), 0)
        max_output_per_class = g.op('Constant', value_t=torch.tensor([sys.maxsize], dtype=torch.long))
        iou_threshold = g.op('Constant', value_t=torch.tensor([iou_threshold], dtype=torch.float))
        nms_out = g.op('NonMaxSuppression', boxes, scores, max_output_per_class, iou_threshold)
        return squeeze(g, select(g, nms_out, 1, g.op('Constant', value_t=torch.tensor([2], dtype=torch.long))), 1)

    @parse_args('v', 'v', 'f', 'i', 'i', 'i', 'i')
    def roi_align(g, input, rois, spatial_scale, pooled_height, pooled_width, sampling_ratio, aligned):
        if(aligned):
            raise RuntimeError('Unsupported: ONNX export of roi_align with aligned')
        batch_indices = _cast_Long(g, squeeze(g, select(g, rois, 1, g.op('Constant',
                                   value_t=torch.tensor([0], dtype=torch.long))), 1), False)
        rois = select(g, rois, 1, g.op('Constant', value_t=torch.tensor([1, 2, 3, 4], dtype=torch.long)))
        return g.op('RoiAlign', input, rois, batch_indices, spatial_scale_f=spatial_scale,
                    output_height_i=pooled_height, output_width_i=pooled_width, sampling_ratio_i=sampling_ratio)

    @parse_args('v', 'v', 'f', 'i', 'i')
    def roi_pool(g, input, rois, spatial_scale, pooled_height, pooled_width):
        roi_pool = g.op('MaxRoiPool', input, rois,
                        pooled_shape_i=(pooled_height, pooled_width), spatial_scale_f=spatial_scale)
        return roi_pool, None

    @parse_args('v', 'is')
    def new_empty_tensor_op(g, input, shape):
        dtype = input.type().scalarType()
        if dtype is None:
            dtype = 'Float'
        dtype = scalar_type_to_onnx.index(cast_pytorch_to_onnx[dtype])
        shape = g.op("Constant", value_t=torch.tensor(shape))
        return g.op("ConstantOfShape", shape,
                    value_t=torch.tensor([0], dtype=scalar_type_to_pytorch_type[dtype]))

    from torch.onnx import register_custom_op_symbolic
    register_custom_op_symbolic('torchvision::nms', symbolic_multi_label_nms, _onnx_opset_version)
    register_custom_op_symbolic('torchvision::roi_align', roi_align, _onnx_opset_version)
    register_custom_op_symbolic('torchvision::roi_pool', roi_pool, _onnx_opset_version)
    register_custom_op_symbolic('torchvision::_new_empty_tensor_op', new_empty_tensor_op, _onnx_opset_version)
