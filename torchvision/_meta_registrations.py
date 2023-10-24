import functools

import torch
import torch._custom_ops
import torch.library

# Ensure that torch.ops.torchvision is visible
import torchvision.extension  # noqa: F401


@functools.lru_cache(None)
def get_meta_lib():
    return torch.library.Library("torchvision", "IMPL", "Meta")


def register_meta(op_name, overload_name="default"):
    def wrapper(fn):
        if torchvision.extension._has_ops():
            get_meta_lib().impl(getattr(getattr(torch.ops.torchvision, op_name), overload_name), fn)
        return fn

    return wrapper


@register_meta("roi_align")
def meta_roi_align(input, rois, spatial_scale, pooled_height, pooled_width, sampling_ratio, aligned):
    torch._check(rois.size(1) == 5, lambda: "rois must have shape as Tensor[K, 5]")
    torch._check(
        input.dtype == rois.dtype,
        lambda: (
            "Expected tensor for input to have the same type as tensor for rois; "
            f"but type {input.dtype} does not equal {rois.dtype}"
        ),
    )
    num_rois = rois.size(0)
    _, channels, height, width = input.size()
    return input.new_empty((num_rois, channels, pooled_height, pooled_width))


@register_meta("_roi_align_backward")
def meta_roi_align_backward(
    grad, rois, spatial_scale, pooled_height, pooled_width, batch_size, channels, height, width, sampling_ratio, aligned
):
    torch._check(
        grad.dtype == rois.dtype,
        lambda: (
            "Expected tensor for grad to have the same type as tensor for rois; "
            f"but type {grad.dtype} does not equal {rois.dtype}"
        ),
    )
    return grad.new_empty((batch_size, channels, height, width))


@register_meta("deform_conv2d")
def meta_deform_conv2d(
    input,
    weight,
    offset,
    mask,
    bias,
    stride_h,
    stride_w,
    pad_h,
    pad_w,
    dil_h,
    dil_w,
    n_weight_grps,
    n_offset_grps,
    use_mask,
):

    out_height, out_width = offset.shape[-2:]
    out_channels = weight.shape[0]
    batch_size = input.shape[0]
    return input.new_empty((batch_size, out_channels, out_height, out_width))


@register_meta("_deform_conv2d_backward")
def meta_deform_conv2d_backward(
    grad,
    input,
    weight,
    offset,
    mask,
    bias,
    stride_h,
    stride_w,
    pad_h,
    pad_w,
    dilation_h,
    dilation_w,
    groups,
    offset_groups,
    use_mask,
):

    grad_input = input.new_empty(input.shape)
    grad_weight = input.new_empty(weight.shape)
    grad_offset = input.new_empty(offset.shape)
    grad_mask = input.new_empty(mask.shape)
    grad_bias = input.new_empty(bias.shape)
    return grad_input, grad_weight, grad_offset, grad_mask, grad_bias


@torch._custom_ops.impl_abstract("torchvision::nms")
def meta_nms(dets, scores, iou_threshold):
    torch._check(dets.dim() == 2, lambda: f"boxes should be a 2d tensor, got {dets.dim()}D")
    torch._check(dets.size(1) == 4, lambda: f"boxes should have 4 elements in dimension 1, got {dets.size(1)}")
    torch._check(scores.dim() == 1, lambda: f"scores should be a 1d tensor, got {scores.dim()}")
    torch._check(
        dets.size(0) == scores.size(0),
        lambda: f"boxes and scores should have same number of elements in dimension 0, got {dets.size(0)} and {scores.size(0)}",
    )
    ctx = torch._custom_ops.get_ctx()
    num_to_keep = ctx.create_unbacked_symint()
    return dets.new_empty(num_to_keep, dtype=torch.long)
