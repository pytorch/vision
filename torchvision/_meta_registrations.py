import torch
import torch.library

# Ensure that torch.ops.torchvision is visible
import torchvision.extension  # noqa: F401

from torch._prims_common import check

_meta_lib = torch.library.Library("torchvision", "IMPL", "Meta")

vision = torch.ops.torchvision


def register_meta(op):
    def wrapper(fn):
        _meta_lib.impl(op, fn)
        return fn

    return wrapper


@register_meta(vision.roi_align.default)
def meta_roi_align(input, rois, spatial_scale, pooled_height, pooled_width, sampling_ratio, aligned):
    check(rois.size(1) == 5, lambda: "rois must have shape as Tensor[K, 5]")
    check(
        input.dtype == rois.dtype,
        lambda: (
            "Expected tensor for input to have the same type as tensor for rois; "
            f"but type {input.dtype} does not equal {rois.dtype}"
        ),
    )
    num_rois = rois.size(0)
    _, channels, height, width = input.size()
    return input.new_empty((num_rois, channels, pooled_height, pooled_width))


@register_meta(vision._roi_align_backward.default)
def meta_roi_align_backward(
    grad, rois, spatial_scale, pooled_height, pooled_width, batch_size, channels, height, width, sampling_ratio, aligned
):
    check(
        grad.dtype == rois.dtype,
        lambda: (
            "Expected tensor for grad to have the same type as tensor for rois; "
            f"but type {grad.dtype} does not equal {rois.dtype}"
        ),
    )
    return grad.new_empty((batch_size, channels, height, width))
