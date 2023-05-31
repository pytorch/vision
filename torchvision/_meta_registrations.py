import functools

import torch
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
