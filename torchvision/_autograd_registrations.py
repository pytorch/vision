import torch
import torch.library

# Ensure that torch.ops.torchvision is visible
import torchvision.extension  # noqa: F401


# =====================================================================
# Autograd registrations
# =====================================================================


# --- roi_align ---
def _roi_align_setup_context(ctx, inputs, output):
    input, rois, spatial_scale, pooled_height, pooled_width, sampling_ratio, aligned = inputs
    ctx.save_for_backward(rois)
    ctx.input_shape = input.shape
    ctx.spatial_scale = spatial_scale
    ctx.pooled_height = pooled_height
    ctx.pooled_width = pooled_width
    ctx.sampling_ratio = sampling_ratio
    ctx.aligned = aligned


def _roi_align_backward(ctx, grad_output):
    (rois,) = ctx.saved_tensors
    batch_size, channels, height, width = ctx.input_shape
    grad_input = torch.ops.torchvision._roi_align_backward(
        grad_output,
        rois,
        ctx.spatial_scale,
        ctx.pooled_height,
        ctx.pooled_width,
        batch_size,
        channels,
        height,
        width,
        ctx.sampling_ratio,
        ctx.aligned,
    )
    return grad_input, None, None, None, None, None, None


# --- roi_pool ---
def _roi_pool_setup_context(ctx, inputs, output):
    input, rois, spatial_scale, pooled_height, pooled_width = inputs
    pooled, argmax = output
    ctx.save_for_backward(rois, argmax)
    ctx.input_shape = input.shape
    ctx.spatial_scale = spatial_scale
    ctx.pooled_height = pooled_height
    ctx.pooled_width = pooled_width


def _roi_pool_backward(ctx, grad_output, _grad_argmax):
    rois, argmax = ctx.saved_tensors
    batch_size, channels, height, width = ctx.input_shape
    grad_input = torch.ops.torchvision._roi_pool_backward(
        grad_output,
        rois,
        argmax,
        ctx.spatial_scale,
        ctx.pooled_height,
        ctx.pooled_width,
        batch_size,
        channels,
        height,
        width,
    )
    return grad_input, None, None, None, None


# --- ps_roi_align ---
def _ps_roi_align_setup_context(ctx, inputs, output):
    input, rois, spatial_scale, pooled_height, pooled_width, sampling_ratio = inputs
    pooled, channel_mapping = output
    ctx.save_for_backward(rois, channel_mapping)
    ctx.input_shape = input.shape
    ctx.spatial_scale = spatial_scale
    ctx.pooled_height = pooled_height
    ctx.pooled_width = pooled_width
    ctx.sampling_ratio = sampling_ratio


def _ps_roi_align_backward(ctx, grad_output, _grad_channel_mapping):
    rois, channel_mapping = ctx.saved_tensors
    batch_size, channels, height, width = ctx.input_shape
    grad_input = torch.ops.torchvision._ps_roi_align_backward(
        grad_output,
        rois,
        channel_mapping,
        ctx.spatial_scale,
        ctx.pooled_height,
        ctx.pooled_width,
        ctx.sampling_ratio,
        batch_size,
        channels,
        height,
        width,
    )
    return grad_input, None, None, None, None, None


# --- ps_roi_pool ---
def _ps_roi_pool_setup_context(ctx, inputs, output):
    input, rois, spatial_scale, pooled_height, pooled_width = inputs
    pooled, channel_mapping = output
    ctx.save_for_backward(rois, channel_mapping)
    ctx.input_shape = input.shape
    ctx.spatial_scale = spatial_scale
    ctx.pooled_height = pooled_height
    ctx.pooled_width = pooled_width


def _ps_roi_pool_backward(ctx, grad_output, _grad_channel_mapping):
    rois, channel_mapping = ctx.saved_tensors
    batch_size, channels, height, width = ctx.input_shape
    grad_input = torch.ops.torchvision._ps_roi_pool_backward(
        grad_output,
        rois,
        channel_mapping,
        ctx.spatial_scale,
        ctx.pooled_height,
        ctx.pooled_width,
        batch_size,
        channels,
        height,
        width,
    )
    return grad_input, None, None, None, None


# --- deform_conv2d ---
def _deform_conv2d_setup_context(ctx, inputs, output):
    (
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
    ) = inputs
    ctx.save_for_backward(input, weight, offset, mask, bias)
    ctx.stride_h = stride_h
    ctx.stride_w = stride_w
    ctx.pad_h = pad_h
    ctx.pad_w = pad_w
    ctx.dilation_h = dilation_h
    ctx.dilation_w = dilation_w
    ctx.groups = groups
    ctx.offset_groups = offset_groups
    ctx.use_mask = use_mask


def _deform_conv2d_backward(ctx, grad_output):
    input, weight, offset, mask, bias = ctx.saved_tensors
    grad_input, grad_weight, grad_offset, grad_mask, grad_bias = torch.ops.torchvision._deform_conv2d_backward(
        grad_output,
        input,
        weight,
        offset,
        mask,
        bias,
        ctx.stride_h,
        ctx.stride_w,
        ctx.pad_h,
        ctx.pad_w,
        ctx.dilation_h,
        ctx.dilation_w,
        ctx.groups,
        ctx.offset_groups,
        ctx.use_mask,
    )
    return (
        grad_input,
        grad_weight,
        grad_offset,
        grad_mask,
        grad_bias,
        None,
        None,
        None,
        None,
        None,
        None,
        None,
        None,
        None,
    )


# --- Backward op autograd (double-backward blockers) ---
def _no_double_backward(op_name):
    def backward(ctx, *grads):
        raise RuntimeError(f"double backwards on {op_name} not supported")

    return backward


# =====================================================================
# Autocast registrations
# =====================================================================

_all_autocast_keys = (
    torch._C.DispatchKeySet(torch._C.DispatchKey.AutocastCPU)
    | torch._C.DispatchKeySet(torch._C.DispatchKey.AutocastCUDA)
    | torch._C.DispatchKeySet(torch._C.DispatchKey.AutocastXPU)
)


def _autocast_cast(tensor):
    """Cast to float32 matching at::autocast::cached_cast(at::kFloat, ...) behavior.

    Skips non-floating-point tensors and float64 tensors.
    """
    if tensor.is_floating_point() and tensor.dtype is not torch.float64:
        return tensor.float()
    return tensor


def _autocast_nms(dets, scores, iou_threshold):
    with torch._C._ExcludeDispatchKeyGuard(_all_autocast_keys):
        return torch.ops.torchvision.nms(
            _autocast_cast(dets),
            _autocast_cast(scores),
            iou_threshold,
        )


def _autocast_roi_align(input, rois, spatial_scale, pooled_height, pooled_width, sampling_ratio, aligned):
    orig_dtype = input.dtype
    with torch._C._ExcludeDispatchKeyGuard(_all_autocast_keys):
        return torch.ops.torchvision.roi_align(
            _autocast_cast(input),
            _autocast_cast(rois),
            spatial_scale,
            pooled_height,
            pooled_width,
            sampling_ratio,
            aligned,
        ).to(orig_dtype)


def _autocast_roi_pool(input, rois, spatial_scale, pooled_height, pooled_width):
    orig_dtype = input.dtype
    with torch._C._ExcludeDispatchKeyGuard(_all_autocast_keys):
        output, argmax = torch.ops.torchvision.roi_pool(
            _autocast_cast(input),
            _autocast_cast(rois),
            spatial_scale,
            pooled_height,
            pooled_width,
        )
        return output.to(orig_dtype), argmax.to(orig_dtype)


def _autocast_ps_roi_align(input, rois, spatial_scale, pooled_height, pooled_width, sampling_ratio):
    orig_dtype = input.dtype
    with torch._C._ExcludeDispatchKeyGuard(_all_autocast_keys):
        output, channel_mapping = torch.ops.torchvision.ps_roi_align(
            _autocast_cast(input),
            _autocast_cast(rois),
            spatial_scale,
            pooled_height,
            pooled_width,
            sampling_ratio,
        )
        return output.to(orig_dtype), channel_mapping.to(orig_dtype)


def _autocast_ps_roi_pool(input, rois, spatial_scale, pooled_height, pooled_width):
    orig_dtype = input.dtype
    with torch._C._ExcludeDispatchKeyGuard(_all_autocast_keys):
        output, channel_mapping = torch.ops.torchvision.ps_roi_pool(
            _autocast_cast(input),
            _autocast_cast(rois),
            spatial_scale,
            pooled_height,
            pooled_width,
        )
        return output.to(orig_dtype), channel_mapping.to(orig_dtype)


def _autocast_deform_conv2d(
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
    orig_dtype = input.dtype
    with torch._C._ExcludeDispatchKeyGuard(_all_autocast_keys):
        return torch.ops.torchvision.deform_conv2d(
            _autocast_cast(input),
            _autocast_cast(weight),
            _autocast_cast(offset),
            _autocast_cast(mask),
            _autocast_cast(bias),
            stride_h,
            stride_w,
            pad_h,
            pad_w,
            dilation_h,
            dilation_w,
            groups,
            offset_groups,
            use_mask,
        ).to(orig_dtype)


# =====================================================================
# Perform registrations
# =====================================================================

if torchvision.extension._has_ops():
    # --- Autograd: forward ops ---
    torch.library.register_autograd(
        "torchvision::roi_align", _roi_align_backward, setup_context=_roi_align_setup_context
    )
    torch.library.register_autograd("torchvision::roi_pool", _roi_pool_backward, setup_context=_roi_pool_setup_context)
    torch.library.register_autograd(
        "torchvision::ps_roi_align", _ps_roi_align_backward, setup_context=_ps_roi_align_setup_context
    )
    torch.library.register_autograd(
        "torchvision::ps_roi_pool", _ps_roi_pool_backward, setup_context=_ps_roi_pool_setup_context
    )
    torch.library.register_autograd(
        "torchvision::deform_conv2d", _deform_conv2d_backward, setup_context=_deform_conv2d_setup_context
    )

    # --- Autograd: backward ops (block double backward) ---
    torch.library.register_autograd("torchvision::_roi_align_backward", _no_double_backward("roi_align"))
    torch.library.register_autograd("torchvision::_roi_pool_backward", _no_double_backward("roi_pool"))
    torch.library.register_autograd("torchvision::_ps_roi_align_backward", _no_double_backward("ps_roi_align"))
    torch.library.register_autograd("torchvision::_ps_roi_pool_backward", _no_double_backward("ps_roi_pool"))
    torch.library.register_autograd("torchvision::_deform_conv2d_backward", _no_double_backward("deform_conv2d"))

    # --- Autocast ---
    _autocast_lib = torch.library.Library("torchvision", "IMPL")

    # nms and roi_align: registered for all autocast device types
    for _key in ("AutocastCUDA", "AutocastCPU", "AutocastXPU"):
        _autocast_lib.impl("nms", _autocast_nms, _key)
        _autocast_lib.impl("roi_align", _autocast_roi_align, _key)

    # Other ops: CUDA autocast only
    _autocast_lib.impl("roi_pool", _autocast_roi_pool, "AutocastCUDA")
    _autocast_lib.impl("ps_roi_align", _autocast_ps_roi_align, "AutocastCUDA")
    _autocast_lib.impl("ps_roi_pool", _autocast_ps_roi_pool, "AutocastCUDA")
    _autocast_lib.impl("deform_conv2d", _autocast_deform_conv2d, "AutocastCUDA")
