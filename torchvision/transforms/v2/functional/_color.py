from typing import Union

import PIL.Image
import torch
from torch.nn.functional import conv2d
from torchvision import datapoints
from torchvision.transforms import _functional_pil as _FP
from torchvision.transforms._functional_tensor import _max_value

from torchvision.utils import _log_api_usage_once

from ._meta import _num_value_bits, convert_dtype_image_tensor
from ._utils import is_simple_tensor


def _rgb_to_grayscale_image_tensor(
    image: torch.Tensor, num_output_channels: int = 1, preserve_dtype: bool = True
) -> torch.Tensor:
    if image.shape[-3] == 1:
        return image.clone()

    r, g, b = image.unbind(dim=-3)
    l_img = r.mul(0.2989).add_(g, alpha=0.587).add_(b, alpha=0.114)
    l_img = l_img.unsqueeze(dim=-3)
    if preserve_dtype:
        l_img = l_img.to(image.dtype)
    if num_output_channels == 3:
        l_img = l_img.expand(image.shape)
    return l_img


def rgb_to_grayscale_image_tensor(image: torch.Tensor, num_output_channels: int = 1) -> torch.Tensor:
    return _rgb_to_grayscale_image_tensor(image, num_output_channels=num_output_channels, preserve_dtype=True)


rgb_to_grayscale_image_pil = _FP.to_grayscale


def rgb_to_grayscale(
    inpt: Union[datapoints._ImageTypeJIT, datapoints._VideoTypeJIT], num_output_channels: int = 1
) -> Union[datapoints._ImageTypeJIT, datapoints._VideoTypeJIT]:
    if not torch.jit.is_scripting():
        _log_api_usage_once(rgb_to_grayscale)
    if num_output_channels not in (1, 3):
        raise ValueError(f"num_output_channels must be 1 or 3, got {num_output_channels}.")
    if torch.jit.is_scripting() or is_simple_tensor(inpt):
        return rgb_to_grayscale_image_tensor(inpt, num_output_channels=num_output_channels)
    elif isinstance(inpt, datapoints._datapoint.Datapoint):
        return inpt.rgb_to_grayscale(num_output_channels=num_output_channels)
    elif isinstance(inpt, PIL.Image.Image):
        return rgb_to_grayscale_image_pil(inpt, num_output_channels=num_output_channels)
    else:
        raise TypeError(
            f"Input can either be a plain tensor, any TorchVision datapoint, or a PIL image, "
            f"but got {type(inpt)} instead."
        )


def _blend(image1: torch.Tensor, image2: torch.Tensor, ratio: float) -> torch.Tensor:
    ratio = float(ratio)
    fp = image1.is_floating_point()
    bound = _max_value(image1.dtype)
    output = image1.mul(ratio).add_(image2, alpha=(1.0 - ratio)).clamp_(0, bound)
    return output if fp else output.to(image1.dtype)


def adjust_brightness_image_tensor(image: torch.Tensor, brightness_factor: float) -> torch.Tensor:
    if brightness_factor < 0:
        raise ValueError(f"brightness_factor ({brightness_factor}) is not non-negative.")

    c = image.shape[-3]
    if c not in [1, 3]:
        raise TypeError(f"Input image tensor permitted channel values are 1 or 3, but found {c}")

    fp = image.is_floating_point()
    bound = _max_value(image.dtype)
    output = image.mul(brightness_factor).clamp_(0, bound)
    return output if fp else output.to(image.dtype)


adjust_brightness_image_pil = _FP.adjust_brightness


def adjust_brightness_video(video: torch.Tensor, brightness_factor: float) -> torch.Tensor:
    return adjust_brightness_image_tensor(video, brightness_factor=brightness_factor)


def adjust_brightness(inpt: datapoints._InputTypeJIT, brightness_factor: float) -> datapoints._InputTypeJIT:
    if not torch.jit.is_scripting():
        _log_api_usage_once(adjust_brightness)

    if torch.jit.is_scripting() or is_simple_tensor(inpt):
        return adjust_brightness_image_tensor(inpt, brightness_factor=brightness_factor)
    elif isinstance(inpt, datapoints._datapoint.Datapoint):
        return inpt.adjust_brightness(brightness_factor=brightness_factor)
    elif isinstance(inpt, PIL.Image.Image):
        return adjust_brightness_image_pil(inpt, brightness_factor=brightness_factor)
    else:
        raise TypeError(
            f"Input can either be a plain tensor, any TorchVision datapoint, or a PIL image, "
            f"but got {type(inpt)} instead."
        )


def adjust_saturation_image_tensor(image: torch.Tensor, saturation_factor: float) -> torch.Tensor:
    if saturation_factor < 0:
        raise ValueError(f"saturation_factor ({saturation_factor}) is not non-negative.")

    c = image.shape[-3]
    if c not in [1, 3]:
        raise TypeError(f"Input image tensor permitted channel values are 1 or 3, but found {c}")

    if c == 1:  # Match PIL behaviour
        return image

    grayscale_image = _rgb_to_grayscale_image_tensor(image, num_output_channels=1, preserve_dtype=False)
    if not image.is_floating_point():
        grayscale_image = grayscale_image.floor_()

    return _blend(image, grayscale_image, saturation_factor)


adjust_saturation_image_pil = _FP.adjust_saturation


def adjust_saturation_video(video: torch.Tensor, saturation_factor: float) -> torch.Tensor:
    return adjust_saturation_image_tensor(video, saturation_factor=saturation_factor)


def adjust_saturation(inpt: datapoints._InputTypeJIT, saturation_factor: float) -> datapoints._InputTypeJIT:
    if not torch.jit.is_scripting():
        _log_api_usage_once(adjust_saturation)

    if isinstance(inpt, torch.Tensor) and (
        torch.jit.is_scripting() or not isinstance(inpt, datapoints._datapoint.Datapoint)
    ):
        return adjust_saturation_image_tensor(inpt, saturation_factor=saturation_factor)
    elif isinstance(inpt, datapoints._datapoint.Datapoint):
        return inpt.adjust_saturation(saturation_factor=saturation_factor)
    elif isinstance(inpt, PIL.Image.Image):
        return adjust_saturation_image_pil(inpt, saturation_factor=saturation_factor)
    else:
        raise TypeError(
            f"Input can either be a plain tensor, any TorchVision datapoint, or a PIL image, "
            f"but got {type(inpt)} instead."
        )


def adjust_contrast_image_tensor(image: torch.Tensor, contrast_factor: float) -> torch.Tensor:
    if contrast_factor < 0:
        raise ValueError(f"contrast_factor ({contrast_factor}) is not non-negative.")

    c = image.shape[-3]
    if c not in [1, 3]:
        raise TypeError(f"Input image tensor permitted channel values are 1 or 3, but found {c}")
    fp = image.is_floating_point()
    if c == 3:
        grayscale_image = _rgb_to_grayscale_image_tensor(image, num_output_channels=1, preserve_dtype=False)
        if not fp:
            grayscale_image = grayscale_image.floor_()
    else:
        grayscale_image = image if fp else image.to(torch.float32)
    mean = torch.mean(grayscale_image, dim=(-3, -2, -1), keepdim=True)
    return _blend(image, mean, contrast_factor)


adjust_contrast_image_pil = _FP.adjust_contrast


def adjust_contrast_video(video: torch.Tensor, contrast_factor: float) -> torch.Tensor:
    return adjust_contrast_image_tensor(video, contrast_factor=contrast_factor)


def adjust_contrast(inpt: datapoints._InputTypeJIT, contrast_factor: float) -> datapoints._InputTypeJIT:
    if not torch.jit.is_scripting():
        _log_api_usage_once(adjust_contrast)

    if torch.jit.is_scripting() or is_simple_tensor(inpt):
        return adjust_contrast_image_tensor(inpt, contrast_factor=contrast_factor)
    elif isinstance(inpt, datapoints._datapoint.Datapoint):
        return inpt.adjust_contrast(contrast_factor=contrast_factor)
    elif isinstance(inpt, PIL.Image.Image):
        return adjust_contrast_image_pil(inpt, contrast_factor=contrast_factor)
    else:
        raise TypeError(
            f"Input can either be a plain tensor, any TorchVision datapoint, or a PIL image, "
            f"but got {type(inpt)} instead."
        )


def adjust_sharpness_image_tensor(image: torch.Tensor, sharpness_factor: float) -> torch.Tensor:
    num_channels, height, width = image.shape[-3:]
    if num_channels not in (1, 3):
        raise TypeError(f"Input image tensor can have 1 or 3 channels, but found {num_channels}")

    if sharpness_factor < 0:
        raise ValueError(f"sharpness_factor ({sharpness_factor}) is not non-negative.")

    if image.numel() == 0 or height <= 2 or width <= 2:
        return image

    bound = _max_value(image.dtype)
    fp = image.is_floating_point()
    shape = image.shape

    if image.ndim > 4:
        image = image.reshape(-1, num_channels, height, width)
        needs_unsquash = True
    else:
        needs_unsquash = False

    # The following is a normalized 3x3 kernel with 1s in the edges and a 5 in the middle.
    kernel_dtype = image.dtype if fp else torch.float32
    a, b = 1.0 / 13.0, 5.0 / 13.0
    kernel = torch.tensor([[a, a, a], [a, b, a], [a, a, a]], dtype=kernel_dtype, device=image.device)
    kernel = kernel.expand(num_channels, 1, 3, 3)

    # We copy and cast at the same time to avoid modifications on the original data
    output = image.to(dtype=kernel_dtype, copy=True)
    blurred_degenerate = conv2d(output, kernel, groups=num_channels)
    if not fp:
        # it is better to round before cast
        blurred_degenerate = blurred_degenerate.round_()

    # Create a view on the underlying output while pointing at the same data. We do this to avoid indexing twice.
    view = output[..., 1:-1, 1:-1]

    # We speed up blending by minimizing flops and doing in-place. The 2 blend options are mathematically equivalent:
    # x+(1-r)*(y-x) = x + (1-r)*y - (1-r)*x = x*r + y*(1-r)
    view.add_(blurred_degenerate.sub_(view), alpha=(1.0 - sharpness_factor))

    # The actual data of output have been modified by the above. We only need to clamp and cast now.
    output = output.clamp_(0, bound)
    if not fp:
        output = output.to(image.dtype)

    if needs_unsquash:
        output = output.reshape(shape)

    return output


adjust_sharpness_image_pil = _FP.adjust_sharpness


def adjust_sharpness_video(video: torch.Tensor, sharpness_factor: float) -> torch.Tensor:
    return adjust_sharpness_image_tensor(video, sharpness_factor=sharpness_factor)


def adjust_sharpness(inpt: datapoints._InputTypeJIT, sharpness_factor: float) -> datapoints._InputTypeJIT:
    if not torch.jit.is_scripting():
        _log_api_usage_once(adjust_sharpness)

    if isinstance(inpt, torch.Tensor) and (
        torch.jit.is_scripting() or not isinstance(inpt, datapoints._datapoint.Datapoint)
    ):
        return adjust_sharpness_image_tensor(inpt, sharpness_factor=sharpness_factor)
    elif isinstance(inpt, datapoints._datapoint.Datapoint):
        return inpt.adjust_sharpness(sharpness_factor=sharpness_factor)
    elif isinstance(inpt, PIL.Image.Image):
        return adjust_sharpness_image_pil(inpt, sharpness_factor=sharpness_factor)
    else:
        raise TypeError(
            f"Input can either be a plain tensor, any TorchVision datapoint, or a PIL image, "
            f"but got {type(inpt)} instead."
        )


def _rgb_to_hsv(image: torch.Tensor) -> torch.Tensor:
    r, g, _ = image.unbind(dim=-3)

    # Implementation is based on
    # https://github.com/python-pillow/Pillow/blob/4174d4267616897df3746d315d5a2d0f82c656ee/src/libImaging/Convert.c#L330
    minc, maxc = torch.aminmax(image, dim=-3)

    # The algorithm erases S and H channel where `maxc = minc`. This avoids NaN
    # from happening in the results, because
    #   + S channel has division by `maxc`, which is zero only if `maxc = minc`
    #   + H channel has division by `(maxc - minc)`.
    #
    # Instead of overwriting NaN afterwards, we just prevent it from occurring so
    # we don't need to deal with it in case we save the NaN in a buffer in
    # backprop, if it is ever supported, but it doesn't hurt to do so.
    eqc = maxc == minc

    channels_range = maxc - minc
    # Since `eqc => channels_range = 0`, replacing denominator with 1 when `eqc` is fine.
    ones = torch.ones_like(maxc)
    s = channels_range / torch.where(eqc, ones, maxc)
    # Note that `eqc => maxc = minc = r = g = b`. So the following calculation
    # of `h` would reduce to `bc - gc + 2 + rc - bc + 4 + rc - bc = 6` so it
    # would not matter what values `rc`, `gc`, and `bc` have here, and thus
    # replacing denominator with 1 when `eqc` is fine.
    channels_range_divisor = torch.where(eqc, ones, channels_range).unsqueeze_(dim=-3)
    rc, gc, bc = ((maxc.unsqueeze(dim=-3) - image) / channels_range_divisor).unbind(dim=-3)

    mask_maxc_neq_r = maxc != r
    mask_maxc_eq_g = maxc == g

    hg = rc.add(2.0).sub_(bc).mul_(mask_maxc_eq_g & mask_maxc_neq_r)
    hr = bc.sub_(gc).mul_(~mask_maxc_neq_r)
    hb = gc.add_(4.0).sub_(rc).mul_(mask_maxc_neq_r.logical_and_(mask_maxc_eq_g.logical_not_()))

    h = hr.add_(hg).add_(hb)
    h = h.mul_(1.0 / 6.0).add_(1.0).fmod_(1.0)
    return torch.stack((h, s, maxc), dim=-3)


def _hsv_to_rgb(img: torch.Tensor) -> torch.Tensor:
    h, s, v = img.unbind(dim=-3)
    h6 = h.mul(6)
    i = torch.floor(h6)
    f = h6.sub_(i)
    i = i.to(dtype=torch.int32)

    sxf = s * f
    one_minus_s = 1.0 - s
    q = (1.0 - sxf).mul_(v).clamp_(0.0, 1.0)
    t = sxf.add_(one_minus_s).mul_(v).clamp_(0.0, 1.0)
    p = one_minus_s.mul_(v).clamp_(0.0, 1.0)
    i.remainder_(6)

    mask = i.unsqueeze(dim=-3) == torch.arange(6, device=i.device).view(-1, 1, 1)

    a1 = torch.stack((v, q, p, p, t, v), dim=-3)
    a2 = torch.stack((t, v, v, q, p, p), dim=-3)
    a3 = torch.stack((p, p, t, v, v, q), dim=-3)
    a4 = torch.stack((a1, a2, a3), dim=-4)

    return (a4.mul_(mask.unsqueeze(dim=-4))).sum(dim=-3)


def adjust_hue_image_tensor(image: torch.Tensor, hue_factor: float) -> torch.Tensor:
    if not (-0.5 <= hue_factor <= 0.5):
        raise ValueError(f"hue_factor ({hue_factor}) is not in [-0.5, 0.5].")

    c = image.shape[-3]
    if c not in [1, 3]:
        raise TypeError(f"Input image tensor permitted channel values are 1 or 3, but found {c}")

    if c == 1:  # Match PIL behaviour
        return image

    if image.numel() == 0:
        # exit earlier on empty images
        return image

    orig_dtype = image.dtype
    image = convert_dtype_image_tensor(image, torch.float32)

    image = _rgb_to_hsv(image)
    h, s, v = image.unbind(dim=-3)
    h.add_(hue_factor).remainder_(1.0)
    image = torch.stack((h, s, v), dim=-3)
    image_hue_adj = _hsv_to_rgb(image)

    return convert_dtype_image_tensor(image_hue_adj, orig_dtype)


adjust_hue_image_pil = _FP.adjust_hue


def adjust_hue_video(video: torch.Tensor, hue_factor: float) -> torch.Tensor:
    return adjust_hue_image_tensor(video, hue_factor=hue_factor)


def adjust_hue(inpt: datapoints._InputTypeJIT, hue_factor: float) -> datapoints._InputTypeJIT:
    if not torch.jit.is_scripting():
        _log_api_usage_once(adjust_hue)

    if torch.jit.is_scripting() or is_simple_tensor(inpt):
        return adjust_hue_image_tensor(inpt, hue_factor=hue_factor)
    elif isinstance(inpt, datapoints._datapoint.Datapoint):
        return inpt.adjust_hue(hue_factor=hue_factor)
    elif isinstance(inpt, PIL.Image.Image):
        return adjust_hue_image_pil(inpt, hue_factor=hue_factor)
    else:
        raise TypeError(
            f"Input can either be a plain tensor, any TorchVision datapoint, or a PIL image, "
            f"but got {type(inpt)} instead."
        )


def adjust_gamma_image_tensor(image: torch.Tensor, gamma: float, gain: float = 1.0) -> torch.Tensor:
    if gamma < 0:
        raise ValueError("Gamma should be a non-negative real number")

    # The input image is either assumed to be at [0, 1] scale (if float) or is converted to that scale (if integer).
    # Since the gamma is non-negative, the output remains at [0, 1] scale.
    if not torch.is_floating_point(image):
        output = convert_dtype_image_tensor(image, torch.float32).pow_(gamma)
    else:
        output = image.pow(gamma)

    if gain != 1.0:
        # The clamp operation is needed only if multiplication is performed. It's only when gain != 1, that the scale
        # of the output can go beyond [0, 1].
        output = output.mul_(gain).clamp_(0.0, 1.0)

    return convert_dtype_image_tensor(output, image.dtype)


adjust_gamma_image_pil = _FP.adjust_gamma


def adjust_gamma_video(video: torch.Tensor, gamma: float, gain: float = 1) -> torch.Tensor:
    return adjust_gamma_image_tensor(video, gamma=gamma, gain=gain)


def adjust_gamma(inpt: datapoints._InputTypeJIT, gamma: float, gain: float = 1) -> datapoints._InputTypeJIT:
    if not torch.jit.is_scripting():
        _log_api_usage_once(adjust_gamma)

    if torch.jit.is_scripting() or is_simple_tensor(inpt):
        return adjust_gamma_image_tensor(inpt, gamma=gamma, gain=gain)
    elif isinstance(inpt, datapoints._datapoint.Datapoint):
        return inpt.adjust_gamma(gamma=gamma, gain=gain)
    elif isinstance(inpt, PIL.Image.Image):
        return adjust_gamma_image_pil(inpt, gamma=gamma, gain=gain)
    else:
        raise TypeError(
            f"Input can either be a plain tensor, any TorchVision datapoint, or a PIL image, "
            f"but got {type(inpt)} instead."
        )


def posterize_image_tensor(image: torch.Tensor, bits: int) -> torch.Tensor:
    if image.is_floating_point():
        levels = 1 << bits
        return image.mul(levels).floor_().clamp_(0, levels - 1).mul_(1.0 / levels)
    else:
        num_value_bits = _num_value_bits(image.dtype)
        if bits >= num_value_bits:
            return image

        mask = ((1 << bits) - 1) << (num_value_bits - bits)
        return image & mask


posterize_image_pil = _FP.posterize


def posterize_video(video: torch.Tensor, bits: int) -> torch.Tensor:
    return posterize_image_tensor(video, bits=bits)


def posterize(inpt: datapoints._InputTypeJIT, bits: int) -> datapoints._InputTypeJIT:
    if not torch.jit.is_scripting():
        _log_api_usage_once(posterize)

    if torch.jit.is_scripting() or is_simple_tensor(inpt):
        return posterize_image_tensor(inpt, bits=bits)
    elif isinstance(inpt, datapoints._datapoint.Datapoint):
        return inpt.posterize(bits=bits)
    elif isinstance(inpt, PIL.Image.Image):
        return posterize_image_pil(inpt, bits=bits)
    else:
        raise TypeError(
            f"Input can either be a plain tensor, any TorchVision datapoint, or a PIL image, "
            f"but got {type(inpt)} instead."
        )


def solarize_image_tensor(image: torch.Tensor, threshold: float) -> torch.Tensor:
    if threshold > _max_value(image.dtype):
        raise TypeError(f"Threshold should be less or equal the maximum value of the dtype, but got {threshold}")

    return torch.where(image >= threshold, invert_image_tensor(image), image)


solarize_image_pil = _FP.solarize


def solarize_video(video: torch.Tensor, threshold: float) -> torch.Tensor:
    return solarize_image_tensor(video, threshold=threshold)


def solarize(inpt: datapoints._InputTypeJIT, threshold: float) -> datapoints._InputTypeJIT:
    if not torch.jit.is_scripting():
        _log_api_usage_once(solarize)

    if torch.jit.is_scripting() or is_simple_tensor(inpt):
        return solarize_image_tensor(inpt, threshold=threshold)
    elif isinstance(inpt, datapoints._datapoint.Datapoint):
        return inpt.solarize(threshold=threshold)
    elif isinstance(inpt, PIL.Image.Image):
        return solarize_image_pil(inpt, threshold=threshold)
    else:
        raise TypeError(
            f"Input can either be a plain tensor, any TorchVision datapoint, or a PIL image, "
            f"but got {type(inpt)} instead."
        )


def autocontrast_image_tensor(image: torch.Tensor) -> torch.Tensor:
    c = image.shape[-3]
    if c not in [1, 3]:
        raise TypeError(f"Input image tensor permitted channel values are 1 or 3, but found {c}")

    if image.numel() == 0:
        # exit earlier on empty images
        return image

    bound = _max_value(image.dtype)
    fp = image.is_floating_point()
    float_image = image if fp else image.to(torch.float32)

    minimum = float_image.amin(dim=(-2, -1), keepdim=True)
    maximum = float_image.amax(dim=(-2, -1), keepdim=True)

    eq_idxs = maximum == minimum
    inv_scale = maximum.sub_(minimum).mul_(1.0 / bound)
    minimum[eq_idxs] = 0.0
    inv_scale[eq_idxs] = 1.0

    if fp:
        diff = float_image.sub(minimum)
    else:
        diff = float_image.sub_(minimum)

    return diff.div_(inv_scale).clamp_(0, bound).to(image.dtype)


autocontrast_image_pil = _FP.autocontrast


def autocontrast_video(video: torch.Tensor) -> torch.Tensor:
    return autocontrast_image_tensor(video)


def autocontrast(inpt: datapoints._InputTypeJIT) -> datapoints._InputTypeJIT:
    if not torch.jit.is_scripting():
        _log_api_usage_once(autocontrast)

    if torch.jit.is_scripting() or is_simple_tensor(inpt):
        return autocontrast_image_tensor(inpt)
    elif isinstance(inpt, datapoints._datapoint.Datapoint):
        return inpt.autocontrast()
    elif isinstance(inpt, PIL.Image.Image):
        return autocontrast_image_pil(inpt)
    else:
        raise TypeError(
            f"Input can either be a plain tensor, any TorchVision datapoint, or a PIL image, "
            f"but got {type(inpt)} instead."
        )


def equalize_image_tensor(image: torch.Tensor) -> torch.Tensor:
    if image.numel() == 0:
        return image

    # 1. The algorithm below can easily be extended to support arbitrary integer dtypes. However, the histogram that
    #    would be needed to computed will have at least `torch.iinfo(dtype).max + 1` values. That is perfectly fine for
    #    `torch.int8`, `torch.uint8`, and `torch.int16`, at least questionable for `torch.int32` and completely
    #    unfeasible for `torch.int64`.
    # 2. Floating point inputs need to be binned for this algorithm. Apart from converting them to an integer dtype, we
    #    could also use PyTorch's builtin histogram functionality. However, that has its own set of issues: in addition
    #    to being slow in general, PyTorch's implementation also doesn't support batches. In total, that makes it slower
    #    and more complicated to implement than a simple conversion and a fast histogram implementation for integers.
    # Since we need to convert in most cases anyway and out of the acceptable dtypes mentioned in 1. `torch.uint8` is
    # by far the most common, we choose it as base.
    output_dtype = image.dtype
    image = convert_dtype_image_tensor(image, torch.uint8)

    # The histogram is computed by using the flattened image as index. For example, a pixel value of 127 in the image
    # corresponds to adding 1 to index 127 in the histogram.
    batch_shape = image.shape[:-2]
    flat_image = image.flatten(start_dim=-2).to(torch.long)
    hist = flat_image.new_zeros(batch_shape + (256,), dtype=torch.int32)
    hist.scatter_add_(dim=-1, index=flat_image, src=hist.new_ones(1).expand_as(flat_image))
    cum_hist = hist.cumsum(dim=-1)

    # The simplest form of lookup-table (LUT) that also achieves histogram equalization is
    # `lut = cum_hist / flat_image.shape[-1] * 255`
    # However, PIL uses a more elaborate scheme:
    # https://github.com/python-pillow/Pillow/blob/eb59cb61d5239ee69cbbf12709a0c6fd7314e6d7/src/PIL/ImageOps.py#L368-L385
    # `lut = ((cum_hist + num_non_max_pixels // (2 * 255)) // num_non_max_pixels) * 255`

    # The last non-zero element in the histogram is the first element in the cumulative histogram with the maximum
    # value. Thus, the "max" in `num_non_max_pixels` does not refer to 255 as the maximum value of uint8 images, but
    # rather the maximum value in the image, which might be or not be 255.
    index = cum_hist.argmax(dim=-1)
    num_non_max_pixels = flat_image.shape[-1] - hist.gather(dim=-1, index=index.unsqueeze_(-1))

    # This is performance optimization that saves us one multiplication later. With this, the LUT computation simplifies
    # to `lut = (cum_hist + step // 2) // step` and thus saving the final multiplication by 255 while keeping the
    # division count the same. PIL uses the variable name `step` for this, so we keep that for easier comparison.
    step = num_non_max_pixels.div_(255, rounding_mode="floor")

    # Although it looks like we could return early if we find `step == 0` like PIL does, that is unfortunately not as
    # easy due to our support for batched images. We can only return early if `(step == 0).all()` holds. If it doesn't,
    # we have to go through the computation below anyway. Since `step == 0` is an edge case anyway, it makes no sense to
    # pay the runtime cost for checking it every time.
    valid_equalization = step.ne(0).unsqueeze_(-1)

    # `lut[k]` is computed with `cum_hist[k-1]` with `lut[0] == (step // 2) // step == 0`. Thus, we perform the
    # computation only for `lut[1:]` with `cum_hist[:-1]` and add `lut[0] == 0` afterwards.
    cum_hist = cum_hist[..., :-1]
    (
        cum_hist.add_(step // 2)
        # We need the `clamp_`(min=1) call here to avoid zero division since they fail for integer dtypes. This has no
        # effect on the returned result of this kernel since images inside the batch with `step == 0` are returned as is
        # instead of equalized version.
        .div_(step.clamp_(min=1), rounding_mode="floor")
        # We need the `clamp_` call here since PILs LUT computation scheme can produce values outside the valid value
        # range of uint8 images
        .clamp_(0, 255)
    )
    lut = cum_hist.to(torch.uint8)
    lut = torch.cat([lut.new_zeros(1).expand(batch_shape + (1,)), lut], dim=-1)
    equalized_image = lut.gather(dim=-1, index=flat_image).view_as(image)

    output = torch.where(valid_equalization, equalized_image, image)
    return convert_dtype_image_tensor(output, output_dtype)


equalize_image_pil = _FP.equalize


def equalize_video(video: torch.Tensor) -> torch.Tensor:
    return equalize_image_tensor(video)


def equalize(inpt: datapoints._InputTypeJIT) -> datapoints._InputTypeJIT:
    if not torch.jit.is_scripting():
        _log_api_usage_once(equalize)

    if torch.jit.is_scripting() or is_simple_tensor(inpt):
        return equalize_image_tensor(inpt)
    elif isinstance(inpt, datapoints._datapoint.Datapoint):
        return inpt.equalize()
    elif isinstance(inpt, PIL.Image.Image):
        return equalize_image_pil(inpt)
    else:
        raise TypeError(
            f"Input can either be a plain tensor, any TorchVision datapoint, or a PIL image, "
            f"but got {type(inpt)} instead."
        )


def invert_image_tensor(image: torch.Tensor) -> torch.Tensor:
    if image.is_floating_point():
        return 1.0 - image
    elif image.dtype == torch.uint8:
        return image.bitwise_not()
    else:  # signed integer dtypes
        # We can't use `Tensor.bitwise_not` here, since we want to retain the leading zero bit that encodes the sign
        return image.bitwise_xor((1 << _num_value_bits(image.dtype)) - 1)


invert_image_pil = _FP.invert


def invert_video(video: torch.Tensor) -> torch.Tensor:
    return invert_image_tensor(video)


def invert(inpt: datapoints._InputTypeJIT) -> datapoints._InputTypeJIT:
    if not torch.jit.is_scripting():
        _log_api_usage_once(invert)

    if torch.jit.is_scripting() or is_simple_tensor(inpt):
        return invert_image_tensor(inpt)
    elif isinstance(inpt, datapoints._datapoint.Datapoint):
        return inpt.invert()
    elif isinstance(inpt, PIL.Image.Image):
        return invert_image_pil(inpt)
    else:
        raise TypeError(
            f"Input can either be a plain tensor, any TorchVision datapoint, or a PIL image, "
            f"but got {type(inpt)} instead."
        )
