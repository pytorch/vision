import torch
from torchvision.prototype import features
from torchvision.transforms import functional_pil as _FP, functional_tensor as _FT

from ._meta import _rgb_to_gray, convert_dtype_image_tensor, get_dimensions_image_tensor, get_num_channels_image_tensor


def _blend(image1: torch.Tensor, image2: torch.Tensor, ratio: float) -> torch.Tensor:
    ratio = float(ratio)
    fp = image1.is_floating_point()
    bound = 1.0 if fp else 255.0
    output = image1.mul(ratio).add_(image2, alpha=(1.0 - ratio)).clamp_(0, bound)
    return output if fp else output.to(image1.dtype)


def adjust_brightness_image_tensor(image: torch.Tensor, brightness_factor: float) -> torch.Tensor:
    if brightness_factor < 0:
        raise ValueError(f"brightness_factor ({brightness_factor}) is not non-negative.")

    _FT._assert_channels(image, [1, 3])

    fp = image.is_floating_point()
    bound = 1.0 if fp else 255.0
    output = image.mul(brightness_factor).clamp_(0, bound)
    return output if fp else output.to(image.dtype)


adjust_brightness_image_pil = _FP.adjust_brightness


def adjust_brightness_video(video: torch.Tensor, brightness_factor: float) -> torch.Tensor:
    return adjust_brightness_image_tensor(video, brightness_factor=brightness_factor)


def adjust_brightness(inpt: features.InputTypeJIT, brightness_factor: float) -> features.InputTypeJIT:
    if isinstance(inpt, torch.Tensor) and (torch.jit.is_scripting() or not isinstance(inpt, features._Feature)):
        return adjust_brightness_image_tensor(inpt, brightness_factor=brightness_factor)
    elif isinstance(inpt, features._Feature):
        return inpt.adjust_brightness(brightness_factor=brightness_factor)
    else:
        return adjust_brightness_image_pil(inpt, brightness_factor=brightness_factor)


def adjust_saturation_image_tensor(image: torch.Tensor, saturation_factor: float) -> torch.Tensor:
    if saturation_factor < 0:
        raise ValueError(f"saturation_factor ({saturation_factor}) is not non-negative.")

    c = get_num_channels_image_tensor(image)
    if c not in [1, 3]:
        raise TypeError(f"Input image tensor permitted channel values are {[1, 3]}, but found {c}")

    if c == 1:  # Match PIL behaviour
        return image

    return _blend(image, _rgb_to_gray(image), saturation_factor)


adjust_saturation_image_pil = _FP.adjust_saturation


def adjust_saturation_video(video: torch.Tensor, saturation_factor: float) -> torch.Tensor:
    return adjust_saturation_image_tensor(video, saturation_factor=saturation_factor)


def adjust_saturation(inpt: features.InputTypeJIT, saturation_factor: float) -> features.InputTypeJIT:
    if isinstance(inpt, torch.Tensor) and (torch.jit.is_scripting() or not isinstance(inpt, features._Feature)):
        return adjust_saturation_image_tensor(inpt, saturation_factor=saturation_factor)
    elif isinstance(inpt, features._Feature):
        return inpt.adjust_saturation(saturation_factor=saturation_factor)
    else:
        return adjust_saturation_image_pil(inpt, saturation_factor=saturation_factor)


def adjust_contrast_image_tensor(image: torch.Tensor, contrast_factor: float) -> torch.Tensor:
    if contrast_factor < 0:
        raise ValueError(f"contrast_factor ({contrast_factor}) is not non-negative.")

    c = get_num_channels_image_tensor(image)
    if c not in [1, 3]:
        raise TypeError(f"Input image tensor permitted channel values are {[1, 3]}, but found {c}")
    dtype = image.dtype if torch.is_floating_point(image) else torch.float32
    grayscale_image = _rgb_to_gray(image) if c == 3 else image
    mean = torch.mean(grayscale_image.to(dtype), dim=(-3, -2, -1), keepdim=True)
    return _blend(image, mean, contrast_factor)


adjust_contrast_image_pil = _FP.adjust_contrast


def adjust_contrast_video(video: torch.Tensor, contrast_factor: float) -> torch.Tensor:
    return adjust_contrast_image_tensor(video, contrast_factor=contrast_factor)


def adjust_contrast(inpt: features.InputTypeJIT, contrast_factor: float) -> features.InputTypeJIT:
    if isinstance(inpt, torch.Tensor) and (torch.jit.is_scripting() or not isinstance(inpt, features._Feature)):
        return adjust_contrast_image_tensor(inpt, contrast_factor=contrast_factor)
    elif isinstance(inpt, features._Feature):
        return inpt.adjust_contrast(contrast_factor=contrast_factor)
    else:
        return adjust_contrast_image_pil(inpt, contrast_factor=contrast_factor)


def adjust_sharpness_image_tensor(image: torch.Tensor, sharpness_factor: float) -> torch.Tensor:
    num_channels, height, width = get_dimensions_image_tensor(image)
    if num_channels not in (1, 3):
        raise TypeError(f"Input image tensor can have 1 or 3 channels, but found {num_channels}")

    if sharpness_factor < 0:
        raise ValueError(f"sharpness_factor ({sharpness_factor}) is not non-negative.")

    if image.numel() == 0 or height <= 2 or width <= 2:
        return image

    shape = image.shape

    if image.ndim > 4:
        image = image.reshape(-1, num_channels, height, width)
        needs_unsquash = True
    else:
        needs_unsquash = False

    output = _blend(image, _FT._blurred_degenerate_image(image), sharpness_factor)

    if needs_unsquash:
        output = output.reshape(shape)

    return output


adjust_sharpness_image_pil = _FP.adjust_sharpness


def adjust_sharpness_video(video: torch.Tensor, sharpness_factor: float) -> torch.Tensor:
    return adjust_sharpness_image_tensor(video, sharpness_factor=sharpness_factor)


def adjust_sharpness(inpt: features.InputTypeJIT, sharpness_factor: float) -> features.InputTypeJIT:
    if isinstance(inpt, torch.Tensor) and (torch.jit.is_scripting() or not isinstance(inpt, features._Feature)):
        return adjust_sharpness_image_tensor(inpt, sharpness_factor=sharpness_factor)
    elif isinstance(inpt, features._Feature):
        return inpt.adjust_sharpness(sharpness_factor=sharpness_factor)
    else:
        return adjust_sharpness_image_pil(inpt, sharpness_factor=sharpness_factor)


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
    # Instead of overwriting NaN afterwards, we just prevent it from occuring so
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
    mask_maxc_neq_g = ~mask_maxc_eq_g

    hr = (bc - gc).mul_(~mask_maxc_neq_r)
    hg = (2.0 + rc).sub_(bc).mul_(mask_maxc_eq_g & mask_maxc_neq_r)
    hb = (4.0 + gc).sub_(rc).mul_(mask_maxc_neq_g & mask_maxc_neq_r)

    h = hr.add_(hg).add_(hb)
    h = h.div_(6.0).add_(1.0).fmod_(1.0)
    return torch.stack((h, s, maxc), dim=-3)


def _hsv_to_rgb(img: torch.Tensor) -> torch.Tensor:
    h, s, v = img.unbind(dim=-3)
    h6 = h * 6
    i = torch.floor(h6)
    f = h6 - i
    i = i.to(dtype=torch.int32)

    p = (v * (1.0 - s)).clamp_(0.0, 1.0)
    q = (v * (1.0 - s * f)).clamp_(0.0, 1.0)
    t = (v * (1.0 - s * (1.0 - f))).clamp_(0.0, 1.0)
    i.remainder_(6)

    mask = i.unsqueeze(dim=-3) == torch.arange(6, device=i.device).view(-1, 1, 1)

    a1 = torch.stack((v, q, p, p, t, v), dim=-3)
    a2 = torch.stack((t, v, v, q, p, p), dim=-3)
    a3 = torch.stack((p, p, t, v, v, q), dim=-3)
    a4 = torch.stack((a1, a2, a3), dim=-4)

    return (a4.mul_(mask.to(dtype=img.dtype).unsqueeze(dim=-4))).sum(dim=-3)


def adjust_hue_image_tensor(image: torch.Tensor, hue_factor: float) -> torch.Tensor:
    if not (-0.5 <= hue_factor <= 0.5):
        raise ValueError(f"hue_factor ({hue_factor}) is not in [-0.5, 0.5].")

    c = get_num_channels_image_tensor(image)

    if c not in [1, 3]:
        raise TypeError(f"Input image tensor permitted channel values are {[1, 3]}, but found {c}")

    if c == 1:  # Match PIL behaviour
        return image

    if image.numel() == 0:
        # exit earlier on empty images
        return image

    orig_dtype = image.dtype
    if image.dtype == torch.uint8:
        image = image / 255.0

    image = _rgb_to_hsv(image)
    h, s, v = image.unbind(dim=-3)
    h.add_(hue_factor).remainder_(1.0)
    image = torch.stack((h, s, v), dim=-3)
    image_hue_adj = _hsv_to_rgb(image)

    if orig_dtype == torch.uint8:
        image_hue_adj = image_hue_adj.mul_(255.0).to(dtype=orig_dtype)

    return image_hue_adj


adjust_hue_image_pil = _FP.adjust_hue


def adjust_hue_video(video: torch.Tensor, hue_factor: float) -> torch.Tensor:
    return adjust_hue_image_tensor(video, hue_factor=hue_factor)


def adjust_hue(inpt: features.InputTypeJIT, hue_factor: float) -> features.InputTypeJIT:
    if isinstance(inpt, torch.Tensor) and (torch.jit.is_scripting() or not isinstance(inpt, features._Feature)):
        return adjust_hue_image_tensor(inpt, hue_factor=hue_factor)
    elif isinstance(inpt, features._Feature):
        return inpt.adjust_hue(hue_factor=hue_factor)
    else:
        return adjust_hue_image_pil(inpt, hue_factor=hue_factor)


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


def adjust_gamma(inpt: features.InputTypeJIT, gamma: float, gain: float = 1) -> features.InputTypeJIT:
    if isinstance(inpt, torch.Tensor) and (torch.jit.is_scripting() or not isinstance(inpt, features._Feature)):
        return adjust_gamma_image_tensor(inpt, gamma=gamma, gain=gain)
    elif isinstance(inpt, features._Feature):
        return inpt.adjust_gamma(gamma=gamma, gain=gain)
    else:
        return adjust_gamma_image_pil(inpt, gamma=gamma, gain=gain)


posterize_image_tensor = _FT.posterize
posterize_image_pil = _FP.posterize


def posterize_video(video: torch.Tensor, bits: int) -> torch.Tensor:
    return posterize_image_tensor(video, bits=bits)


def posterize(inpt: features.InputTypeJIT, bits: int) -> features.InputTypeJIT:
    if isinstance(inpt, torch.Tensor) and (torch.jit.is_scripting() or not isinstance(inpt, features._Feature)):
        return posterize_image_tensor(inpt, bits=bits)
    elif isinstance(inpt, features._Feature):
        return inpt.posterize(bits=bits)
    else:
        return posterize_image_pil(inpt, bits=bits)


def solarize_image_tensor(image: torch.Tensor, threshold: float) -> torch.Tensor:
    bound = 1 if image.is_floating_point() else 255
    if threshold > bound:
        raise TypeError(f"Threshold should be less or equal the maximum value of the dtype, but got {threshold}")

    return torch.where(image >= threshold, invert_image_tensor(image), image)


solarize_image_pil = _FP.solarize


def solarize_video(video: torch.Tensor, threshold: float) -> torch.Tensor:
    return solarize_image_tensor(video, threshold=threshold)


def solarize(inpt: features.InputTypeJIT, threshold: float) -> features.InputTypeJIT:
    if isinstance(inpt, torch.Tensor) and (torch.jit.is_scripting() or not isinstance(inpt, features._Feature)):
        return solarize_image_tensor(inpt, threshold=threshold)
    elif isinstance(inpt, features._Feature):
        return inpt.solarize(threshold=threshold)
    else:
        return solarize_image_pil(inpt, threshold=threshold)


def autocontrast_image_tensor(image: torch.Tensor) -> torch.Tensor:
    c = get_num_channels_image_tensor(image)

    if c not in [1, 3]:
        raise TypeError(f"Input image tensor permitted channel values are {[1, 3]}, but found {c}")

    if image.numel() == 0:
        # exit earlier on empty images
        return image

    bound = 1.0 if image.is_floating_point() else 255.0
    dtype = image.dtype if torch.is_floating_point(image) else torch.float32

    minimum = image.amin(dim=(-2, -1), keepdim=True).to(dtype)
    maximum = image.amax(dim=(-2, -1), keepdim=True).to(dtype)

    scale = bound / (maximum - minimum)
    eq_idxs = maximum == minimum
    minimum[eq_idxs] = 0.0
    scale[eq_idxs] = 1.0

    return (image - minimum).mul_(scale).clamp_(0, bound).to(image.dtype)


autocontrast_image_pil = _FP.autocontrast


def autocontrast_video(video: torch.Tensor) -> torch.Tensor:
    return autocontrast_image_tensor(video)


def autocontrast(inpt: features.InputTypeJIT) -> features.InputTypeJIT:
    if isinstance(inpt, torch.Tensor) and (torch.jit.is_scripting() or not isinstance(inpt, features._Feature)):
        return autocontrast_image_tensor(inpt)
    elif isinstance(inpt, features._Feature):
        return inpt.autocontrast()
    else:
        return autocontrast_image_pil(inpt)


def equalize_image_tensor(image: torch.Tensor) -> torch.Tensor:
    if image.dtype != torch.uint8:
        raise TypeError(f"Only torch.uint8 image tensors are supported, but found {image.dtype}")

    num_channels, height, width = get_dimensions_image_tensor(image)
    if num_channels not in (1, 3):
        raise TypeError(f"Input image tensor can have 1 or 3 channels, but found {num_channels}")

    if image.numel() == 0:
        return image

    batch_shape = image.shape[:-2]
    flat_image = image.flatten(start_dim=-2).to(torch.long)

    # The algorithm for histogram equalization is mirrored from PIL:
    # https://github.com/python-pillow/Pillow/blob/eb59cb61d5239ee69cbbf12709a0c6fd7314e6d7/src/PIL/ImageOps.py#L368-L385

    # Although PyTorch has builtin functionality for histograms, it doesn't support batches. Since we deal with uint8
    # images here and thus the values are already binned, the computation is trivial. The histogram is computed by using
    # the flattened image as index. For example, a pixel value of 127 in the image corresponds to adding 1 to index 127
    # in the histogram.
    hist = flat_image.new_zeros(batch_shape + (256,), dtype=torch.int32)
    hist.scatter_add_(dim=-1, index=flat_image, src=hist.new_ones(1).expand_as(flat_image))
    cum_hist = hist.cumsum(dim=-1)

    # The simplest form of lookup-table (LUT) that also achieves histogram equalization is
    # `lut = cum_hist / flat_image.shape[-1] * 255`
    # However, PIL uses a more elaborate scheme:
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
    no_equalization = step.eq(0).unsqueeze_(-1)

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

    return torch.where(no_equalization, image, equalized_image)


equalize_image_pil = _FP.equalize


def equalize_video(video: torch.Tensor) -> torch.Tensor:
    return equalize_image_tensor(video)


def equalize(inpt: features.InputTypeJIT) -> features.InputTypeJIT:
    if isinstance(inpt, torch.Tensor) and (torch.jit.is_scripting() or not isinstance(inpt, features._Feature)):
        return equalize_image_tensor(inpt)
    elif isinstance(inpt, features._Feature):
        return inpt.equalize()
    else:
        return equalize_image_pil(inpt)


def invert_image_tensor(image: torch.Tensor) -> torch.Tensor:
    if image.dtype == torch.uint8:
        return image.bitwise_not()
    else:
        return (1 if image.is_floating_point() else 255) - image  # type: ignore[no-any-return]


invert_image_pil = _FP.invert


def invert_video(video: torch.Tensor) -> torch.Tensor:
    return invert_image_tensor(video)


def invert(inpt: features.InputTypeJIT) -> features.InputTypeJIT:
    if isinstance(inpt, torch.Tensor) and (torch.jit.is_scripting() or not isinstance(inpt, features._Feature)):
        return invert_image_tensor(inpt)
    elif isinstance(inpt, features._Feature):
        return inpt.invert()
    else:
        return invert_image_pil(inpt)
