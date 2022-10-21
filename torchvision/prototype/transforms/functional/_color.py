import torch
from torchvision.prototype import features
from torchvision.transforms import functional_pil as _FP, functional_tensor as _FT

from ._meta import get_dimensions_image_tensor

adjust_brightness_image_tensor = _FT.adjust_brightness
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


adjust_saturation_image_tensor = _FT.adjust_saturation
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


adjust_contrast_image_tensor = _FT.adjust_contrast
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

    output = _FT._blend(image, _FT._blurred_degenerate_image(image), sharpness_factor)

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


adjust_hue_image_tensor = _FT.adjust_hue
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


adjust_gamma_image_tensor = _FT.adjust_gamma
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


solarize_image_tensor = _FT.solarize
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


autocontrast_image_tensor = _FT.autocontrast
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

    # The algorithm for histogram equalization is mirrored from PIL:
    # https://github.com/python-pillow/Pillow/blob/eb59cb61d5239ee69cbbf12709a0c6fd7314e6d7/src/PIL/ImageOps.py#L368-L385
    # Although PyTorch has builtin functionality for histograms, we are not using them here since they can only be
    # applied to the complete tensor. We need to apply them to the spatial size, i.e. the last two dimensions only, and
    # thus we would need to call them in a for loop, which is inefficient.
    batch_shape = image.shape[:-2]
    flat_image = image.flatten(start_dim=-2).to(torch.long)

    # The histogram is computed by using the flattened image as index. For example, a pixel value of 127 in the image
    # corresponds to adding 1 to index 127 in the histogram.
    hist = flat_image.new_zeros(batch_shape + (256,), dtype=torch.int32)
    hist.scatter_add_(dim=-1, index=flat_image, src=hist.new_ones(1).expand_as(flat_image))
    cum_hist = hist.cumsum(dim=-1)

    # The simplest form of lookup-table (LUT) that also achieves histogram equalization is
    # `lut = cum_hist / flat_image.shape[-1] * 255`
    # However, PIL uses a more elaborate scheme:
    # 1. Instead of normalizing the cumulative histogram by the number of pixels (` / flat_image.shape[-1]` above),
    #    it is normalized by the number of pixels that are don't have the maximum value. Note that maximum value here
    #    does not mean 255 per se but rather the maximum value in the image, which might be or not be 255.
    # 2. Instead of normalizing just the cumulative histogram, a constant based on the number of non-maximum is added
    #    to it.
    # This brings the computation of the  LUT to
    # `lut = ((cum_hist + num_non_max_pixels // (2 * 255)) // num_non_max_pixels) * 255`

    # The last non-zero element in the histogram is the first element in the cumulative histogram with the maximum value
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
    # computation only for `lut[1:]` and add `lut[0] == 0` afterwards.
    cum_hist = cum_hist[..., :-1]
    # We need the `step.clamp_`(min=1) call here to avoid zero division. This has no effect on the returned result of
    # this kernel since images inside the batch with `step == 0` are returned as is rather than their equalized version.
    cum_hist.add_(step // 2).div_(step.clamp_(min=1), rounding_mode="floor").clamp_(0, 255)
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


invert_image_tensor = _FT.invert
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
