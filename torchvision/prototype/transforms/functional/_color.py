import torch
from torchvision.prototype import features
from torchvision.transforms import functional_pil as _FP, functional_tensor as _FT

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
    if sharpness_factor < 0:
        raise ValueError(f"sharpness_factor ({sharpness_factor}) is not non-negative.")

    _FT._assert_image_tensor(image)

    _FT._assert_channels(image, [1, 3])

    if image.numel() == 0 or any(dim <= 2 for dim in image.shape[-2:]):
        return image

    shape = image.shape

    if image.ndim > 4:
        image = image.view((-1,) + shape[-3:])
        needs_unsquash = True
    else:
        needs_unsquash = False

    output = _FT._blend(image, _FT._blurred_degenerate_image(image), sharpness_factor)

    if needs_unsquash:
        output = output.view(shape)

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
    _FT._assert_image_tensor(image)

    if image.dtype != torch.uint8:
        raise TypeError(f"Only torch.uint8 image tensors are supported, but found {image.dtype}")

    _FT._assert_channels(image, [1, 3])

    if image.numel() == 0:
        return image
    elif image.ndim == 2:
        return _FT._scale_channel(image)
    else:
        shape = image.shape
        channels = image.view((-1,) + shape[-2:]).unbind()
        return torch.stack([_FT._scale_channel(channel) for channel in channels]).view(shape)


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
