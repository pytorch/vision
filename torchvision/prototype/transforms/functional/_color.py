from typing import Union

import PIL.Image
import torch
from torchvision.prototype import features
from torchvision.transforms import functional_pil as _FP, functional_tensor as _FT


# shortcut type
DType = Union[torch.Tensor, PIL.Image.Image, features._Feature]

adjust_brightness_image_tensor = _FT.adjust_brightness
adjust_brightness_image_pil = _FP.adjust_brightness


def adjust_brightness(inpt: DType, brightness_factor: float) -> DType:
    if isinstance(inpt, features._Feature):
        return inpt.adjust_brightness(brightness_factor=brightness_factor)
    elif isinstance(inpt, PIL.Image.Image):
        return adjust_brightness_image_pil(inpt, brightness_factor=brightness_factor)
    else:
        return adjust_brightness_image_tensor(inpt, brightness_factor=brightness_factor)


adjust_saturation_image_tensor = _FT.adjust_saturation
adjust_saturation_image_pil = _FP.adjust_saturation


def adjust_saturation(inpt: DType, saturation_factor: float) -> DType:
    if isinstance(inpt, features._Feature):
        return inpt.adjust_saturation(saturation_factor=saturation_factor)
    elif isinstance(inpt, PIL.Image.Image):
        return adjust_saturation_image_pil(inpt, saturation_factor=saturation_factor)
    else:
        return adjust_saturation_image_tensor(inpt, saturation_factor=saturation_factor)


adjust_contrast_image_tensor = _FT.adjust_contrast
adjust_contrast_image_pil = _FP.adjust_contrast


def adjust_contrast(inpt: DType, contrast_factor: float) -> DType:
    if isinstance(inpt, features._Feature):
        return inpt.adjust_contrast(contrast_factor=contrast_factor)
    elif isinstance(inpt, PIL.Image.Image):
        return adjust_contrast_image_pil(inpt, contrast_factor=contrast_factor)
    else:
        return adjust_contrast_image_tensor(inpt, contrast_factor=contrast_factor)


adjust_sharpness_image_tensor = _FT.adjust_sharpness
adjust_sharpness_image_pil = _FP.adjust_sharpness


def adjust_sharpness(inpt: DType, sharpness_factor: float) -> DType:
    if isinstance(inpt, features._Feature):
        return inpt.adjust_sharpness(sharpness_factor=sharpness_factor)
    elif isinstance(inpt, PIL.Image.Image):
        return adjust_sharpness_image_pil(inpt, sharpness_factor=sharpness_factor)
    else:
        return adjust_sharpness_image_tensor(inpt, sharpness_factor=sharpness_factor)


adjust_hue_image_tensor = _FT.adjust_hue
adjust_hue_image_pil = _FP.adjust_hue


def adjust_hue(inpt: DType, hue_factor: float) -> DType:
    if isinstance(inpt, features._Feature):
        return inpt.adjust_hue(hue_factor=hue_factor)
    elif isinstance(inpt, PIL.Image.Image):
        return adjust_hue_image_pil(inpt, hue_factor=hue_factor)
    else:
        return adjust_hue_image_tensor(inpt, hue_factor=hue_factor)


adjust_gamma_image_tensor = _FT.adjust_gamma
adjust_gamma_image_pil = _FP.adjust_gamma


def adjust_gamma(inpt: DType, gamma: float, gain: float = 1) -> DType:
    if isinstance(inpt, features._Feature):
        return inpt.adjust_gamma(gamma=gamma, gain=gain)
    elif isinstance(inpt, PIL.Image.Image):
        return adjust_gamma_image_pil(inpt, gamma=gamma, gain=gain)
    else:
        return adjust_gamma_image_tensor(inpt, gamma=gamma, gain=gain)


posterize_image_tensor = _FT.posterize
posterize_image_pil = _FP.posterize


def posterize(inpt: DType, bits: int) -> DType:
    if isinstance(inpt, features._Feature):
        return inpt.posterize(bits=bits)
    elif isinstance(inpt, PIL.Image.Image):
        return posterize_image_pil(inpt, bits=bits)
    else:
        return posterize_image_tensor(inpt, bits=bits)


solarize_image_tensor = _FT.solarize
solarize_image_pil = _FP.solarize


def solarize(inpt: DType, threshold: float) -> DType:
    if isinstance(inpt, features._Feature):
        return inpt.solarize(threshold=threshold)
    elif isinstance(inpt, PIL.Image.Image):
        return solarize_image_pil(inpt, threshold=threshold)
    else:
        return solarize_image_tensor(inpt, threshold=threshold)


autocontrast_image_tensor = _FT.autocontrast
autocontrast_image_pil = _FP.autocontrast


def autocontrast(inpt: DType) -> DType:
    if isinstance(inpt, features._Feature):
        return inpt.autocontrast()
    elif isinstance(inpt, PIL.Image.Image):
        return autocontrast_image_pil(inpt)
    else:
        return autocontrast_image_tensor(inpt)


equalize_image_tensor = _FT.equalize
equalize_image_pil = _FP.equalize


def equalize(inpt: DType) -> DType:
    if isinstance(inpt, features._Feature):
        return inpt.equalize()
    elif isinstance(inpt, PIL.Image.Image):
        return equalize_image_pil(inpt)
    else:
        return equalize_image_tensor(inpt)


invert_image_tensor = _FT.invert
invert_image_pil = _FP.invert


def invert(inpt: DType) -> DType:
    if isinstance(inpt, features._Feature):
        return inpt.invert()
    elif isinstance(inpt, PIL.Image.Image):
        return invert_image_pil(inpt)
    else:
        return invert_image_tensor(inpt)
