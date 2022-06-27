from typing import Any

import PIL.Image
import torch
from torchvision.prototype import features
from torchvision.transforms import functional_tensor as _FT, functional_pil as _FP


adjust_brightness_image_tensor = _FT.adjust_brightness
adjust_brightness_image_pil = _FP.adjust_brightness


def adjust_brightness(inpt: Any, brightness_factor: float) -> Any:
    if isinstance(inpt, features._Feature):
        return inpt.adjust_brightness(brightness_factor=brightness_factor)
    elif isinstance(inpt, PIL.Image.Image):
        return adjust_brightness_image_pil(inpt, brightness_factor=brightness_factor)
    elif isinstance(inpt, torch.Tensor):
        return adjust_brightness_image_tensor(inpt, brightness_factor=brightness_factor)
    else:
        return inpt


adjust_saturation_image_tensor = _FT.adjust_saturation
adjust_saturation_image_pil = _FP.adjust_saturation


def adjust_saturation(inpt: Any, saturation_factor: float) -> Any:
    if isinstance(inpt, features._Feature):
        return inpt.adjust_saturation(saturation_factor=saturation_factor)
    elif isinstance(inpt, PIL.Image.Image):
        return adjust_saturation_image_pil(inpt, saturation_factor=saturation_factor)
    elif isinstance(inpt, torch.Tensor):
        return adjust_saturation_image_tensor(inpt, saturation_factor=saturation_factor)
    else:
        return inpt


adjust_contrast_image_tensor = _FT.adjust_contrast
adjust_contrast_image_pil = _FP.adjust_contrast


def adjust_contrast(inpt: Any, contrast_factor: float) -> Any:
    if isinstance(inpt, features._Feature):
        return inpt.adjust_contrast(contrast_factor=contrast_factor)
    elif isinstance(inpt, PIL.Image.Image):
        return adjust_contrast_image_pil(inpt, contrast_factor=contrast_factor)
    elif isinstance(inpt, torch.Tensor):
        return adjust_contrast_image_tensor(inpt, contrast_factor=contrast_factor)
    else:
        return inpt


adjust_sharpness_image_tensor = _FT.adjust_sharpness
adjust_sharpness_image_pil = _FP.adjust_sharpness


def adjust_sharpness(inpt: Any, sharpness_factor: float) -> Any:
    if isinstance(inpt, features._Feature):
        return inpt.adjust_sharpness(sharpness_factor=sharpness_factor)
    elif isinstance(inpt, PIL.Image.Image):
        return adjust_sharpness_image_pil(inpt, sharpness_factor=sharpness_factor)
    elif isinstance(inpt, torch.Tensor):
        return adjust_sharpness_image_tensor(inpt, sharpness_factor=sharpness_factor)
    else:
        return inpt


adjust_hue_image_tensor = _FT.adjust_hue
adjust_hue_image_pil = _FP.adjust_hue


def adjust_hue(inpt: Any, hue_factor: float) -> Any:
    if isinstance(inpt, features._Feature):
        return inpt.adjust_hue(hue_factor=hue_factor)
    elif isinstance(inpt, PIL.Image.Image):
        return adjust_hue_image_pil(inpt, hue_factor=hue_factor)
    elif isinstance(inpt, torch.Tensor):
        return adjust_hue_image_tensor(inpt, hue_factor=hue_factor)
    else:
        return inpt


adjust_gamma_image_tensor = _FT.adjust_gamma
adjust_gamma_image_pil = _FP.adjust_gamma


def adjust_gamma(inpt: Any, gamma: float, gain: float = 1) -> Any:
    if isinstance(inpt, features._Feature):
        return inpt.adjust_gamma(gamma=gamma, gain=gain)
    elif isinstance(inpt, PIL.Image.Image):
        return adjust_gamma_image_pil(inpt, gamma=gamma, gain=gain)
    elif isinstance(inpt, torch.Tensor):
        return adjust_gamma_image_tensor(inpt, gamma=gamma, gain=gain)
    else:
        return inpt


posterize_image_tensor = _FT.posterize
posterize_image_pil = _FP.posterize


def posterize(inpt: Any, bits: int) -> Any:
    if isinstance(inpt, features._Feature):
        return inpt.posterize(bits=bits)
    elif isinstance(inpt, PIL.Image.Image):
        return posterize_image_pil(inpt, bits=bits)
    elif isinstance(inpt, torch.Tensor):
        return posterize_image_tensor(inpt, bits=bits)
    else:
        return inpt


solarize_image_tensor = _FT.solarize
solarize_image_pil = _FP.solarize


def solarize(inpt: Any, threshold: float) -> Any:
    if isinstance(inpt, features._Feature):
        return inpt.solarize(threshold=threshold)
    elif isinstance(inpt, PIL.Image.Image):
        return solarize_image_pil(inpt, threshold=threshold)
    elif isinstance(inpt, torch.Tensor):
        return solarize_image_tensor(inpt, threshold=threshold)
    else:
        return inpt


autocontrast_image_tensor = _FT.autocontrast
autocontrast_image_pil = _FP.autocontrast


def autocontrast(inpt: Any) -> Any:
    if isinstance(inpt, features._Feature):
        return inpt.autocontrast()
    elif isinstance(inpt, PIL.Image.Image):
        return autocontrast_image_pil(inpt)
    elif isinstance(inpt, torch.Tensor):
        return autocontrast_image_tensor(inpt)
    else:
        return inpt


equalize_image_tensor = _FT.equalize
equalize_image_pil = _FP.equalize


def equalize(inpt: Any) -> Any:
    if isinstance(inpt, features._Feature):
        return inpt.equalize()
    elif isinstance(inpt, PIL.Image.Image):
        return equalize_image_pil(inpt)
    elif isinstance(inpt, torch.Tensor):
        return equalize_image_tensor(inpt)
    else:
        return inpt


invert_image_tensor = _FT.invert
invert_image_pil = _FP.invert


def invert(inpt: Any) -> Any:
    if isinstance(inpt, features._Feature):
        return inpt.invert()
    elif isinstance(inpt, PIL.Image.Image):
        return invert_image_pil(inpt)
    elif isinstance(inpt, torch.Tensor):
        return invert_image_tensor(inpt)
    else:
        return inpt
