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

posterize_image_tensor = _FT.posterize
posterize_image_pil = _FP.posterize

solarize_image_tensor = _FT.solarize
solarize_image_pil = _FP.solarize

autocontrast_image_tensor = _FT.autocontrast
autocontrast_image_pil = _FP.autocontrast

equalize_image_tensor = _FT.equalize
equalize_image_pil = _FP.equalize

invert_image_tensor = _FT.invert
invert_image_pil = _FP.invert

adjust_hue_image_tensor = _FT.adjust_hue
adjust_hue_image_pil = _FP.adjust_hue

adjust_gamma_image_tensor = _FT.adjust_gamma
adjust_gamma_image_pil = _FP.adjust_gamma
