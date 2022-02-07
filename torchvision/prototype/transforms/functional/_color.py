from typing import TypeVar

from torchvision.prototype import features
from torchvision.transforms import functional as _F

from .utils import dispatch

T = TypeVar("T", bound=features.Feature)


@dispatch
def adjust_brightness(input: T, *, brightness_factor: float) -> T:
    """ADDME"""
    pass


adjust_brightness_image = _F.adjust_brightness

adjust_brightness.register(features.Image, adjust_brightness_image)


@dispatch
def adjust_saturation(input: T, *, saturation_factor: float) -> T:
    """ADDME"""
    pass


adjust_saturation_image = _F.adjust_saturation
adjust_saturation.register(features.Image, adjust_saturation_image)


@dispatch
def adjust_contrast(input: T, *, contrast_factor: float) -> T:
    """ADDME"""
    pass


adjust_contrast_image = _F.adjust_contrast
adjust_contrast.register(features.Image, adjust_contrast_image)


@dispatch
def adjust_sharpness(input: T, *, sharpness_factor: float) -> T:
    """ADDME"""
    pass


adjust_sharpness_image = _F.adjust_sharpness
adjust_sharpness.register(features.Image, adjust_sharpness_image)


@dispatch
def posterize(input: T, *, bits: int) -> T:
    """ADDME"""
    pass


posterize_image = _F.posterize
posterize.register(features.Image, posterize_image)


@dispatch
def solarize(input: T, *, threshold: float) -> T:
    """ADDME"""
    pass


solarize_image = _F.solarize
solarize.register(features.Image, solarize_image)


@dispatch
def autocontrast(input: T) -> T:
    """ADDME"""
    pass


autocontrast_image = _F.autocontrast
autocontrast.register(features.Image, autocontrast_image)


@dispatch
def equalize(input: T) -> T:
    """ADDME"""
    pass


equalize_image = _F.equalize
equalize.register(features.Image, equalize_image)


@dispatch
def invert(input: T) -> T:
    """ADDME"""
    pass


invert_image = _F.invert
invert.register(features.Image, invert_image)
