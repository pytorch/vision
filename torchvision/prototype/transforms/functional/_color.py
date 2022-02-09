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

adjust_brightness.register(adjust_brightness_image, features.Image)


@dispatch
def adjust_saturation(input: T, *, saturation_factor: float) -> T:
    """ADDME"""
    pass


adjust_saturation_image = _F.adjust_saturation
adjust_saturation.register(adjust_saturation_image, features.Image)


@dispatch
def adjust_contrast(input: T, *, contrast_factor: float) -> T:
    """ADDME"""
    pass


adjust_contrast_image = _F.adjust_contrast
adjust_contrast.register(adjust_contrast_image, features.Image)


@dispatch
def adjust_sharpness(input: T, *, sharpness_factor: float) -> T:
    """ADDME"""
    pass


adjust_sharpness_image = _F.adjust_sharpness
adjust_sharpness.register(adjust_sharpness_image, features.Image)


@dispatch
def posterize(input: T, *, bits: int) -> T:
    """ADDME"""
    pass


posterize_image = _F.posterize
posterize.register(posterize_image, features.Image)


@dispatch
def solarize(input: T, *, threshold: float) -> T:
    """ADDME"""
    pass


solarize_image = _F.solarize
solarize.register(solarize_image, features.Image)


@dispatch
def autocontrast(input: T) -> T:
    """ADDME"""
    pass


autocontrast_image = _F.autocontrast
autocontrast.register(autocontrast_image, features.Image)


@dispatch
def equalize(input: T) -> T:
    """ADDME"""
    pass


equalize_image = _F.equalize
equalize.register(equalize_image, features.Image)


@dispatch
def invert(input: T) -> T:
    """ADDME"""
    pass


invert_image = _F.invert
invert.register(invert_image, features.Image)
