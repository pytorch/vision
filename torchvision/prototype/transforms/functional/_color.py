from typing import TypeVar

from torchvision.prototype import features
from torchvision.transforms import functional as _F

from .utils import dispatch

T = TypeVar("T", bound=features.Feature)


adjust_brightness_image = _F.adjust_brightness


@dispatch
def adjust_brightness(input: T, *, brightness_factor: float) -> T:
    """ADDME"""
    pass


adjust_brightness.register(adjust_brightness_image, features.Image)


adjust_saturation_image = _F.adjust_saturation


@dispatch
def adjust_saturation(input: T, *, saturation_factor: float) -> T:
    """ADDME"""
    pass


adjust_saturation.register(adjust_saturation_image, features.Image)


adjust_contrast_image = _F.adjust_contrast


@dispatch
def adjust_contrast(input: T, *, contrast_factor: float) -> T:
    """ADDME"""
    pass


adjust_contrast.register(adjust_contrast_image, features.Image)


adjust_sharpness_image = _F.adjust_sharpness


@dispatch
def adjust_sharpness(input: T, *, sharpness_factor: float) -> T:
    """ADDME"""
    pass


adjust_sharpness.register(adjust_sharpness_image, features.Image)


posterize_image = _F.posterize


@dispatch
def posterize(input: T, *, bits: int) -> T:
    """ADDME"""
    pass


posterize.register(posterize_image, features.Image)


solarize_image = _F.solarize


@dispatch
def solarize(input: T, *, threshold: float) -> T:
    """ADDME"""
    pass


solarize.register(solarize_image, features.Image)


autocontrast_image = _F.autocontrast


@dispatch
def autocontrast(input: T) -> T:
    """ADDME"""
    pass


autocontrast.register(autocontrast_image, features.Image)


equalize_image = _F.equalize


@dispatch
def equalize(input: T) -> T:
    """ADDME"""
    pass


equalize.register(equalize_image, features.Image)


invert_image = _F.invert


@dispatch
def invert(input: T) -> T:
    """ADDME"""
    pass


invert.register(invert_image, features.Image)
