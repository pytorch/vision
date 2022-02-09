from typing import TypeVar

from torchvision.prototype import features
from torchvision.transforms import functional as _F

from .utils import dispatch

T = TypeVar("T", bound=features.Feature)


adjust_brightness_image = _F.adjust_brightness


@dispatch(
    {
        features.Image: adjust_brightness_image,
    }
)
def adjust_brightness(input: T, *, brightness_factor: float) -> T:
    """ADDME"""
    pass


adjust_saturation_image = _F.adjust_saturation


@dispatch(
    {
        features.Image: adjust_saturation_image,
    }
)
def adjust_saturation(input: T, *, saturation_factor: float) -> T:
    """ADDME"""
    pass


adjust_contrast_image = _F.adjust_contrast


@dispatch(
    {
        features.Image: adjust_contrast_image,
    }
)
def adjust_contrast(input: T, *, contrast_factor: float) -> T:
    """ADDME"""
    pass


adjust_sharpness_image = _F.adjust_sharpness


@dispatch(
    {
        features.Image: adjust_sharpness_image,
    }
)
def adjust_sharpness(input: T, *, sharpness_factor: float) -> T:
    """ADDME"""
    pass


posterize_image = _F.posterize


@dispatch(
    {
        features.Image: posterize_image,
    }
)
def posterize(input: T, *, bits: int) -> T:
    """ADDME"""
    pass


solarize_image = _F.solarize


@dispatch(
    {
        features.Image: solarize_image,
    }
)
def solarize(input: T, *, threshold: float) -> T:
    """ADDME"""
    pass


autocontrast_image = _F.autocontrast


@dispatch(
    {
        features.Image: autocontrast_image,
    }
)
def autocontrast(input: T) -> T:
    """ADDME"""
    pass


equalize_image = _F.equalize


@dispatch(
    {
        features.Image: equalize_image,
    }
)
def equalize(input: T) -> T:
    """ADDME"""
    pass


invert_image = _F.invert


@dispatch(
    {
        features.Image: invert_image,
    }
)
def invert(input: T) -> T:
    """ADDME"""
    pass
