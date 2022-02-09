from typing import TypeVar

from torchvision.prototype import features
from torchvision.prototype.transforms import kernels as K

from ._utils import dispatch

T = TypeVar("T", bound=features.Feature)


@dispatch(
    {
        features.Image: K.adjust_brightness_image,
    }
)
def adjust_brightness(input: T, *, brightness_factor: float) -> T:
    """ADDME"""
    ...


@dispatch(
    {
        features.Image: K.adjust_saturation_image,
    }
)
def adjust_saturation(input: T, *, saturation_factor: float) -> T:
    """ADDME"""
    ...


@dispatch(
    {
        features.Image: K.adjust_contrast_image,
    }
)
def adjust_contrast(input: T, *, contrast_factor: float) -> T:
    """ADDME"""
    ...


@dispatch(
    {
        features.Image: K.adjust_sharpness_image,
    }
)
def adjust_sharpness(input: T, *, sharpness_factor: float) -> T:
    """ADDME"""
    ...


@dispatch(
    {
        features.Image: K.posterize_image,
    }
)
def posterize(input: T, *, bits: int) -> T:
    """ADDME"""
    ...


@dispatch(
    {
        features.Image: K.solarize_image,
    }
)
def solarize(input: T, *, threshold: float) -> T:
    """ADDME"""
    ...


@dispatch(
    {
        features.Image: K.autocontrast_image,
    }
)
def autocontrast(input: T) -> T:
    """ADDME"""
    ...


@dispatch(
    {
        features.Image: K.equalize_image,
    }
)
def equalize(input: T) -> T:
    """ADDME"""
    ...


@dispatch(
    {
        features.Image: K.invert_image,
    }
)
def invert(input: T) -> T:
    """ADDME"""
    ...
