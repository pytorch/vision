from typing import TypeVar, Any

from torchvision.prototype import features
from torchvision.prototype.transforms import kernels as K

from ._utils import dispatch

T = TypeVar("T", bound=features.Feature)


@dispatch({features.Image: K.adjust_brightness_image})
def adjust_brightness(input: T, *args: Any, **kwargs: Any) -> T:
    """ADDME"""
    ...


@dispatch({features.Image: K.adjust_saturation_image})
def adjust_saturation(input: T, *args: Any, **kwargs: Any) -> T:
    """ADDME"""
    ...


@dispatch({features.Image: K.adjust_contrast_image})
def adjust_contrast(input: T, *args: Any, **kwargs: Any) -> T:
    """ADDME"""
    ...


@dispatch({features.Image: K.adjust_sharpness_image})
def adjust_sharpness(input: T, *args: Any, **kwargs: Any) -> T:
    """ADDME"""
    ...


@dispatch({features.Image: K.posterize_image})
def posterize(input: T, *args: Any, **kwargs: Any) -> T:
    """ADDME"""
    ...


@dispatch({features.Image: K.solarize_image})
def solarize(input: T, *args: Any, **kwargs: Any) -> T:
    """ADDME"""
    ...


@dispatch({features.Image: K.autocontrast_image})
def autocontrast(input: T, *args: Any, **kwargs: Any) -> T:
    """ADDME"""
    ...


@dispatch({features.Image: K.equalize_image})
def equalize(input: T, *args: Any, **kwargs: Any) -> T:
    """ADDME"""
    ...


@dispatch({features.Image: K.invert_image})
def invert(input: T, *args: Any, **kwargs: Any) -> T:
    """ADDME"""
    ...
