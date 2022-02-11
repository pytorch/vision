from typing import TypeVar, Any

import PIL.Image
import torch
from torchvision.prototype import features
from torchvision.prototype.transforms import kernels as K
from torchvision.transforms import functional as _F

from ._utils import dispatch

T = TypeVar("T", bound=features._Feature)


@dispatch(
    {
        torch.Tensor: _F.adjust_brightness,
        PIL.Image.Image: _F.adjust_brightness,
        features.Image: K.adjust_brightness_image,
    }
)
def adjust_brightness(input: T, *args: Any, **kwargs: Any) -> T:
    """ADDME"""
    ...


@dispatch(
    {
        torch.Tensor: _F.adjust_saturation,
        PIL.Image.Image: _F.adjust_saturation,
        features.Image: K.adjust_saturation_image,
    }
)
def adjust_saturation(input: T, *args: Any, **kwargs: Any) -> T:
    """ADDME"""
    ...


@dispatch(
    {
        torch.Tensor: _F.adjust_contrast,
        PIL.Image.Image: _F.adjust_contrast,
        features.Image: K.adjust_contrast_image,
    }
)
def adjust_contrast(input: T, *args: Any, **kwargs: Any) -> T:
    """ADDME"""
    ...


@dispatch(
    {
        torch.Tensor: _F.adjust_sharpness,
        PIL.Image.Image: _F.adjust_sharpness,
        features.Image: K.adjust_sharpness_image,
    }
)
def adjust_sharpness(input: T, *args: Any, **kwargs: Any) -> T:
    """ADDME"""
    ...


@dispatch(
    {
        torch.Tensor: _F.posterize,
        PIL.Image.Image: _F.posterize,
        features.Image: K.posterize_image,
    }
)
def posterize(input: T, *args: Any, **kwargs: Any) -> T:
    """ADDME"""
    ...


@dispatch(
    {
        torch.Tensor: _F.solarize,
        PIL.Image.Image: _F.solarize,
        features.Image: K.solarize_image,
    }
)
def solarize(input: T, *args: Any, **kwargs: Any) -> T:
    """ADDME"""
    ...


@dispatch(
    {
        torch.Tensor: _F.autocontrast,
        PIL.Image.Image: _F.autocontrast,
        features.Image: K.autocontrast_image,
    }
)
def autocontrast(input: T, *args: Any, **kwargs: Any) -> T:
    """ADDME"""
    ...


@dispatch(
    {
        torch.Tensor: _F.equalize,
        PIL.Image.Image: _F.equalize,
        features.Image: K.equalize_image,
    }
)
def equalize(input: T, *args: Any, **kwargs: Any) -> T:
    """ADDME"""
    ...


@dispatch(
    {
        torch.Tensor: _F.invert,
        PIL.Image.Image: _F.invert,
        features.Image: K.invert_image,
    }
)
def invert(input: T, *args: Any, **kwargs: Any) -> T:
    """ADDME"""
    ...
