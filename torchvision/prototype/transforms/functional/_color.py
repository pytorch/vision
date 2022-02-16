from typing import Any

import PIL.Image
import torch
from torchvision.prototype import features
from torchvision.prototype.transforms import kernels as K
from torchvision.transforms import functional as _F

from ._utils import dispatch


@dispatch(
    {
        torch.Tensor: _F.adjust_brightness,
        PIL.Image.Image: _F.adjust_brightness,
        features.Image: K.adjust_brightness_image,
    }
)
def adjust_brightness(input: Any, *args: Any, **kwargs: Any) -> Any:
    """TODO: add docstring"""
    ...


@dispatch(
    {
        torch.Tensor: _F.adjust_saturation,
        PIL.Image.Image: _F.adjust_saturation,
        features.Image: K.adjust_saturation_image,
    }
)
def adjust_saturation(input: Any, *args: Any, **kwargs: Any) -> Any:
    """TODO: add docstring"""
    ...


@dispatch(
    {
        torch.Tensor: _F.adjust_contrast,
        PIL.Image.Image: _F.adjust_contrast,
        features.Image: K.adjust_contrast_image,
    }
)
def adjust_contrast(input: Any, *args: Any, **kwargs: Any) -> Any:
    """TODO: add docstring"""
    ...


@dispatch(
    {
        torch.Tensor: _F.adjust_sharpness,
        PIL.Image.Image: _F.adjust_sharpness,
        features.Image: K.adjust_sharpness_image,
    }
)
def adjust_sharpness(input: Any, *args: Any, **kwargs: Any) -> Any:
    """TODO: add docstring"""
    ...


@dispatch(
    {
        torch.Tensor: _F.posterize,
        PIL.Image.Image: _F.posterize,
        features.Image: K.posterize_image,
    }
)
def posterize(input: Any, *args: Any, **kwargs: Any) -> Any:
    """TODO: add docstring"""
    ...


@dispatch(
    {
        torch.Tensor: _F.solarize,
        PIL.Image.Image: _F.solarize,
        features.Image: K.solarize_image,
    }
)
def solarize(input: Any, *args: Any, **kwargs: Any) -> Any:
    """TODO: add docstring"""
    ...


@dispatch(
    {
        torch.Tensor: _F.autocontrast,
        PIL.Image.Image: _F.autocontrast,
        features.Image: K.autocontrast_image,
    }
)
def autocontrast(input: Any, *args: Any, **kwargs: Any) -> Any:
    """TODO: add docstring"""
    ...


@dispatch(
    {
        torch.Tensor: _F.equalize,
        PIL.Image.Image: _F.equalize,
        features.Image: K.equalize_image,
    }
)
def equalize(input: Any, *args: Any, **kwargs: Any) -> Any:
    """TODO: add docstring"""
    ...


@dispatch(
    {
        torch.Tensor: _F.invert,
        PIL.Image.Image: _F.invert,
        features.Image: K.invert_image,
    }
)
def invert(input: Any, *args: Any, **kwargs: Any) -> Any:
    """TODO: add docstring"""
    ...


@dispatch(
    {
        torch.Tensor: _F.adjust_hue,
        PIL.Image.Image: _F.adjust_hue,
        features.Image: K.adjust_hue_image,
    }
)
def adjust_hue(input: Any, *args: Any, **kwargs: Any) -> Any:
    """TODO: add docstring"""
    ...


@dispatch(
    {
        torch.Tensor: _F.adjust_gamma,
        PIL.Image.Image: _F.adjust_gamma,
        features.Image: K.adjust_gamma_image,
    }
)
def adjust_gamma(input: Any, *args: Any, **kwargs: Any) -> Any:
    """TODO: add docstring"""
    ...
