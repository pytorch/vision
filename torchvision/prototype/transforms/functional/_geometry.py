from typing import Any

import PIL.Image
import torch
from torchvision.prototype import features
from torchvision.prototype.transforms import kernels as K
from torchvision.transforms import functional as _F

from ._utils import dispatch


@dispatch(
    {
        torch.Tensor: _F.hflip,
        PIL.Image.Image: _F.hflip,
        features.Image: K.horizontal_flip_image,
        features.BoundingBox: None,
    },
)
def horizontal_flip(input: Any, *args: Any, **kwargs: Any) -> Any:
    """TODO: add docstring"""
    if isinstance(input, features.BoundingBox):
        output = K.horizontal_flip_bounding_box(input, format=input.format, image_size=input.image_size)
        return features.BoundingBox.new_like(input, output)

    raise RuntimeError


@dispatch(
    {
        torch.Tensor: _F.resize,
        PIL.Image.Image: _F.resize,
        features.Image: K.resize_image,
        features.SegmentationMask: K.resize_segmentation_mask,
        features.BoundingBox: None,
    }
)
def resize(input: Any, *args: Any, **kwargs: Any) -> Any:
    """TODO: add docstring"""
    if isinstance(input, features.BoundingBox):
        size = kwargs.pop("size")
        output = K.resize_bounding_box(input, size=size, image_size=input.image_size)
        return features.BoundingBox.new_like(input, output, image_size=size)

    raise RuntimeError


@dispatch(
    {
        torch.Tensor: _F.center_crop,
        PIL.Image.Image: _F.center_crop,
        features.Image: K.center_crop_image,
    }
)
def center_crop(input: Any, *args: Any, **kwargs: Any) -> Any:
    """TODO: add docstring"""
    ...


@dispatch(
    {
        torch.Tensor: _F.resized_crop,
        PIL.Image.Image: _F.resized_crop,
        features.Image: K.resized_crop_image,
    }
)
def resized_crop(input: Any, *args: Any, **kwargs: Any) -> Any:
    """TODO: add docstring"""
    ...


@dispatch(
    {
        torch.Tensor: _F.affine,
        PIL.Image.Image: _F.affine,
        features.Image: K.affine_image,
    }
)
def affine(input: Any, *args: Any, **kwargs: Any) -> Any:
    """TODO: add docstring"""
    ...


@dispatch(
    {
        torch.Tensor: _F.rotate,
        PIL.Image.Image: _F.rotate,
        features.Image: K.rotate_image,
    }
)
def rotate(input: Any, *args: Any, **kwargs: Any) -> Any:
    """TODO: add docstring"""
    ...


@dispatch(
    {
        torch.Tensor: _F.pad,
        PIL.Image.Image: _F.pad,
        features.Image: K.pad_image,
    }
)
def pad(input: Any, *args: Any, **kwargs: Any) -> Any:
    """TODO: add docstring"""
    ...


@dispatch(
    {
        torch.Tensor: _F.crop,
        PIL.Image.Image: _F.crop,
        features.Image: K.crop_image,
    }
)
def crop(input: Any, *args: Any, **kwargs: Any) -> Any:
    """TODO: add docstring"""
    ...


@dispatch(
    {
        torch.Tensor: _F.perspective,
        PIL.Image.Image: _F.perspective,
        features.Image: K.perspective_image,
    }
)
def perspective(input: Any, *args: Any, **kwargs: Any) -> Any:
    """TODO: add docstring"""
    ...


@dispatch(
    {
        torch.Tensor: _F.vflip,
        PIL.Image.Image: _F.vflip,
        features.Image: K.vertical_flip_image,
    }
)
def vertical_flip(input: Any, *args: Any, **kwargs: Any) -> Any:
    """TODO: add docstring"""
    ...


@dispatch(
    {
        torch.Tensor: _F.five_crop,
        PIL.Image.Image: _F.five_crop,
        features.Image: K.five_crop_image,
    }
)
def five_crop(input: Any, *args: Any, **kwargs: Any) -> Any:
    """TODO: add docstring"""
    ...


@dispatch(
    {
        torch.Tensor: _F.ten_crop,
        PIL.Image.Image: _F.ten_crop,
        features.Image: K.ten_crop_image,
    }
)
def ten_crop(input: Any, *args: Any, **kwargs: Any) -> Any:
    """TODO: add docstring"""
    ...
