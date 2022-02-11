from typing import TypeVar, Any, cast

import PIL.Image
import torch
from torchvision.prototype import features
from torchvision.prototype.transforms import kernels as K
from torchvision.transforms import functional as _F

from ._utils import dispatch

T = TypeVar("T", bound=features.Feature)


@dispatch(
    {
        torch.Tensor: _F.hflip,
        PIL.Image.Image: _F.hflip,
        features.Image: K.horizontal_flip_image,
        features.BoundingBox: None,
    },
)
def horizontal_flip(input: T, *args: Any, **kwargs: Any) -> T:
    """ADDME"""
    if isinstance(input, features.BoundingBox):
        output = K.horizontal_flip_bounding_box(input, format=input.format, image_size=input.image_size)
        return cast(T, features.BoundingBox.new_like(input, output))

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
def resize(input: T, *args: Any, **kwargs: Any) -> T:
    """ADDME"""
    if isinstance(input, features.BoundingBox):
        size = kwargs.pop("size")
        output = K.resize_bounding_box(input, size=size, image_size=input.image_size)
        return cast(T, features.BoundingBox.new_like(input, output, image_size=size))

    raise RuntimeError


@dispatch(
    {
        torch.Tensor: _F.center_crop,
        PIL.Image.Image: _F.center_crop,
        features.Image: K.center_crop_image,
    }
)
def center_crop(input: T, *args: Any, **kwargs: Any) -> T:
    """ADDME"""
    ...


@dispatch(
    {
        torch.Tensor: _F.resized_crop,
        PIL.Image.Image: _F.resized_crop,
        features.Image: K.resized_crop_image,
    }
)
def resized_crop(input: T, *args: Any, **kwargs: Any) -> T:
    """ADDME"""
    ...


@dispatch(
    {
        torch.Tensor: _F.affine,
        PIL.Image.Image: _F.affine,
        features.Image: K.affine_image,
    }
)
def affine(input: T, *args: Any, **kwargs: Any) -> T:
    """ADDME"""
    ...


@dispatch(
    {
        torch.Tensor: _F.rotate,
        PIL.Image.Image: _F.rotate,
        features.Image: K.rotate_image,
    }
)
def rotate(input: T, *args: Any, **kwargs: Any) -> T:
    """ADDME"""
    ...
