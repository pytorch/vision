# THIS FILE IS auto-generated!!

from typing import Any, TypeVar, List, Optional

import torch
from torchvision import transforms
from torchvision.prototype import features

from .. import functional as F

# TODO: add explanation
# just a sentinel to have a default argument for parameters that have different default for features
# the actual value is not used
FEATURE_SPECIFIC_DEFAULT = object()

T = TypeVar("T", bound=features.Feature)


__all__ = [
    "horizontal_flip",
    "resize",
    "convert_format",
    "convert_dtype",
    "center_crop",
    "normalize",
    "resized_crop",
    "erase",
]


@F.utils.Dispatcher
def horizontal_flip(input: T) -> T:
    """ADDME"""
    pass


@horizontal_flip.implements(features.Image)
def _horizontal_flip_image(input: features.Image) -> features.Image:
    converted_input = input.data

    output = F.horizontal_flip_image(converted_input)

    return features.Image.new_like(input, output)


@horizontal_flip.implements(features.BoundingBox)
def _horizontal_flip_bounding_box(input: features.BoundingBox) -> features.BoundingBox:
    converted_input = input.data

    intermediate_format = features.BoundingBoxFormat.XYXY
    converted_input = F.convert_format_bounding_box(
        converted_input, old_format=input.format, new_format=intermediate_format
    )

    output = F.horizontal_flip_bounding_box(converted_input, image_size=input.image_size)
    output = F.convert_format_bounding_box(output, old_format=intermediate_format, new_format=input.format)

    return features.BoundingBox.new_like(input, output)


@F.utils.Dispatcher
def resize(
    input: T,
    *,
    size: List[int],
    interpolation: str = FEATURE_SPECIFIC_DEFAULT,  # type: ignore[assignment]
    max_size: Optional[int] = None,
    antialias: Optional[bool] = None,
) -> T:
    """ADDME"""
    pass


@resize.implements(features.Image)
def _resize_image(
    input: features.Image,
    *,
    size: List[int],
    interpolation: str = "bilinear",
    max_size: Optional[int] = None,
    antialias: Optional[bool] = None,
) -> features.Image:
    converted_input = input.data

    output = F.resize_image(
        converted_input, size=size, interpolation=interpolation, max_size=max_size, antialias=antialias
    )

    return features.Image.new_like(input, output)


@resize.implements(features.BoundingBox)
def _resize_bounding_box(input: features.BoundingBox, *, size: List[int], **_: Any) -> features.BoundingBox:
    converted_input = input.data

    intermediate_format = features.BoundingBoxFormat.XYXY
    converted_input = F.convert_format_bounding_box(
        converted_input, old_format=input.format, new_format=intermediate_format
    )

    output = F.resize_bounding_box(converted_input, old_image_size=input.image_size, new_image_size=size)
    output = F.convert_format_bounding_box(output, old_format=intermediate_format, new_format=input.format)

    return features.BoundingBox.new_like(input, output, image_size=size)


@resize.implements(features.SegmentationMask)
def _resize_segmentation_mask(
    input: features.SegmentationMask,
    *,
    size: List[int],
    interpolation: str = "nearest",
    max_size: Optional[int] = None,
    antialias: Optional[bool] = None,
) -> features.SegmentationMask:
    converted_input = input.data

    output = F.resize_segmentation_mask(
        converted_input, size=size, interpolation=interpolation, max_size=max_size, antialias=antialias
    )

    return features.SegmentationMask.new_like(input, output)


@F.utils.Dispatcher
def convert_format(input: T, *, new_format: features.BoundingBoxFormat) -> T:
    """ADDME"""
    pass


@convert_format.implements(features.BoundingBox)
def _convert_format_bounding_box(
    input: features.BoundingBox, *, new_format: features.BoundingBoxFormat
) -> features.BoundingBox:
    converted_input = input.data

    output = F.convert_format_bounding_box(converted_input, old_format=input.format, new_format=new_format)

    return features.BoundingBox.new_like(input, output, format=new_format)


@F.utils.Dispatcher
def convert_dtype(input: T, *, new_dtype: torch.dtype = torch.float32) -> T:
    """ADDME"""
    pass


@convert_dtype.implements(features.Image)
def _convert_dtype_image(input: features.Image, *, new_dtype: torch.dtype = torch.float32) -> features.Image:
    converted_input = input.data

    output = F.convert_dtype_image(converted_input, new_dtype=new_dtype)

    return features.Image.new_like(input, output, dtype=new_dtype)


@F.utils.Dispatcher
def center_crop(input: T, *, output_size: List[int]) -> T:
    """ADDME"""
    pass


@center_crop.implements(features.Image)
def _center_crop_image(input: features.Image, *, output_size: List[int]) -> features.Image:
    converted_input = input.data

    output = F.center_crop_image(converted_input, output_size=output_size)

    return features.Image.new_like(input, output)


@F.utils.Dispatcher
def normalize(input: T, *, mean: List[float], std: List[float], inplace: bool = False) -> T:
    """ADDME"""
    pass


@normalize.implements(features.Image)
def _normalize_image(
    input: features.Image, *, mean: List[float], std: List[float], inplace: bool = False
) -> features.Image:
    converted_input = input.data

    output = F.normalize_image(converted_input, mean=mean, std=std, inplace=inplace)

    return features.Image.new_like(input, output)


@F.utils.Dispatcher
def resized_crop(
    input: T,
    *,
    top: int,
    left: int,
    height: int,
    width: int,
    size: List[int],
    interpolation: transforms.InterpolationMode = transforms.InterpolationMode.BILINEAR,
) -> T:
    """ADDME"""
    pass


@resized_crop.implements(features.Image)
def _resized_crop_image(
    input: features.Image,
    *,
    top: int,
    left: int,
    height: int,
    width: int,
    size: List[int],
    interpolation: transforms.InterpolationMode = transforms.InterpolationMode.BILINEAR,
) -> features.Image:
    converted_input = input.data

    output = F.resized_crop_image(
        converted_input, top=top, left=left, height=height, width=width, size=size, interpolation=interpolation
    )

    return features.Image.new_like(input, output)


@F.utils.Dispatcher
def erase(input: T, *, i: int, j: int, h: int, w: int, v: torch.Tensor, inplace: bool = False) -> T:
    """ADDME"""
    pass


@erase.implements(features.Image)
def _erase_image(
    input: features.Image, *, i: int, j: int, h: int, w: int, v: torch.Tensor, inplace: bool = False
) -> features.Image:
    converted_input = input.data

    output = F.erase_image(converted_input, i=i, j=j, h=h, w=w, v=v, inplace=inplace)

    return features.Image.new_like(input, output)
