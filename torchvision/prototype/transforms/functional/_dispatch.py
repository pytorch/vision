# THIS FILE IS auto-generated!!

from typing import Any, Tuple, TypeVar

import torch
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
    "decode_image",
]


@F.utils.dispatches
def horizontal_flip(input: T) -> T:
    """ADDME"""
    pass


@F.utils.implements(horizontal_flip, features.Image)
def _horizontal_flip_image(input: features.Image) -> features.Image:
    converted_input = input.data

    output = F.horizontal_flip_image(converted_input)

    return features.Image(output, like=input)


@F.utils.implements(horizontal_flip, features.BoundingBox)
def _horizontal_flip_bounding_box(input: features.BoundingBox) -> features.BoundingBox:
    converted_input = input.data

    intermediate_format = features.BoundingBoxFormat.XYXY
    converted_input = F.convert_format_bounding_box(
        converted_input, old_format=input.format, new_format=intermediate_format
    )

    output = F.horizontal_flip_bounding_box(converted_input, image_size=input.image_size)
    output = F.convert_format_bounding_box(output, old_format=intermediate_format, new_format=input.format)

    return features.BoundingBox(output, like=input)


@F.utils.dispatches
def resize(
    input: T,
    *,
    size: Tuple[int, int],
    interpolation_mode: str = FEATURE_SPECIFIC_DEFAULT,  # type: ignore[assignment]
) -> T:
    """ADDME"""
    pass


@F.utils.implements(resize, features.Image)
def _resize_image(
    input: features.Image, *, size: Tuple[int, int], interpolation_mode: str = "bilinear"
) -> features.Image:
    converted_input = input.data

    output = F.resize_image(converted_input, size=size, interpolation_mode=interpolation_mode)

    return features.Image(output, like=input)


@F.utils.implements(resize, features.BoundingBox)
def _resize_bounding_box(input: features.BoundingBox, *, size: Tuple[int, int], **_: Any) -> features.BoundingBox:
    converted_input = input.data

    intermediate_format = features.BoundingBoxFormat.XYXY
    converted_input = F.convert_format_bounding_box(
        converted_input, old_format=input.format, new_format=intermediate_format
    )

    output = F.resize_bounding_box(converted_input, old_image_size=input.image_size, new_image_size=size)
    output = F.convert_format_bounding_box(output, old_format=intermediate_format, new_format=input.format)

    return features.BoundingBox(output, like=input, image_size=size)


@F.utils.implements(resize, features.SegmentationMask)
def _resize_segmentation_mask(
    input: features.SegmentationMask, *, size: Tuple[int, int], interpolation_mode: str = "nearest"
) -> features.SegmentationMask:
    converted_input = input.data

    output = F.resize_segmentation_mask(converted_input, size=size, interpolation_mode=interpolation_mode)

    return features.SegmentationMask(output, like=input)


@F.utils.dispatches
def convert_format(input: T, *, new_format: features.BoundingBoxFormat) -> T:
    """ADDME"""
    pass


@F.utils.implements(convert_format, features.BoundingBox)
def _convert_format_bounding_box(
    input: features.BoundingBox, *, new_format: features.BoundingBoxFormat
) -> features.BoundingBox:
    converted_input = input.data

    output = F.convert_format_bounding_box(converted_input, old_format=input.format, new_format=new_format)

    return features.BoundingBox(output, like=input, format=new_format)


@F.utils.dispatches
def convert_dtype(input: T, *, new_dtype: torch.dtype = torch.float32) -> T:
    """ADDME"""
    pass


@F.utils.implements(convert_dtype, features.Image)
def _convert_dtype_image(input: features.Image, *, new_dtype: torch.dtype = torch.float32) -> features.Image:
    converted_input = input.data

    output = F.convert_dtype_image(converted_input, new_dtype=new_dtype)

    return features.Image(output, like=input, dtype=new_dtype)


@F.utils.dispatches
def decode_image(input: T) -> T:
    """ADDME"""
    pass


@F.utils.implements(decode_image, features.EncodedImage)
def _decode_image_encoded_image(input: features.EncodedImage) -> features.Image:
    converted_input = input.data

    output = F.decode_image_with_pil(converted_input)

    return features.Image(output)
