from typing import Tuple, List, Optional, TypeVar, Any

import torch
from torchvision.prototype import features
from torchvision.transforms import functional as _F, InterpolationMode

from ._meta_conversion import convert_bounding_box_format
from .utils import dispatch

T = TypeVar("T", bound=features.Feature)


@dispatch
def horizontal_flip(input: T) -> T:
    """ADDME"""
    pass


horizontal_flip_image = _F.hflip
horizontal_flip.register(horizontal_flip_image, features.Image)


def horizontal_flip_bounding_box(
    bounding_box: torch.Tensor, *, format: features.BoundingBoxFormat, image_size: Tuple[int, int]
) -> torch.Tensor:
    bounding_box = convert_bounding_box_format(
        bounding_box, old_format=format, new_format=features.BoundingBoxFormat.XYXY, copy=True
    )
    bounding_box[..., (0, 2)] = image_size[1] - bounding_box[..., (2, 0)]
    return convert_bounding_box_format(bounding_box, old_format=features.BoundingBoxFormat.XYXY, new_format=format)


def _horizontal_flip_bounding_box(input: features.BoundingBox) -> torch.Tensor:
    return horizontal_flip_bounding_box(input, format=input.format, image_size=input.image_size)


horizontal_flip.register(_horizontal_flip_bounding_box, features.BoundingBox)


@dispatch
def resize(
    input: T,
    *,
    size: List[int],
    interpolation: InterpolationMode = dispatch.FEATURE_SPECIFIC_DEFAULT,  # type: ignore[assignment]
    max_size: Optional[int] = None,
    antialias: Optional[bool] = None,
) -> T:
    """ADDME"""
    pass


def resize_image(
    image: torch.Tensor,
    size: List[int],
    interpolation: InterpolationMode = InterpolationMode.BILINEAR,
    max_size: Optional[int] = None,
    antialias: Optional[bool] = None,
) -> torch.Tensor:
    new_height, new_width = size
    num_channels, old_height, old_width = image.shape[-3:]
    batch_shape = image.shape[:-3]
    return _F.resize(
        image.reshape((-1, num_channels, old_height, old_width)),
        size=size,
        interpolation=interpolation,
        max_size=max_size,
        antialias=antialias,
    ).reshape(batch_shape + (num_channels, new_height, new_width))


resize.register(resize_image, features.Image, pil_kernel=_F.resize)


def resize_segmentation_mask(
    segmentation_mask: torch.Tensor,
    size: List[int],
    interpolation: InterpolationMode = InterpolationMode.NEAREST,
    max_size: Optional[int] = None,
    antialias: Optional[bool] = None,
) -> torch.Tensor:
    return resize_image(
        segmentation_mask, size=size, interpolation=interpolation, max_size=max_size, antialias=antialias
    )


resize.register(resize_segmentation_mask, features.SegmentationMask)


# TODO: handle max_size
def resize_bounding_box(
    bounding_box: torch.Tensor, *, old_image_size: List[int], new_image_size: List[int]
) -> torch.Tensor:
    old_height, old_width = old_image_size
    new_height, new_width = new_image_size
    ratios = torch.tensor((new_width / old_width, new_height / old_height))
    return bounding_box.view(-1, 2, 2).mul(ratios).view(bounding_box.shape)


def _resize_bounding_box(input: features.BoundingBox, *, size: List[int], **_: Any) -> features.BoundingBox:
    output = resize_bounding_box(input, old_image_size=list(input.image_size), new_image_size=size)
    return features.BoundingBox.new_like(input, output, image_size=size)


resize.register(_resize_bounding_box, features.BoundingBox, wrap_output=False)


@dispatch
def center_crop(input: T, *, output_size: List[int]) -> T:
    """ADDME"""
    pass


center_crop_image = _F.center_crop
center_crop.register(center_crop_image, features.Image)


@dispatch
def resized_crop(
    input: T,
    *,
    top: int,
    left: int,
    height: int,
    width: int,
    size: List[int],
    interpolation: InterpolationMode = InterpolationMode.BILINEAR,
) -> T:
    """ADDME"""
    pass


resized_crop_image = _F.resized_crop
resized_crop.register(resized_crop_image, features.Image)


@dispatch
def affine(
    input: T,
    *,
    angle: float,
    translate: List[int],
    scale: float,
    shear: List[float],
    interpolation: InterpolationMode = InterpolationMode.NEAREST,
    fill: Optional[List[float]] = None,
    resample: Optional[int] = None,
    fillcolor: Optional[List[float]] = None,
    center: Optional[List[int]] = None,
) -> T:
    """ADDME"""
    pass


affine_image = _F.affine
affine.register(affine_image, features.Image)


@dispatch
def rotate(
    input: T,
    *,
    angle: float,
    interpolation: InterpolationMode = InterpolationMode.NEAREST,
    expand: bool = False,
    center: Optional[List[int]] = None,
    fill: Optional[List[float]] = None,
    resample: Optional[int] = None,
) -> T:
    """ADDME"""
    pass


rotate_image = _F.rotate
rotate.register(rotate_image, features.Image)
