from typing import Tuple, List, Optional

import torch
from torchvision.prototype.features import BoundingBoxFormat
from torchvision.transforms import (  # noqa: F401
    functional as _F,
    InterpolationMode,
)

from ._meta_conversion import convert_bounding_box_format


def horizontal_flip_image(image: torch.Tensor) -> torch.Tensor:
    return image.flip((-1,))


def horizontal_flip_bounding_box(bounding_box: torch.Tensor, *, image_size: Tuple[int, int]) -> torch.Tensor:
    x, y, w, h = convert_bounding_box_format(
        bounding_box,
        old_format=BoundingBoxFormat.XYXY,
        new_format=BoundingBoxFormat.XYWH,
    ).unbind(-1)
    x = image_size[1] - (x + w)
    return convert_bounding_box_format(
        torch.stack((x, y, w, h), dim=-1),
        old_format=BoundingBoxFormat.XYWH,
        new_format=BoundingBoxFormat.XYXY,
    )


_resize_image = _F.resize


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
    return _resize_image(
        image.reshape((-1, num_channels, old_height, old_width)),
        size=size,
        interpolation=interpolation,
        max_size=max_size,
        antialias=antialias,
    ).reshape(batch_shape + (num_channels, new_height, new_width))


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


# TODO: handle max_size
def resize_bounding_box(
    bounding_box: torch.Tensor,
    *,
    old_image_size: List[int],
    new_image_size: List[int],
) -> torch.Tensor:
    old_height, old_width = old_image_size
    new_height, new_width = new_image_size
    return (
        bounding_box.view(-1, 2, 2)
        .mul(torch.tensor([new_width / old_width, new_height / old_height]))
        .view(bounding_box.shape)
    )


center_crop_image = _F.center_crop

resized_crop_image = _F.resized_crop

affine_image = _F.affine

rotate_image = _F.rotate
