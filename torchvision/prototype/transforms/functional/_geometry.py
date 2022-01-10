from typing import Tuple

import torch
import torch.overrides
from torch.nn.functional import interpolate
from torchvision.prototype.features import BoundingBoxFormat

from ._meta_conversion import convert_format_bounding_box


def horizontal_flip_image(image: torch.Tensor) -> torch.Tensor:
    return image.flip((-1,))


def horizontal_flip_bounding_box(bounding_box: torch.Tensor, *, image_size: Tuple[int, int]) -> torch.Tensor:
    x, y, w, h = convert_format_bounding_box(
        bounding_box,
        old_format=BoundingBoxFormat.XYXY,
        new_format=BoundingBoxFormat.XYWH,
    ).unbind(-1)
    x = image_size[1] - (x + w)
    return convert_format_bounding_box(
        torch.stack((x, y, w, h), dim=-1),
        old_format=BoundingBoxFormat.XYWH,
        new_format=BoundingBoxFormat.XYXY,
    )


def resize_image(image: torch.Tensor, size: Tuple[int, int], interpolation_mode: str = "bilinear") -> torch.Tensor:
    return interpolate(image.view(-1, *image.shape[-3:]), size=size, mode=interpolation_mode).view(
        *image.shape[:-2], *size
    )


def resize_segmentation_mask(
    segmentation_mask: torch.Tensor,
    size: Tuple[int, int],
    interpolation_mode: str = "nearest",
) -> torch.Tensor:
    return resize_image(segmentation_mask, size=size, interpolation_mode=interpolation_mode)


def resize_bounding_box(
    bounding_box: torch.Tensor,
    *,
    old_image_size: Tuple[int, int],
    new_image_size: Tuple[int, int],
) -> torch.Tensor:
    old_height, old_width = old_image_size
    new_height, new_width = new_image_size
    return (
        bounding_box.view(-1, 2, 2)
        .mul(torch.tensor([new_width / old_width, new_height / old_height]))
        .view(bounding_box.shape)
    )
