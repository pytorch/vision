from typing import Tuple, List, Optional, TypeVar

import torch
from torchvision.prototype import features
from torchvision.transforms import functional as _F, InterpolationMode

from ._meta_conversion import convert_bounding_box_format


T = TypeVar("T", bound=features._Feature)


horizontal_flip_image = _F.hflip


def horizontal_flip_bounding_box(
    bounding_box: torch.Tensor, *, format: features.BoundingBoxFormat, image_size: Tuple[int, int]
) -> torch.Tensor:
    shape = bounding_box.shape

    bounding_box = convert_bounding_box_format(
        bounding_box, old_format=format, new_format=features.BoundingBoxFormat.XYXY
    ).view(-1, 4)

    bounding_box[:, [0, 2]] = image_size[1] - bounding_box[:, [2, 0]]

    return convert_bounding_box_format(
        bounding_box, old_format=features.BoundingBoxFormat.XYXY, new_format=format
    ).view(shape)


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


def resize_segmentation_mask(
    segmentation_mask: torch.Tensor,
    size: List[int],
    max_size: Optional[int] = None,
) -> torch.Tensor:
    return resize_image(segmentation_mask, size=size, interpolation=InterpolationMode.NEAREST, max_size=max_size)


# TODO: handle max_size
def resize_bounding_box(bounding_box: torch.Tensor, *, size: List[int], image_size: Tuple[int, int]) -> torch.Tensor:
    old_height, old_width = image_size
    new_height, new_width = size
    ratios = torch.tensor((new_width / old_width, new_height / old_height), device=bounding_box.device)
    return bounding_box.view(-1, 2, 2).mul(ratios).view(bounding_box.shape)


center_crop_image = _F.center_crop
resized_crop_image = _F.resized_crop
affine_image = _F.affine
rotate_image = _F.rotate
pad_image = _F.pad
crop_image = _F.crop
perspective_image = _F.perspective
vertical_flip_image = _F.vflip
five_crop_image = _F.five_crop
ten_crop_image = _F.ten_crop
