from typing import Tuple

import torch
from torchvision.prototype.constants import BoundingBoxFormat
from torchvision.transforms import functional_pil as _FP, functional_tensor as _FT

from ._meta import convert_bounding_box_format

horizontal_flip_image_tensor = _FT.hflip
horizontal_flip_image_pil = _FP.hflip


def horizontal_flip_segmentation_mask(segmentation_mask: torch.Tensor) -> torch.Tensor:
    return horizontal_flip_image_tensor(segmentation_mask)


def horizontal_flip_bounding_box(
    bounding_box: torch.Tensor, format: BoundingBoxFormat, image_size: Tuple[int, int]
) -> torch.Tensor:
    shape = bounding_box.shape

    bounding_box = convert_bounding_box_format(bounding_box, old_format=format, new_format=BoundingBoxFormat.XYXY).view(
        -1, 4
    )

    bounding_box[:, [0, 2]] = image_size[1] - bounding_box[:, [2, 0]]

    return convert_bounding_box_format(
        bounding_box, old_format=BoundingBoxFormat.XYXY, new_format=format, copy=False
    ).view(shape)
