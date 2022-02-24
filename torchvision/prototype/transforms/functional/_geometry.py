from typing import Tuple, List, Optional

import PIL.Image
import torch
from torchvision.prototype import features
from torchvision.prototype.transforms.functional import get_image_size
from torchvision.transforms import functional_tensor as _FT, functional_pil as _FP

from ._meta_conversion import convert_bounding_box_format


horizontal_flip_image_tensor = _FT.hflip
horizontal_flip_image_pil = _FP.hflip


def horizontal_flip_bounding_box(
    bounding_box: torch.Tensor, format: features.BoundingBoxFormat, image_size: Tuple[int, int]
) -> torch.Tensor:
    shape = bounding_box.shape

    bounding_box = convert_bounding_box_format(
        bounding_box, old_format=format, new_format=features.BoundingBoxFormat.XYXY
    ).view(-1, 4)

    bounding_box[:, [0, 2]] = image_size[1] - bounding_box[:, [2, 0]]

    return convert_bounding_box_format(
        bounding_box, old_format=features.BoundingBoxFormat.XYXY, new_format=format
    ).view(shape)


def resize_image_tensor(
    image: torch.Tensor,
    size: List[int],
    interpolation: str = "nearest",
    max_size: Optional[int] = None,
    antialias: Optional[bool] = None,
) -> torch.Tensor:
    new_height, new_width = size
    num_channels, old_height, old_width = image.shape[-3:]
    batch_shape = image.shape[:-3]
    return _FT.resize(
        image.reshape((-1, num_channels, old_height, old_width)),
        size=size,
        interpolation=interpolation,
        max_size=max_size,
        antialias=antialias,
    ).reshape(batch_shape + (num_channels, new_height, new_width))


resize_image_pil = _FP.resize


def resize_segmentation_mask(
    segmentation_mask: torch.Tensor, size: List[int], max_size: Optional[int] = None
) -> torch.Tensor:
    return resize_image_tensor(segmentation_mask, size=size, interpolation="nearest", max_size=max_size)


# TODO: handle max_size
def resize_bounding_box(bounding_box: torch.Tensor, size: List[int], image_size: Tuple[int, int]) -> torch.Tensor:
    old_height, old_width = image_size
    new_height, new_width = size
    ratios = torch.tensor((new_width / old_width, new_height / old_height), device=bounding_box.device)
    return bounding_box.view(-1, 2, 2).mul(ratios).view(bounding_box.shape)


vertical_flip_image_tensor = _FT.vflip
vertical_flip_image_pil = _FP.vflip

affine_image_tensor = _FT.affine
affine_image_pil = _FP.affine

rotate_image_tensor = _FT.rotate
rotate_image_pil = _FP.rotate

pad_image_tensor = _FT.pad
pad_image_pil = _FP.pad

crop_image_tensor = _FT.crop
crop_image_pil = _FP.crop

perspective_image_tensor = _FT.perspective
perspective_image_pil = _FP.perspective


import numbers


def _center_crop_parse_output_size(output_size: List[int]) -> List[int]:
    if isinstance(output_size, numbers.Number):
        return [int(output_size), int(output_size)]
    elif isinstance(output_size, (tuple, list)) and len(output_size) == 1:
        return [output_size[0], output_size[0]]
    else:
        return list(output_size)


def _center_crop_compute_padding(crop_height: int, crop_width: int, image_height: int, image_width: int) -> List[int]:
    return [
        (crop_width - image_width) // 2 if crop_width > image_width else 0,
        (crop_height - image_height) // 2 if crop_height > image_height else 0,
        (crop_width - image_width + 1) // 2 if crop_width > image_width else 0,
        (crop_height - image_height + 1) // 2 if crop_height > image_height else 0,
    ]


def _center_crop_compute_crop_anchor(
    crop_height: int, crop_width: int, image_height: int, image_width: int
) -> Tuple[int, int]:
    crop_top = int(round((image_height - crop_height) / 2.0))
    crop_left = int(round((image_width - crop_width) / 2.0))
    return crop_top, crop_left


def center_crop_image_tensor(img: torch.Tensor, output_size: List[int]) -> torch.Tensor:
    crop_height, crop_width = _center_crop_parse_output_size(output_size)
    image_height, image_width = get_image_size(img)

    if crop_height > image_height or crop_width > image_width:
        padding_ltrb = _center_crop_compute_padding(crop_height, crop_width, image_height, image_width)
        img = pad_image_tensor(img, padding_ltrb, fill=0)

        image_height, image_width = get_image_size(img)
        if crop_width == image_width and crop_height == image_height:
            return img

    crop_top, crop_left = _center_crop_compute_crop_anchor(crop_height, crop_width, image_height, image_width)
    return crop_image_tensor(img, crop_top, crop_left, crop_height, crop_width)


def center_crop_image_pil(img: PIL.Image.Image, output_size: List[int]) -> PIL.Image.Image:
    crop_height, crop_width = _center_crop_parse_output_size(output_size)
    image_height, image_width = get_image_size(img)

    if crop_height > image_height or crop_width > image_width:
        padding_ltrb = _center_crop_compute_padding(crop_height, crop_width, image_height, image_width)
        img = pad_image_pil(img, padding_ltrb, fill=0)

        image_height, image_width = get_image_size(img)
        if crop_width == image_width and crop_height == image_height:
            return img

    crop_top, crop_left = _center_crop_compute_crop_anchor(crop_height, crop_width, image_height, image_width)
    return crop_image_pil(img, crop_top, crop_left, crop_height, crop_width)


def resized_crop_image_tensor(
    img: torch.Tensor,
    top: int,
    left: int,
    height: int,
    width: int,
    size: List[int],
    interpolation: str = "bilinear",
) -> torch.Tensor:
    img = crop_image_tensor(img, top, left, height, width)
    return resize_image_tensor(img, size, interpolation)


def resized_crop_image_pil(
    img: PIL.Image.Image,
    top: int,
    left: int,
    height: int,
    width: int,
    size: List[int],
    interpolation: int = PIL.Image.BILINEAR,
) -> PIL.Image.Image:
    img = crop_image_pil(img, top, left, height, width)
    return resize_image_pil(img, size, interpolation)
