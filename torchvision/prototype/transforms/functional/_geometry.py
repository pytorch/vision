import numbers
from typing import Tuple, List, Optional, Sequence, Union

import PIL.Image
import torch
from torchvision.prototype import features
from torchvision.prototype.transforms import InterpolationMode
from torchvision.prototype.transforms.functional import get_image_size
from torchvision.transforms import functional_tensor as _FT, functional_pil as _FP
from torchvision.transforms.functional import pil_modes_mapping

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
    interpolation: InterpolationMode = InterpolationMode.BILINEAR,
    max_size: Optional[int] = None,
    antialias: Optional[bool] = None,
) -> torch.Tensor:
    new_height, new_width = size
    old_height, old_width = _FT.get_image_size(image)
    num_channels = _FT.get_image_num_channels(image)
    batch_shape = image.shape[:-3]
    return _FT.resize(
        image.reshape((-1, num_channels, old_height, old_width)),
        size=size,
        interpolation=interpolation.value,
        max_size=max_size,
        antialias=antialias,
    ).reshape(batch_shape + (num_channels, new_height, new_width))


def resize_image_pil(
    img: PIL.Image.Image,
    size: Union[Sequence[int], int],
    interpolation: InterpolationMode = InterpolationMode.BILINEAR,
    max_size: Optional[int] = None,
) -> PIL.Image.Image:
    return _FP.resize(img, size, interpolation=pil_modes_mapping[interpolation], max_size=max_size)


def resize_segmentation_mask(
    segmentation_mask: torch.Tensor, size: List[int], max_size: Optional[int] = None
) -> torch.Tensor:
    return resize_image_tensor(segmentation_mask, size=size, interpolation=InterpolationMode.NEAREST, max_size=max_size)


# TODO: handle max_size
def resize_bounding_box(bounding_box: torch.Tensor, size: List[int], image_size: Tuple[int, int]) -> torch.Tensor:
    old_height, old_width = image_size
    new_height, new_width = size
    ratios = torch.tensor((new_width / old_width, new_height / old_height), device=bounding_box.device)
    return bounding_box.view(-1, 2, 2).mul(ratios).view(bounding_box.shape)


vertical_flip_image_tensor = _FT.vflip
vertical_flip_image_pil = _FP.vflip


def affine_image_tensor(
    img: torch.Tensor,
    matrix: List[float],
    interpolation: InterpolationMode = InterpolationMode.NEAREST,
    fill: Optional[List[float]] = None,
) -> torch.Tensor:
    return _FT.affine(img, matrix, interpolation=interpolation.value, fill=fill)


def affine_image_pil(
    img: PIL.Image.Image,
    matrix: List[float],
    interpolation: InterpolationMode = InterpolationMode.NEAREST,
    fill: Optional[List[float]] = None,
) -> PIL.Image.Image:
    return _FP.affine(img, matrix, interpolation=pil_modes_mapping[interpolation], fill=fill)


def rotate_image_tensor(
    img: torch.Tensor,
    matrix: List[float],
    interpolation: InterpolationMode = InterpolationMode.NEAREST,
    expand: bool = False,
    fill: Optional[List[float]] = None,
) -> torch.Tensor:
    return _FT.rotate(img, matrix, interpolation=interpolation.value, expand=expand, fill=fill)


def rotate_image_pil(
    img: PIL.Image.Image,
    angle: float,
    interpolation: InterpolationMode = InterpolationMode.NEAREST,
    expand: bool = False,
    fill: Optional[List[float]] = None,
) -> PIL.Image.Image:
    return _FP.rotate(img, angle, interpolation=pil_modes_mapping[interpolation], expand=expand, fill=fill)


pad_image_tensor = _FT.pad
pad_image_pil = _FP.pad

crop_image_tensor = _FT.crop
crop_image_pil = _FP.crop


def perspective_image_tensor(
    img: torch.Tensor,
    perspective_coeffs: List[float],
    interpolation: InterpolationMode = InterpolationMode.BILINEAR,
    fill: Optional[List[float]] = None,
) -> torch.Tensor:
    return _FT.perspective(img, perspective_coeffs, interpolation=interpolation.value, fill=fill)


def perspective_image_pil(
    img: PIL.Image.Image,
    perspective_coeffs: float,
    interpolation: InterpolationMode = InterpolationMode.BICUBIC,
    fill: Optional[List[float]] = None,
) -> PIL.Image.Image:
    return _FP.perspective(img, perspective_coeffs, interpolation=pil_modes_mapping[interpolation], fill=fill)


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
    interpolation: InterpolationMode = InterpolationMode.BILINEAR,
) -> torch.Tensor:
    img = crop_image_tensor(img, top, left, height, width)
    return resize_image_tensor(img, size, interpolation=interpolation)


def resized_crop_image_pil(
    img: PIL.Image.Image,
    top: int,
    left: int,
    height: int,
    width: int,
    size: List[int],
    interpolation: InterpolationMode = InterpolationMode.BILINEAR,
) -> PIL.Image.Image:
    img = crop_image_pil(img, top, left, height, width)
    return resize_image_pil(img, size, interpolation=interpolation)
