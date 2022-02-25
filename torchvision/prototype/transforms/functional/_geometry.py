import numbers
from typing import Tuple, List, Optional, Sequence, Union

import PIL.Image
import torch
from torchvision.prototype import features
from torchvision.prototype.transforms import InterpolationMode
from torchvision.prototype.transforms.functional import get_image_size
from torchvision.transforms import functional_tensor as _FT, functional_pil as _FP
from torchvision.transforms.functional import pil_modes_mapping, _get_inverse_affine_matrix

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
    old_width, old_height = _FT.get_image_size(image)
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


def _affine_parse_args(
    angle: float,
    translate: List[float],
    scale: float,
    shear: List[float],
    interpolation: InterpolationMode = InterpolationMode.NEAREST,
    center: Optional[List[float]] = None,
) -> Tuple[float, List[float], List[float], Optional[List[float]]]:
    if not isinstance(angle, (int, float)):
        raise TypeError("Argument angle should be int or float")

    if not isinstance(translate, (list, tuple)):
        raise TypeError("Argument translate should be a sequence")

    if len(translate) != 2:
        raise ValueError("Argument translate should be a sequence of length 2")

    if scale <= 0.0:
        raise ValueError("Argument scale should be positive")

    if not isinstance(shear, (numbers.Number, (list, tuple))):
        raise TypeError("Shear should be either a single value or a sequence of two values")

    if not isinstance(interpolation, InterpolationMode):
        raise TypeError("Argument interpolation should be a InterpolationMode")

    if isinstance(angle, int):
        angle = float(angle)

    if isinstance(translate, tuple):
        translate = list(translate)

    if isinstance(shear, numbers.Number):
        shear = [shear, 0.0]

    if isinstance(shear, tuple):
        shear = list(shear)

    if len(shear) == 1:
        shear = [shear[0], shear[0]]

    if len(shear) != 2:
        raise ValueError(f"Shear should be a sequence containing two values. Got {shear}")

    if center is not None and not isinstance(center, (list, tuple)):
        raise TypeError("Argument center should be a sequence")

    return angle, translate, shear, center


def affine_image_tensor(
    img: torch.Tensor,
    angle: float,
    translate: List[float],
    scale: float,
    shear: List[float],
    interpolation: InterpolationMode = InterpolationMode.NEAREST,
    fill: Optional[List[float]] = None,
    center: Optional[List[float]] = None,
) -> torch.Tensor:
    angle, translate, shear, center = _affine_parse_args(angle, translate, scale, shear, interpolation, center)

    center_f = [0.0, 0.0]
    if center is not None:
        width, height = get_image_size(img)
        # Center values should be in pixel coordinates but translated such that (0, 0) corresponds to image center.
        center_f = [1.0 * (c - s * 0.5) for c, s in zip(center, (width, height))]

    translate_f = [1.0 * t for t in translate]
    matrix = _get_inverse_affine_matrix(center_f, angle, translate_f, scale, shear)

    return _FT.affine(img, matrix, interpolation=interpolation.value, fill=fill)


def affine_image_pil(
    img: PIL.Image.Image,
    angle: float,
    translate: List[float],
    scale: float,
    shear: List[float],
    interpolation: InterpolationMode = InterpolationMode.NEAREST,
    fill: Optional[List[float]] = None,
    center: Optional[List[float]] = None,
) -> PIL.Image.Image:
    angle, translate, shear, center = _affine_parse_args(angle, translate, scale, shear, interpolation, center)

    # center = (img_size[0] * 0.5 + 0.5, img_size[1] * 0.5 + 0.5)
    # it is visually better to estimate the center without 0.5 offset
    # otherwise image rotated by 90 degrees is shifted vs output image of torch.rot90 or F_t.affine
    if center is None:
        width, height = get_image_size(img)
        center = [width * 0.5, height * 0.5]
    matrix = _get_inverse_affine_matrix(center, angle, translate, scale, shear)

    return _FP.affine(img, matrix, interpolation=pil_modes_mapping[interpolation], fill=fill)


def rotate_image_tensor(
    img: torch.Tensor,
    angle: float,
    interpolation: InterpolationMode = InterpolationMode.NEAREST,
    expand: bool = False,
    fill: Optional[List[float]] = None,
    center: Optional[List[float]] = None,
) -> torch.Tensor:
    center_f = [0.0, 0.0]
    if center is not None:
        width, height = get_image_size(img)
        # Center values should be in pixel coordinates but translated such that (0, 0) corresponds to image center.
        center_f = [1.0 * (c - s * 0.5) for c, s in zip(center, (width, height))]

    # due to current incoherence of rotation angle direction between affine and rotate implementations
    # we need to set -angle.
    matrix = _get_inverse_affine_matrix(center_f, -angle, [0.0, 0.0], 1.0, [0.0, 0.0])
    return _FT.rotate(img, matrix, interpolation=interpolation.value, expand=expand, fill=fill)


def rotate_image_pil(
    img: PIL.Image.Image,
    angle: float,
    interpolation: InterpolationMode = InterpolationMode.NEAREST,
    expand: bool = False,
    fill: Optional[List[float]] = None,
    center: Optional[List[float]] = None,
) -> PIL.Image.Image:
    return _FP.rotate(
        img, angle, interpolation=pil_modes_mapping[interpolation], expand=expand, fill=fill, center=center
    )


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
    image_width, image_height = get_image_size(img)

    if crop_height > image_height or crop_width > image_width:
        padding_ltrb = _center_crop_compute_padding(crop_height, crop_width, image_height, image_width)
        img = pad_image_tensor(img, padding_ltrb, fill=0)

        image_width, image_height = get_image_size(img)
        if crop_width == image_width and crop_height == image_height:
            return img

    crop_top, crop_left = _center_crop_compute_crop_anchor(crop_height, crop_width, image_height, image_width)
    return crop_image_tensor(img, crop_top, crop_left, crop_height, crop_width)


def center_crop_image_pil(img: PIL.Image.Image, output_size: List[int]) -> PIL.Image.Image:
    crop_height, crop_width = _center_crop_parse_output_size(output_size)
    image_width, image_height = get_image_size(img)

    if crop_height > image_height or crop_width > image_width:
        padding_ltrb = _center_crop_compute_padding(crop_height, crop_width, image_height, image_width)
        img = pad_image_pil(img, padding_ltrb, fill=0)

        image_width, image_height = get_image_size(img)
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
