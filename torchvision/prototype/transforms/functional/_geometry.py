import numbers
import warnings
from typing import List, Optional, Sequence, Tuple, Union

import PIL.Image
import torch
from torchvision.prototype import features
from torchvision.transforms import functional_pil as _FP, functional_tensor as _FT
from torchvision.transforms.functional import (
    _compute_output_size,
    _get_inverse_affine_matrix,
    _get_perspective_coeffs,
    InterpolationMode,
    pil_modes_mapping,
    pil_to_tensor,
    to_pil_image,
)

from ._meta import convert_bounding_box_format, get_dimensions_image_pil, get_dimensions_image_tensor


# shortcut type
DType = Union[torch.Tensor, PIL.Image.Image, features._Feature]


horizontal_flip_image_tensor = _FT.hflip
horizontal_flip_image_pil = _FP.hflip


def horizontal_flip_segmentation_mask(segmentation_mask: torch.Tensor) -> torch.Tensor:
    return horizontal_flip_image_tensor(segmentation_mask)


def horizontal_flip_bounding_box(
    bounding_box: torch.Tensor, format: features.BoundingBoxFormat, image_size: Tuple[int, int]
) -> torch.Tensor:
    shape = bounding_box.shape

    bounding_box = convert_bounding_box_format(
        bounding_box, old_format=format, new_format=features.BoundingBoxFormat.XYXY
    ).view(-1, 4)

    bounding_box[:, [0, 2]] = image_size[1] - bounding_box[:, [2, 0]]

    return convert_bounding_box_format(
        bounding_box, old_format=features.BoundingBoxFormat.XYXY, new_format=format, copy=False
    ).view(shape)


def horizontal_flip(inpt: DType) -> DType:
    if isinstance(inpt, features._Feature):
        return inpt.horizontal_flip()
    elif isinstance(inpt, PIL.Image.Image):
        return horizontal_flip_image_pil(inpt)
    else:
        return horizontal_flip_image_tensor(inpt)


vertical_flip_image_tensor = _FT.vflip
vertical_flip_image_pil = _FP.vflip


def vertical_flip_segmentation_mask(segmentation_mask: torch.Tensor) -> torch.Tensor:
    return vertical_flip_image_tensor(segmentation_mask)


def vertical_flip_bounding_box(
    bounding_box: torch.Tensor, format: features.BoundingBoxFormat, image_size: Tuple[int, int]
) -> torch.Tensor:
    shape = bounding_box.shape

    bounding_box = convert_bounding_box_format(
        bounding_box, old_format=format, new_format=features.BoundingBoxFormat.XYXY
    ).view(-1, 4)

    bounding_box[:, [1, 3]] = image_size[0] - bounding_box[:, [3, 1]]

    return convert_bounding_box_format(
        bounding_box, old_format=features.BoundingBoxFormat.XYXY, new_format=format, copy=False
    ).view(shape)


def vertical_flip(inpt: DType) -> DType:
    if isinstance(inpt, features._Feature):
        return inpt.vertical_flip()
    elif isinstance(inpt, PIL.Image.Image):
        return vertical_flip_image_pil(inpt)
    else:
        return vertical_flip_image_tensor(inpt)


# We changed the names to align them with the transforms, i.e. `RandomHorizontalFlip`. Still, `hflip` and `vflip` are
# prevalent and well understood. Thus, we just alias them without deprecating the old names.
hflip = horizontal_flip
vflip = vertical_flip


def resize_image_tensor(
    image: torch.Tensor,
    size: List[int],
    interpolation: InterpolationMode = InterpolationMode.BILINEAR,
    max_size: Optional[int] = None,
    antialias: bool = False,
) -> torch.Tensor:
    num_channels, old_height, old_width = get_dimensions_image_tensor(image)
    new_height, new_width = _compute_output_size((old_height, old_width), size=size, max_size=max_size)
    batch_shape = image.shape[:-3]
    return _FT.resize(
        image.reshape((-1, num_channels, old_height, old_width)),
        size=[new_height, new_width],
        interpolation=interpolation.value,
        antialias=antialias,
    ).reshape(batch_shape + (num_channels, new_height, new_width))


def resize_image_pil(
    img: PIL.Image.Image,
    size: Union[Sequence[int], int],
    interpolation: InterpolationMode = InterpolationMode.BILINEAR,
    max_size: Optional[int] = None,
) -> PIL.Image.Image:
    if isinstance(size, int):
        size = [size, size]
    # Explicitly cast size to list otherwise mypy issue: incompatible type "Sequence[int]"; expected "List[int]"
    size: List[int] = list(size)
    size = _compute_output_size(img.size[::-1], size=size, max_size=max_size)
    return _FP.resize(img, size, interpolation=pil_modes_mapping[interpolation])


def resize_segmentation_mask(
    segmentation_mask: torch.Tensor, size: List[int], max_size: Optional[int] = None
) -> torch.Tensor:
    return resize_image_tensor(segmentation_mask, size=size, interpolation=InterpolationMode.NEAREST, max_size=max_size)


def resize_bounding_box(
    bounding_box: torch.Tensor, size: List[int], image_size: Tuple[int, int], max_size: Optional[int] = None
) -> torch.Tensor:
    old_height, old_width = image_size
    new_height, new_width = _compute_output_size(image_size, size=size, max_size=max_size)
    ratios = torch.tensor((new_width / old_width, new_height / old_height), device=bounding_box.device)
    return bounding_box.view(-1, 2, 2).mul(ratios).view(bounding_box.shape)


def resize(
    inpt: DType,
    size: List[int],
    interpolation: InterpolationMode = InterpolationMode.BILINEAR,
    max_size: Optional[int] = None,
    antialias: Optional[bool] = None,
) -> DType:
    if isinstance(inpt, features._Feature):
        antialias = False if antialias is None else antialias
        return inpt.resize(size, interpolation=interpolation, max_size=max_size, antialias=antialias)
    elif isinstance(inpt, PIL.Image.Image):
        if antialias is not None and not antialias:
            warnings.warn("Anti-alias option is always applied for PIL Image input. Argument antialias is ignored.")
        return resize_image_pil(inpt, size, interpolation=interpolation, max_size=max_size)
    else:
        antialias = False if antialias is None else antialias
        return resize_image_tensor(inpt, size, interpolation=interpolation, max_size=max_size, antialias=antialias)


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
    num_channels, height, width = img.shape[-3:]
    extra_dims = img.shape[:-3]
    img = img.view(-1, num_channels, height, width)

    angle, translate, shear, center = _affine_parse_args(angle, translate, scale, shear, interpolation, center)

    center_f = [0.0, 0.0]
    if center is not None:
        # Center values should be in pixel coordinates but translated such that (0, 0) corresponds to image center.
        center_f = [1.0 * (c - s * 0.5) for c, s in zip(center, [width, height])]

    translate_f = [1.0 * t for t in translate]
    matrix = _get_inverse_affine_matrix(center_f, angle, translate_f, scale, shear)

    output = _FT.affine(img, matrix, interpolation=interpolation.value, fill=fill)
    return output.view(extra_dims + (num_channels, height, width))


def affine_image_pil(
    img: PIL.Image.Image,
    angle: float,
    translate: List[float],
    scale: float,
    shear: List[float],
    interpolation: InterpolationMode = InterpolationMode.NEAREST,
    fill: Optional[Union[int, float, Sequence[int], Sequence[float]]] = None,
    center: Optional[List[float]] = None,
) -> PIL.Image.Image:
    angle, translate, shear, center = _affine_parse_args(angle, translate, scale, shear, interpolation, center)

    # center = (img_size[0] * 0.5 + 0.5, img_size[1] * 0.5 + 0.5)
    # it is visually better to estimate the center without 0.5 offset
    # otherwise image rotated by 90 degrees is shifted vs output image of torch.rot90 or F_t.affine
    if center is None:
        _, height, width = get_dimensions_image_pil(img)
        center = [width * 0.5, height * 0.5]
    matrix = _get_inverse_affine_matrix(center, angle, translate, scale, shear)

    return _FP.affine(img, matrix, interpolation=pil_modes_mapping[interpolation], fill=fill)


def _affine_bounding_box_xyxy(
    bounding_box: torch.Tensor,
    image_size: Tuple[int, int],
    angle: float,
    translate: Optional[List[float]] = None,
    scale: Optional[float] = None,
    shear: Optional[List[float]] = None,
    center: Optional[List[float]] = None,
    expand: bool = False,
) -> torch.Tensor:
    dtype = bounding_box.dtype if torch.is_floating_point(bounding_box) else torch.float32
    device = bounding_box.device

    if translate is None:
        translate = [0.0, 0.0]

    if scale is None:
        scale = 1.0

    if shear is None:
        shear = [0.0, 0.0]

    if center is None:
        height, width = image_size
        center_f = [width * 0.5, height * 0.5]
    else:
        center_f = [float(c) for c in center]

    translate_f = [float(t) for t in translate]
    affine_matrix = torch.tensor(
        _get_inverse_affine_matrix(center_f, angle, translate_f, scale, shear, inverted=False),
        dtype=dtype,
        device=device,
    ).view(2, 3)
    # 1) Let's transform bboxes into a tensor of 4 points (top-left, top-right, bottom-left, bottom-right corners).
    # Tensor of points has shape (N * 4, 3), where N is the number of bboxes
    # Single point structure is similar to
    # [(xmin, ymin, 1), (xmax, ymin, 1), (xmax, ymax, 1), (xmin, ymax, 1)]
    points = bounding_box[:, [[0, 1], [2, 1], [2, 3], [0, 3]]].view(-1, 2)
    points = torch.cat([points, torch.ones(points.shape[0], 1, device=points.device)], dim=-1)
    # 2) Now let's transform the points using affine matrix
    transformed_points = torch.matmul(points, affine_matrix.T)
    # 3) Reshape transformed points to [N boxes, 4 points, x/y coords]
    # and compute bounding box from 4 transformed points:
    transformed_points = transformed_points.view(-1, 4, 2)
    out_bbox_mins, _ = torch.min(transformed_points, dim=1)
    out_bbox_maxs, _ = torch.max(transformed_points, dim=1)
    out_bboxes = torch.cat([out_bbox_mins, out_bbox_maxs], dim=1)

    if expand:
        # Compute minimum point for transformed image frame:
        # Points are Top-Left, Top-Right, Bottom-Left, Bottom-Right points.
        height, width = image_size
        points = torch.tensor(
            [
                [0.0, 0.0, 1.0],
                [0.0, 1.0 * height, 1.0],
                [1.0 * width, 1.0 * height, 1.0],
                [1.0 * width, 0.0, 1.0],
            ],
            dtype=dtype,
            device=device,
        )
        new_points = torch.matmul(points, affine_matrix.T)
        tr, _ = torch.min(new_points, dim=0, keepdim=True)
        # Translate bounding boxes
        out_bboxes[:, 0::2] = out_bboxes[:, 0::2] - tr[:, 0]
        out_bboxes[:, 1::2] = out_bboxes[:, 1::2] - tr[:, 1]

    return out_bboxes


def affine_bounding_box(
    bounding_box: torch.Tensor,
    format: features.BoundingBoxFormat,
    image_size: Tuple[int, int],
    angle: float,
    translate: List[float],
    scale: float,
    shear: List[float],
    center: Optional[List[float]] = None,
) -> torch.Tensor:
    original_shape = bounding_box.shape
    bounding_box = convert_bounding_box_format(
        bounding_box, old_format=format, new_format=features.BoundingBoxFormat.XYXY
    ).view(-1, 4)

    out_bboxes = _affine_bounding_box_xyxy(bounding_box, image_size, angle, translate, scale, shear, center)

    # out_bboxes should be of shape [N boxes, 4]

    return convert_bounding_box_format(
        out_bboxes, old_format=features.BoundingBoxFormat.XYXY, new_format=format, copy=False
    ).view(original_shape)


def affine_segmentation_mask(
    mask: torch.Tensor,
    angle: float,
    translate: List[float],
    scale: float,
    shear: List[float],
    center: Optional[List[float]] = None,
) -> torch.Tensor:
    return affine_image_tensor(
        mask,
        angle=angle,
        translate=translate,
        scale=scale,
        shear=shear,
        interpolation=InterpolationMode.NEAREST,
        center=center,
    )


def _convert_fill_arg(fill: Optional[Union[int, float, Sequence[int], Sequence[float]]]) -> Optional[List[float]]:
    if fill is None:
        fill = 0

    # This cast does Sequence -> List[float] to please mypy and torch.jit.script
    if not isinstance(fill, (int, float)):
        fill = [float(v) for v in list(fill)]
    else:
        # It is OK to cast int to float as later we use inpt.dtype
        fill = [float(fill)]
    return fill


def affine(
    inpt: DType,
    angle: float,
    translate: List[float],
    scale: float,
    shear: List[float],
    interpolation: InterpolationMode = InterpolationMode.NEAREST,
    fill: Optional[Union[int, float, Sequence[int], Sequence[float]]] = None,
    center: Optional[List[float]] = None,
) -> DType:
    if isinstance(inpt, features._Feature):
        return inpt.affine(
            angle, translate=translate, scale=scale, shear=shear, interpolation=interpolation, fill=fill, center=center
        )
    elif isinstance(inpt, PIL.Image.Image):
        return affine_image_pil(
            inpt,
            angle,
            translate=translate,
            scale=scale,
            shear=shear,
            interpolation=interpolation,
            fill=fill,
            center=center,
        )
    else:
        fill = _convert_fill_arg(fill)

        return affine_image_tensor(
            inpt,
            angle,
            translate=translate,
            scale=scale,
            shear=shear,
            interpolation=interpolation,
            fill=fill,
            center=center,
        )


def rotate_image_tensor(
    img: torch.Tensor,
    angle: float,
    interpolation: InterpolationMode = InterpolationMode.NEAREST,
    expand: bool = False,
    fill: Optional[List[float]] = None,
    center: Optional[List[float]] = None,
) -> torch.Tensor:
    num_channels, height, width = img.shape[-3:]
    extra_dims = img.shape[:-3]
    img = img.view(-1, num_channels, height, width)

    center_f = [0.0, 0.0]
    if center is not None:
        if expand:
            warnings.warn("The provided center argument has no effect on the result if expand is True")
        else:
            _, height, width = get_dimensions_image_tensor(img)
            # Center values should be in pixel coordinates but translated such that (0, 0) corresponds to image center.
            center_f = [1.0 * (c - s * 0.5) for c, s in zip(center, [width, height])]

    # due to current incoherence of rotation angle direction between affine and rotate implementations
    # we need to set -angle.
    matrix = _get_inverse_affine_matrix(center_f, -angle, [0.0, 0.0], 1.0, [0.0, 0.0])
    output = _FT.rotate(img, matrix, interpolation=interpolation.value, expand=expand, fill=fill)
    new_height, new_width = output.shape[-2:]
    return output.view(extra_dims + (num_channels, new_height, new_width))


def rotate_image_pil(
    img: PIL.Image.Image,
    angle: float,
    interpolation: InterpolationMode = InterpolationMode.NEAREST,
    expand: bool = False,
    fill: Optional[Union[int, float, Sequence[int], Sequence[float]]] = None,
    center: Optional[List[float]] = None,
) -> PIL.Image.Image:
    if center is not None and expand:
        warnings.warn("The provided center argument has no effect on the result if expand is True")
        center = None

    return _FP.rotate(
        img, angle, interpolation=pil_modes_mapping[interpolation], expand=expand, fill=fill, center=center
    )


def rotate_bounding_box(
    bounding_box: torch.Tensor,
    format: features.BoundingBoxFormat,
    image_size: Tuple[int, int],
    angle: float,
    expand: bool = False,
    center: Optional[List[float]] = None,
) -> torch.Tensor:
    if center is not None and expand:
        warnings.warn("The provided center argument has no effect on the result if expand is True")
        center = None

    original_shape = bounding_box.shape
    bounding_box = convert_bounding_box_format(
        bounding_box, old_format=format, new_format=features.BoundingBoxFormat.XYXY
    ).view(-1, 4)

    out_bboxes = _affine_bounding_box_xyxy(bounding_box, image_size, angle=-angle, center=center, expand=expand)

    return convert_bounding_box_format(
        out_bboxes, old_format=features.BoundingBoxFormat.XYXY, new_format=format, copy=False
    ).view(original_shape)


def rotate_segmentation_mask(
    img: torch.Tensor,
    angle: float,
    expand: bool = False,
    center: Optional[List[float]] = None,
) -> torch.Tensor:
    return rotate_image_tensor(
        img,
        angle=angle,
        expand=expand,
        interpolation=InterpolationMode.NEAREST,
        center=center,
    )


def rotate(
    inpt: DType,
    angle: float,
    interpolation: InterpolationMode = InterpolationMode.NEAREST,
    expand: bool = False,
    fill: Optional[Union[int, float, Sequence[int], Sequence[float]]] = None,
    center: Optional[List[float]] = None,
) -> DType:
    if isinstance(inpt, features._Feature):
        return inpt.rotate(angle, interpolation=interpolation, expand=expand, fill=fill, center=center)
    elif isinstance(inpt, PIL.Image.Image):
        return rotate_image_pil(inpt, angle, interpolation=interpolation, expand=expand, fill=fill, center=center)
    else:
        fill = _convert_fill_arg(fill)

        return rotate_image_tensor(inpt, angle, interpolation=interpolation, expand=expand, fill=fill, center=center)


pad_image_pil = _FP.pad


def pad_image_tensor(
    img: torch.Tensor,
    padding: Union[int, List[int]],
    fill: Optional[Union[int, float]] = 0,
    padding_mode: str = "constant",
) -> torch.Tensor:
    num_channels, height, width = img.shape[-3:]
    extra_dims = img.shape[:-3]

    padded_image = _FT.pad(
        img=img.view(-1, num_channels, height, width), padding=padding, fill=fill, padding_mode=padding_mode
    )

    new_height, new_width = padded_image.shape[-2:]
    return padded_image.view(extra_dims + (num_channels, new_height, new_width))


# TODO: This should be removed once pytorch pad supports non-scalar padding values
def _pad_with_vector_fill(
    img: torch.Tensor,
    padding: Union[int, List[int]],
    fill: Sequence[float] = [0.0],
    padding_mode: str = "constant",
) -> torch.Tensor:
    if padding_mode != "constant":
        raise ValueError(f"Padding mode '{padding_mode}' is not supported if fill is not scalar")

    output = pad_image_tensor(img, padding, fill=0, padding_mode="constant")
    left, right, top, bottom = _FT._parse_pad_padding(padding)
    fill = torch.tensor(fill, dtype=img.dtype, device=img.device).view(-1, 1, 1)

    if top > 0:
        output[..., :top, :] = fill
    if left > 0:
        output[..., :, :left] = fill
    if bottom > 0:
        output[..., -bottom:, :] = fill
    if right > 0:
        output[..., :, -right:] = fill
    return output


def pad_segmentation_mask(
    segmentation_mask: torch.Tensor, padding: Union[int, List[int]], padding_mode: str = "constant"
) -> torch.Tensor:
    num_masks, height, width = segmentation_mask.shape[-3:]
    extra_dims = segmentation_mask.shape[:-3]

    padded_mask = pad_image_tensor(
        img=segmentation_mask.view(-1, num_masks, height, width), padding=padding, fill=0, padding_mode=padding_mode
    )

    new_height, new_width = padded_mask.shape[-2:]
    return padded_mask.view(extra_dims + (num_masks, new_height, new_width))


def pad_bounding_box(
    bounding_box: torch.Tensor, padding: Union[int, List[int]], format: features.BoundingBoxFormat
) -> torch.Tensor:
    left, _, top, _ = _FT._parse_pad_padding(padding)

    bounding_box = bounding_box.clone()

    # this works without conversion since padding only affects xy coordinates
    bounding_box[..., 0] += left
    bounding_box[..., 1] += top
    if format == features.BoundingBoxFormat.XYXY:
        bounding_box[..., 2] += left
        bounding_box[..., 3] += top
    return bounding_box


def pad(
    inpt: DType,
    padding: Union[int, Sequence[int]],
    fill: Optional[Union[int, float, Sequence[int], Sequence[float]]] = None,
    padding_mode: str = "constant",
) -> DType:
    if isinstance(inpt, features._Feature):
        return inpt.pad(padding, fill=fill, padding_mode=padding_mode)
    elif isinstance(inpt, PIL.Image.Image):
        return pad_image_pil(inpt, padding, fill=fill, padding_mode=padding_mode)
    else:
        # This cast does Sequence[int] -> List[int] and is required to make mypy happy
        if not isinstance(padding, int):
            padding = list(padding)

        # TODO: PyTorch's pad supports only scalars on fill. So we need to overwrite the colour
        if isinstance(fill, (int, float)) or fill is None:
            return pad_image_tensor(inpt, padding, fill=fill, padding_mode=padding_mode)
        return _pad_with_vector_fill(inpt, padding, fill=fill, padding_mode=padding_mode)


crop_image_tensor = _FT.crop
crop_image_pil = _FP.crop


def crop_bounding_box(
    bounding_box: torch.Tensor,
    format: features.BoundingBoxFormat,
    top: int,
    left: int,
) -> torch.Tensor:
    bounding_box = convert_bounding_box_format(
        bounding_box, old_format=format, new_format=features.BoundingBoxFormat.XYXY
    )

    # Crop or implicit pad if left and/or top have negative values:
    bounding_box[..., 0::2] -= left
    bounding_box[..., 1::2] -= top

    return convert_bounding_box_format(
        bounding_box, old_format=features.BoundingBoxFormat.XYXY, new_format=format, copy=False
    )


def crop_segmentation_mask(img: torch.Tensor, top: int, left: int, height: int, width: int) -> torch.Tensor:
    return crop_image_tensor(img, top, left, height, width)


def crop(inpt: DType, top: int, left: int, height: int, width: int) -> DType:
    if isinstance(inpt, features._Feature):
        return inpt.crop(top, left, height, width)
    elif isinstance(inpt, PIL.Image.Image):
        return crop_image_pil(inpt, top, left, height, width)
    else:
        return crop_image_tensor(inpt, top, left, height, width)


def perspective_image_tensor(
    img: torch.Tensor,
    perspective_coeffs: List[float],
    interpolation: InterpolationMode = InterpolationMode.BILINEAR,
    fill: Optional[List[float]] = None,
) -> torch.Tensor:
    return _FT.perspective(img, perspective_coeffs, interpolation=interpolation.value, fill=fill)


def perspective_image_pil(
    img: PIL.Image.Image,
    perspective_coeffs: List[float],
    interpolation: InterpolationMode = InterpolationMode.BICUBIC,
    fill: Optional[Union[int, float, Sequence[int], Sequence[float]]] = None,
) -> PIL.Image.Image:
    return _FP.perspective(img, perspective_coeffs, interpolation=pil_modes_mapping[interpolation], fill=fill)


def perspective_bounding_box(
    bounding_box: torch.Tensor,
    format: features.BoundingBoxFormat,
    perspective_coeffs: List[float],
) -> torch.Tensor:

    if len(perspective_coeffs) != 8:
        raise ValueError("Argument perspective_coeffs should have 8 float values")

    original_shape = bounding_box.shape
    bounding_box = convert_bounding_box_format(
        bounding_box, old_format=format, new_format=features.BoundingBoxFormat.XYXY
    ).view(-1, 4)

    dtype = bounding_box.dtype if torch.is_floating_point(bounding_box) else torch.float32
    device = bounding_box.device

    # perspective_coeffs are computed as endpoint -> start point
    # We have to invert perspective_coeffs for bboxes:
    # (x, y) - end point and (x_out, y_out) - start point
    #   x_out = (coeffs[0] * x + coeffs[1] * y + coeffs[2]) / (coeffs[6] * x + coeffs[7] * y + 1)
    #   y_out = (coeffs[3] * x + coeffs[4] * y + coeffs[5]) / (coeffs[6] * x + coeffs[7] * y + 1)
    # and we would like to get:
    # x = (inv_coeffs[0] * x_out + inv_coeffs[1] * y_out + inv_coeffs[2])
    #       / (inv_coeffs[6] * x_out + inv_coeffs[7] * y_out + 1)
    # y = (inv_coeffs[3] * x_out + inv_coeffs[4] * y_out + inv_coeffs[5])
    #       / (inv_coeffs[6] * x_out + inv_coeffs[7] * y_out + 1)
    # and compute inv_coeffs in terms of coeffs

    denom = perspective_coeffs[0] * perspective_coeffs[4] - perspective_coeffs[1] * perspective_coeffs[3]
    if denom == 0:
        raise RuntimeError(
            f"Provided perspective_coeffs {perspective_coeffs} can not be inverted to transform bounding boxes. "
            f"Denominator is zero, denom={denom}"
        )

    inv_coeffs = [
        (perspective_coeffs[4] - perspective_coeffs[5] * perspective_coeffs[7]) / denom,
        (-perspective_coeffs[1] + perspective_coeffs[2] * perspective_coeffs[7]) / denom,
        (perspective_coeffs[1] * perspective_coeffs[5] - perspective_coeffs[2] * perspective_coeffs[4]) / denom,
        (-perspective_coeffs[3] + perspective_coeffs[5] * perspective_coeffs[6]) / denom,
        (perspective_coeffs[0] - perspective_coeffs[2] * perspective_coeffs[6]) / denom,
        (-perspective_coeffs[0] * perspective_coeffs[5] + perspective_coeffs[2] * perspective_coeffs[3]) / denom,
        (-perspective_coeffs[4] * perspective_coeffs[6] + perspective_coeffs[3] * perspective_coeffs[7]) / denom,
        (-perspective_coeffs[0] * perspective_coeffs[7] + perspective_coeffs[1] * perspective_coeffs[6]) / denom,
    ]

    theta1 = torch.tensor(
        [[inv_coeffs[0], inv_coeffs[1], inv_coeffs[2]], [inv_coeffs[3], inv_coeffs[4], inv_coeffs[5]]],
        dtype=dtype,
        device=device,
    )

    theta2 = torch.tensor(
        [[inv_coeffs[6], inv_coeffs[7], 1.0], [inv_coeffs[6], inv_coeffs[7], 1.0]], dtype=dtype, device=device
    )

    # 1) Let's transform bboxes into a tensor of 4 points (top-left, top-right, bottom-left, bottom-right corners).
    # Tensor of points has shape (N * 4, 3), where N is the number of bboxes
    # Single point structure is similar to
    # [(xmin, ymin, 1), (xmax, ymin, 1), (xmax, ymax, 1), (xmin, ymax, 1)]
    points = bounding_box[:, [[0, 1], [2, 1], [2, 3], [0, 3]]].view(-1, 2)
    points = torch.cat([points, torch.ones(points.shape[0], 1, device=points.device)], dim=-1)
    # 2) Now let's transform the points using perspective matrices
    #   x_out = (coeffs[0] * x + coeffs[1] * y + coeffs[2]) / (coeffs[6] * x + coeffs[7] * y + 1)
    #   y_out = (coeffs[3] * x + coeffs[4] * y + coeffs[5]) / (coeffs[6] * x + coeffs[7] * y + 1)

    numer_points = torch.matmul(points, theta1.T)
    denom_points = torch.matmul(points, theta2.T)
    transformed_points = numer_points / denom_points

    # 3) Reshape transformed points to [N boxes, 4 points, x/y coords]
    # and compute bounding box from 4 transformed points:
    transformed_points = transformed_points.view(-1, 4, 2)
    out_bbox_mins, _ = torch.min(transformed_points, dim=1)
    out_bbox_maxs, _ = torch.max(transformed_points, dim=1)
    out_bboxes = torch.cat([out_bbox_mins, out_bbox_maxs], dim=1)

    # out_bboxes should be of shape [N boxes, 4]

    return convert_bounding_box_format(
        out_bboxes, old_format=features.BoundingBoxFormat.XYXY, new_format=format, copy=False
    ).view(original_shape)


def perspective_segmentation_mask(mask: torch.Tensor, perspective_coeffs: List[float]) -> torch.Tensor:
    return perspective_image_tensor(
        mask, perspective_coeffs=perspective_coeffs, interpolation=InterpolationMode.NEAREST
    )


def perspective(
    inpt: DType,
    startpoints: List[List[int]],
    endpoints: List[List[int]],
    interpolation: InterpolationMode = InterpolationMode.BILINEAR,
    fill: Optional[Union[int, float, Sequence[int], Sequence[float]]] = None,
) -> DType:
    perspective_coeffs = _get_perspective_coeffs(startpoints, endpoints)

    if isinstance(inpt, features._Feature):
        return inpt.perspective(perspective_coeffs, interpolation=interpolation, fill=fill)
    elif isinstance(inpt, PIL.Image.Image):
        return perspective_image_pil(inpt, perspective_coeffs, interpolation=interpolation, fill=fill)
    else:
        fill = _convert_fill_arg(fill)

        return perspective_image_tensor(inpt, perspective_coeffs, interpolation=interpolation, fill=fill)


def elastic_image_tensor(
    img: torch.Tensor,
    displacement: torch.Tensor,
    interpolation: InterpolationMode = InterpolationMode.BILINEAR,
    fill: Optional[List[float]] = None,
) -> torch.Tensor:
    return _FT.elastic_transform(img, displacement, interpolation=interpolation.value, fill=fill)


def elastic_image_pil(
    img: PIL.Image.Image,
    displacement: torch.Tensor,
    interpolation: InterpolationMode = InterpolationMode.BILINEAR,
    fill: Optional[Union[int, float, Sequence[int], Sequence[float]]] = None,
) -> PIL.Image.Image:
    t_img = pil_to_tensor(img)
    fill = _convert_fill_arg(fill)

    output = elastic_image_tensor(t_img, displacement, interpolation=interpolation, fill=fill)
    return to_pil_image(output, mode=img.mode)


def elastic_bounding_box(
    bounding_box: torch.Tensor,
    format: features.BoundingBoxFormat,
    displacement: torch.Tensor,
) -> torch.Tensor:
    # TODO: add in docstring about approximation we are doing for grid inversion
    displacement = displacement.to(bounding_box.device)

    original_shape = bounding_box.shape
    bounding_box = convert_bounding_box_format(
        bounding_box, old_format=format, new_format=features.BoundingBoxFormat.XYXY
    ).view(-1, 4)

    # Question (vfdev-5): should we rely on good displacement shape and fetch image size from it
    # Or add image_size arg and check displacement shape
    image_size = displacement.shape[-3], displacement.shape[-2]

    id_grid = _FT._create_identity_grid(list(image_size)).to(bounding_box.device)
    # We construct an approximation of inverse grid as inv_grid = id_grid - displacement
    # This is not an exact inverse of the grid
    inv_grid = id_grid - displacement

    # Get points from bboxes
    points = bounding_box[:, [[0, 1], [2, 1], [2, 3], [0, 3]]].view(-1, 2)
    index_x = torch.floor(points[:, 0] + 0.5).to(dtype=torch.long)
    index_y = torch.floor(points[:, 1] + 0.5).to(dtype=torch.long)
    # Transform points:
    t_size = torch.tensor(image_size[::-1], device=displacement.device, dtype=displacement.dtype)
    transformed_points = (inv_grid[0, index_y, index_x, :] + 1) * 0.5 * t_size - 0.5

    transformed_points = transformed_points.view(-1, 4, 2)
    out_bbox_mins, _ = torch.min(transformed_points, dim=1)
    out_bbox_maxs, _ = torch.max(transformed_points, dim=1)
    out_bboxes = torch.cat([out_bbox_mins, out_bbox_maxs], dim=1)

    return convert_bounding_box_format(
        out_bboxes, old_format=features.BoundingBoxFormat.XYXY, new_format=format, copy=False
    ).view(original_shape)


def elastic_segmentation_mask(mask: torch.Tensor, displacement: torch.Tensor) -> torch.Tensor:
    return elastic_image_tensor(mask, displacement=displacement, interpolation=InterpolationMode.NEAREST)


def elastic(
    inpt: DType,
    displacement: torch.Tensor,
    interpolation: InterpolationMode = InterpolationMode.BILINEAR,
    fill: Optional[Union[int, float, Sequence[int], Sequence[float]]] = None,
) -> DType:
    if isinstance(inpt, features._Feature):
        return inpt.elastic(displacement, interpolation=interpolation, fill=fill)
    elif isinstance(inpt, PIL.Image.Image):
        return elastic_image_pil(inpt, displacement, interpolation=interpolation, fill=fill)
    else:
        fill = _convert_fill_arg(fill)

        return elastic_image_tensor(inpt, displacement, interpolation=interpolation, fill=fill)


elastic_transform = elastic


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
    _, image_height, image_width = get_dimensions_image_tensor(img)

    if crop_height > image_height or crop_width > image_width:
        padding_ltrb = _center_crop_compute_padding(crop_height, crop_width, image_height, image_width)
        img = pad_image_tensor(img, padding_ltrb, fill=0)

        _, image_height, image_width = get_dimensions_image_tensor(img)
        if crop_width == image_width and crop_height == image_height:
            return img

    crop_top, crop_left = _center_crop_compute_crop_anchor(crop_height, crop_width, image_height, image_width)
    return crop_image_tensor(img, crop_top, crop_left, crop_height, crop_width)


def center_crop_image_pil(img: PIL.Image.Image, output_size: List[int]) -> PIL.Image.Image:
    crop_height, crop_width = _center_crop_parse_output_size(output_size)
    _, image_height, image_width = get_dimensions_image_pil(img)

    if crop_height > image_height or crop_width > image_width:
        padding_ltrb = _center_crop_compute_padding(crop_height, crop_width, image_height, image_width)
        img = pad_image_pil(img, padding_ltrb, fill=0)

        _, image_height, image_width = get_dimensions_image_pil(img)
        if crop_width == image_width and crop_height == image_height:
            return img

    crop_top, crop_left = _center_crop_compute_crop_anchor(crop_height, crop_width, image_height, image_width)
    return crop_image_pil(img, crop_top, crop_left, crop_height, crop_width)


def center_crop_bounding_box(
    bounding_box: torch.Tensor,
    format: features.BoundingBoxFormat,
    output_size: List[int],
    image_size: Tuple[int, int],
) -> torch.Tensor:
    crop_height, crop_width = _center_crop_parse_output_size(output_size)
    crop_top, crop_left = _center_crop_compute_crop_anchor(crop_height, crop_width, *image_size)
    return crop_bounding_box(bounding_box, format, top=crop_top, left=crop_left)


def center_crop_segmentation_mask(segmentation_mask: torch.Tensor, output_size: List[int]) -> torch.Tensor:
    return center_crop_image_tensor(img=segmentation_mask, output_size=output_size)


def center_crop(inpt: DType, output_size: List[int]) -> DType:
    if isinstance(inpt, features._Feature):
        return inpt.center_crop(output_size)
    elif isinstance(inpt, PIL.Image.Image):
        return center_crop_image_pil(inpt, output_size)
    else:
        return center_crop_image_tensor(inpt, output_size)


def resized_crop_image_tensor(
    img: torch.Tensor,
    top: int,
    left: int,
    height: int,
    width: int,
    size: List[int],
    interpolation: InterpolationMode = InterpolationMode.BILINEAR,
    antialias: bool = False,
) -> torch.Tensor:
    img = crop_image_tensor(img, top, left, height, width)
    return resize_image_tensor(img, size, interpolation=interpolation, antialias=antialias)


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


def resized_crop_bounding_box(
    bounding_box: torch.Tensor,
    format: features.BoundingBoxFormat,
    top: int,
    left: int,
    height: int,
    width: int,
    size: List[int],
) -> torch.Tensor:
    bounding_box = crop_bounding_box(bounding_box, format, top, left)
    return resize_bounding_box(bounding_box, size, (height, width))


def resized_crop_segmentation_mask(
    mask: torch.Tensor,
    top: int,
    left: int,
    height: int,
    width: int,
    size: List[int],
) -> torch.Tensor:
    mask = crop_segmentation_mask(mask, top, left, height, width)
    return resize_segmentation_mask(mask, size)


def resized_crop(
    inpt: DType,
    top: int,
    left: int,
    height: int,
    width: int,
    size: List[int],
    interpolation: InterpolationMode = InterpolationMode.BILINEAR,
    antialias: Optional[bool] = None,
) -> DType:
    if isinstance(inpt, features._Feature):
        antialias = False if antialias is None else antialias
        return inpt.resized_crop(top, left, height, width, antialias=antialias, size=size, interpolation=interpolation)
    elif isinstance(inpt, PIL.Image.Image):
        return resized_crop_image_pil(inpt, top, left, height, width, size=size, interpolation=interpolation)
    else:
        antialias = False if antialias is None else antialias
        return resized_crop_image_tensor(
            inpt, top, left, height, width, antialias=antialias, size=size, interpolation=interpolation
        )


def _parse_five_crop_size(size: List[int]) -> List[int]:
    if isinstance(size, numbers.Number):
        size = [int(size), int(size)]
    elif isinstance(size, (tuple, list)) and len(size) == 1:
        size = [size[0], size[0]]

    if len(size) != 2:
        raise ValueError("Please provide only two dimensions (h, w) for size.")

    return size


def five_crop_image_tensor(
    img: torch.Tensor, size: List[int]
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    crop_height, crop_width = _parse_five_crop_size(size)
    _, image_height, image_width = get_dimensions_image_tensor(img)

    if crop_width > image_width or crop_height > image_height:
        msg = "Requested crop size {} is bigger than input size {}"
        raise ValueError(msg.format(size, (image_height, image_width)))

    tl = crop_image_tensor(img, 0, 0, crop_height, crop_width)
    tr = crop_image_tensor(img, 0, image_width - crop_width, crop_height, crop_width)
    bl = crop_image_tensor(img, image_height - crop_height, 0, crop_height, crop_width)
    br = crop_image_tensor(img, image_height - crop_height, image_width - crop_width, crop_height, crop_width)
    center = center_crop_image_tensor(img, [crop_height, crop_width])

    return tl, tr, bl, br, center


def five_crop_image_pil(
    img: PIL.Image.Image, size: List[int]
) -> Tuple[PIL.Image.Image, PIL.Image.Image, PIL.Image.Image, PIL.Image.Image, PIL.Image.Image]:
    crop_height, crop_width = _parse_five_crop_size(size)
    _, image_height, image_width = get_dimensions_image_pil(img)

    if crop_width > image_width or crop_height > image_height:
        msg = "Requested crop size {} is bigger than input size {}"
        raise ValueError(msg.format(size, (image_height, image_width)))

    tl = crop_image_pil(img, 0, 0, crop_height, crop_width)
    tr = crop_image_pil(img, 0, image_width - crop_width, crop_height, crop_width)
    bl = crop_image_pil(img, image_height - crop_height, 0, crop_height, crop_width)
    br = crop_image_pil(img, image_height - crop_height, image_width - crop_width, crop_height, crop_width)
    center = center_crop_image_pil(img, [crop_height, crop_width])

    return tl, tr, bl, br, center


def five_crop(inpt: DType, size: List[int]) -> Tuple[DType, DType, DType, DType, DType]:
    # TODO: consider breaking BC here to return List[DType] to align this op with `ten_crop`
    if isinstance(inpt, torch.Tensor):
        output = five_crop_image_tensor(inpt, size)
        if isinstance(inpt, features.Image):
            output = tuple(features.Image.new_like(inpt, item) for item in output)  # type: ignore[assignment]
        return output
    else:  # isinstance(inpt, PIL.Image.Image):
        return five_crop_image_pil(inpt, size)


def ten_crop_image_tensor(img: torch.Tensor, size: List[int], vertical_flip: bool = False) -> List[torch.Tensor]:
    tl, tr, bl, br, center = five_crop_image_tensor(img, size)

    if vertical_flip:
        img = vertical_flip_image_tensor(img)
    else:
        img = horizontal_flip_image_tensor(img)

    tl_flip, tr_flip, bl_flip, br_flip, center_flip = five_crop_image_tensor(img, size)

    return [tl, tr, bl, br, center, tl_flip, tr_flip, bl_flip, br_flip, center_flip]


def ten_crop_image_pil(img: PIL.Image.Image, size: List[int], vertical_flip: bool = False) -> List[PIL.Image.Image]:
    tl, tr, bl, br, center = five_crop_image_pil(img, size)

    if vertical_flip:
        img = vertical_flip_image_pil(img)
    else:
        img = horizontal_flip_image_pil(img)

    tl_flip, tr_flip, bl_flip, br_flip, center_flip = five_crop_image_pil(img, size)

    return [tl, tr, bl, br, center, tl_flip, tr_flip, bl_flip, br_flip, center_flip]


def ten_crop(inpt: DType, size: List[int], *, vertical_flip: bool = False) -> List[DType]:
    if isinstance(inpt, torch.Tensor):
        output = ten_crop_image_tensor(inpt, size, vertical_flip=vertical_flip)
        if isinstance(inpt, features.Image):
            output = [features.Image.new_like(inpt, item) for item in output]
        return output
    else:  # isinstance(inpt, PIL.Image.Image):
        return ten_crop_image_pil(inpt, size, vertical_flip=vertical_flip)
