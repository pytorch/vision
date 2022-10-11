import numbers
import warnings
from typing import List, Optional, Sequence, Tuple, Union

import PIL.Image
import torch
from torchvision.prototype import features
from torchvision.transforms import functional_pil as _FP, functional_tensor as _FT
from torchvision.transforms.functional import (
    _compute_resized_output_size as __compute_resized_output_size,
    _get_inverse_affine_matrix,
    InterpolationMode,
    pil_modes_mapping,
    pil_to_tensor,
    to_pil_image,
)
from torchvision.transforms.functional_tensor import _parse_pad_padding

from ._meta import (
    convert_format_bounding_box,
    get_dimensions_image_tensor,
    get_spatial_size_image_pil,
    get_spatial_size_image_tensor,
)

horizontal_flip_image_tensor = _FT.hflip
horizontal_flip_image_pil = _FP.hflip


def horizontal_flip_mask(mask: torch.Tensor) -> torch.Tensor:
    return horizontal_flip_image_tensor(mask)


def horizontal_flip_bounding_box(
    bounding_box: torch.Tensor, format: features.BoundingBoxFormat, spatial_size: Tuple[int, int]
) -> torch.Tensor:
    shape = bounding_box.shape

    bounding_box = convert_format_bounding_box(
        bounding_box, old_format=format, new_format=features.BoundingBoxFormat.XYXY
    ).view(-1, 4)

    bounding_box[:, [0, 2]] = spatial_size[1] - bounding_box[:, [2, 0]]

    return convert_format_bounding_box(
        bounding_box, old_format=features.BoundingBoxFormat.XYXY, new_format=format, copy=False
    ).view(shape)


def horizontal_flip_video(video: torch.Tensor) -> torch.Tensor:
    return horizontal_flip_image_tensor(video)


def horizontal_flip(inpt: features.InputTypeJIT) -> features.InputTypeJIT:
    if isinstance(inpt, torch.Tensor) and (torch.jit.is_scripting() or not isinstance(inpt, features._Feature)):
        return horizontal_flip_image_tensor(inpt)
    elif isinstance(inpt, features._Feature):
        return inpt.horizontal_flip()
    else:
        return horizontal_flip_image_pil(inpt)


vertical_flip_image_tensor = _FT.vflip
vertical_flip_image_pil = _FP.vflip


def vertical_flip_mask(mask: torch.Tensor) -> torch.Tensor:
    return vertical_flip_image_tensor(mask)


def vertical_flip_bounding_box(
    bounding_box: torch.Tensor, format: features.BoundingBoxFormat, spatial_size: Tuple[int, int]
) -> torch.Tensor:
    shape = bounding_box.shape

    bounding_box = convert_format_bounding_box(
        bounding_box, old_format=format, new_format=features.BoundingBoxFormat.XYXY
    ).view(-1, 4)

    bounding_box[:, [1, 3]] = spatial_size[0] - bounding_box[:, [3, 1]]

    return convert_format_bounding_box(
        bounding_box, old_format=features.BoundingBoxFormat.XYXY, new_format=format, copy=False
    ).view(shape)


def vertical_flip_video(video: torch.Tensor) -> torch.Tensor:
    return vertical_flip_image_tensor(video)


def vertical_flip(inpt: features.InputTypeJIT) -> features.InputTypeJIT:
    if isinstance(inpt, torch.Tensor) and (torch.jit.is_scripting() or not isinstance(inpt, features._Feature)):
        return vertical_flip_image_tensor(inpt)
    elif isinstance(inpt, features._Feature):
        return inpt.vertical_flip()
    else:
        return vertical_flip_image_pil(inpt)


# We changed the names to align them with the transforms, i.e. `RandomHorizontalFlip`. Still, `hflip` and `vflip` are
# prevalent and well understood. Thus, we just alias them without deprecating the old names.
hflip = horizontal_flip
vflip = vertical_flip


def _compute_resized_output_size(
    spatial_size: Tuple[int, int], size: List[int], max_size: Optional[int] = None
) -> List[int]:
    if isinstance(size, int):
        size = [size]
    return __compute_resized_output_size(spatial_size, size=size, max_size=max_size)


def resize_image_tensor(
    image: torch.Tensor,
    size: List[int],
    interpolation: InterpolationMode = InterpolationMode.BILINEAR,
    max_size: Optional[int] = None,
    antialias: bool = False,
) -> torch.Tensor:
    num_channels, old_height, old_width = get_dimensions_image_tensor(image)
    new_height, new_width = _compute_resized_output_size((old_height, old_width), size=size, max_size=max_size)
    extra_dims = image.shape[:-3]

    if image.numel() > 0:
        image = image.view(-1, num_channels, old_height, old_width)

        image = _FT.resize(
            image,
            size=[new_height, new_width],
            interpolation=interpolation.value,
            antialias=antialias,
        )

    return image.view(extra_dims + (num_channels, new_height, new_width))


@torch.jit.unused
def resize_image_pil(
    image: PIL.Image.Image,
    size: Union[Sequence[int], int],
    interpolation: InterpolationMode = InterpolationMode.BILINEAR,
    max_size: Optional[int] = None,
) -> PIL.Image.Image:
    size = _compute_resized_output_size(image.size[::-1], size=size, max_size=max_size)  # type: ignore[arg-type]
    return _FP.resize(image, size, interpolation=pil_modes_mapping[interpolation])


def resize_mask(mask: torch.Tensor, size: List[int], max_size: Optional[int] = None) -> torch.Tensor:
    if mask.ndim < 3:
        mask = mask.unsqueeze(0)
        needs_squeeze = True
    else:
        needs_squeeze = False

    output = resize_image_tensor(mask, size=size, interpolation=InterpolationMode.NEAREST, max_size=max_size)

    if needs_squeeze:
        output = output.squeeze(0)

    return output


def resize_bounding_box(
    bounding_box: torch.Tensor, spatial_size: Tuple[int, int], size: List[int], max_size: Optional[int] = None
) -> Tuple[torch.Tensor, Tuple[int, int]]:
    old_height, old_width = spatial_size
    new_height, new_width = _compute_resized_output_size(spatial_size, size=size, max_size=max_size)
    ratios = torch.tensor((new_width / old_width, new_height / old_height), device=bounding_box.device)
    return (
        bounding_box.view(-1, 2, 2).mul(ratios).to(bounding_box.dtype).view(bounding_box.shape),
        (new_height, new_width),
    )


def resize_video(
    video: torch.Tensor,
    size: List[int],
    interpolation: InterpolationMode = InterpolationMode.BILINEAR,
    max_size: Optional[int] = None,
    antialias: bool = False,
) -> torch.Tensor:
    return resize_image_tensor(video, size=size, interpolation=interpolation, max_size=max_size, antialias=antialias)


def resize(
    inpt: features.InputTypeJIT,
    size: List[int],
    interpolation: InterpolationMode = InterpolationMode.BILINEAR,
    max_size: Optional[int] = None,
    antialias: Optional[bool] = None,
) -> features.InputTypeJIT:
    if isinstance(inpt, torch.Tensor) and (torch.jit.is_scripting() or not isinstance(inpt, features._Feature)):
        antialias = False if antialias is None else antialias
        return resize_image_tensor(inpt, size, interpolation=interpolation, max_size=max_size, antialias=antialias)
    elif isinstance(inpt, features._Feature):
        antialias = False if antialias is None else antialias
        return inpt.resize(size, interpolation=interpolation, max_size=max_size, antialias=antialias)
    else:
        if antialias is not None and not antialias:
            warnings.warn("Anti-alias option is always applied for PIL Image input. Argument antialias is ignored.")
        return resize_image_pil(inpt, size, interpolation=interpolation, max_size=max_size)


def _affine_parse_args(
    angle: Union[int, float],
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

    if center is not None:
        if not isinstance(center, (list, tuple)):
            raise TypeError("Argument center should be a sequence")
        else:
            center = [float(c) for c in center]

    return angle, translate, shear, center


def affine_image_tensor(
    image: torch.Tensor,
    angle: Union[int, float],
    translate: List[float],
    scale: float,
    shear: List[float],
    interpolation: InterpolationMode = InterpolationMode.NEAREST,
    fill: features.FillTypeJIT = None,
    center: Optional[List[float]] = None,
) -> torch.Tensor:
    if image.numel() == 0:
        return image

    num_channels, height, width = image.shape[-3:]
    extra_dims = image.shape[:-3]
    image = image.view(-1, num_channels, height, width)

    angle, translate, shear, center = _affine_parse_args(angle, translate, scale, shear, interpolation, center)

    center_f = [0.0, 0.0]
    if center is not None:
        # Center values should be in pixel coordinates but translated such that (0, 0) corresponds to image center.
        center_f = [(c - s * 0.5) for c, s in zip(center, [width, height])]

    translate_f = [float(t) for t in translate]
    matrix = _get_inverse_affine_matrix(center_f, angle, translate_f, scale, shear)

    output = _FT.affine(image, matrix, interpolation=interpolation.value, fill=fill)
    return output.view(extra_dims + (num_channels, height, width))


@torch.jit.unused
def affine_image_pil(
    image: PIL.Image.Image,
    angle: Union[int, float],
    translate: List[float],
    scale: float,
    shear: List[float],
    interpolation: InterpolationMode = InterpolationMode.NEAREST,
    fill: features.FillTypeJIT = None,
    center: Optional[List[float]] = None,
) -> PIL.Image.Image:
    angle, translate, shear, center = _affine_parse_args(angle, translate, scale, shear, interpolation, center)

    # center = (img_size[0] * 0.5 + 0.5, img_size[1] * 0.5 + 0.5)
    # it is visually better to estimate the center without 0.5 offset
    # otherwise image rotated by 90 degrees is shifted vs output image of torch.rot90 or F_t.affine
    if center is None:
        height, width = get_spatial_size_image_pil(image)
        center = [width * 0.5, height * 0.5]
    matrix = _get_inverse_affine_matrix(center, angle, translate, scale, shear)

    return _FP.affine(image, matrix, interpolation=pil_modes_mapping[interpolation], fill=fill)


def _affine_bounding_box_xyxy(
    bounding_box: torch.Tensor,
    spatial_size: Tuple[int, int],
    angle: Union[int, float],
    translate: List[float],
    scale: float,
    shear: List[float],
    center: Optional[List[float]] = None,
    expand: bool = False,
) -> Tuple[torch.Tensor, Tuple[int, int]]:
    angle, translate, shear, center = _affine_parse_args(
        angle, translate, scale, shear, InterpolationMode.NEAREST, center
    )

    if center is None:
        height, width = spatial_size
        center = [width * 0.5, height * 0.5]

    dtype = bounding_box.dtype if torch.is_floating_point(bounding_box) else torch.float32
    device = bounding_box.device

    affine_vector = _get_inverse_affine_matrix(center, angle, translate, scale, shear, inverted=False)
    transposed_affine_matrix = (
        torch.tensor(
            affine_vector,
            dtype=dtype,
            device=device,
        )
        .view(2, 3)
        .T
    )
    # 1) Let's transform bboxes into a tensor of 4 points (top-left, top-right, bottom-left, bottom-right corners).
    # Tensor of points has shape (N * 4, 3), where N is the number of bboxes
    # Single point structure is similar to
    # [(xmin, ymin, 1), (xmax, ymin, 1), (xmax, ymax, 1), (xmin, ymax, 1)]
    points = bounding_box[:, [[0, 1], [2, 1], [2, 3], [0, 3]]].view(-1, 2)
    points = torch.cat([points, torch.ones(points.shape[0], 1, device=points.device)], dim=-1)
    # 2) Now let's transform the points using affine matrix
    transformed_points = torch.matmul(points, transposed_affine_matrix)
    # 3) Reshape transformed points to [N boxes, 4 points, x/y coords]
    # and compute bounding box from 4 transformed points:
    transformed_points = transformed_points.view(-1, 4, 2)
    out_bbox_mins, _ = torch.min(transformed_points, dim=1)
    out_bbox_maxs, _ = torch.max(transformed_points, dim=1)
    out_bboxes = torch.cat([out_bbox_mins, out_bbox_maxs], dim=1)

    if expand:
        # Compute minimum point for transformed image frame:
        # Points are Top-Left, Top-Right, Bottom-Left, Bottom-Right points.
        height, width = spatial_size
        points = torch.tensor(
            [
                [0.0, 0.0, 1.0],
                [0.0, float(height), 1.0],
                [float(width), float(height), 1.0],
                [float(width), 0.0, 1.0],
            ],
            dtype=dtype,
            device=device,
        )
        new_points = torch.matmul(points, transposed_affine_matrix)
        tr, _ = torch.min(new_points, dim=0, keepdim=True)
        # Translate bounding boxes
        out_bboxes[:, 0::2] = out_bboxes[:, 0::2] - tr[:, 0]
        out_bboxes[:, 1::2] = out_bboxes[:, 1::2] - tr[:, 1]
        # Estimate meta-data for image with inverted=True and with center=[0,0]
        affine_vector = _get_inverse_affine_matrix([0.0, 0.0], angle, translate, scale, shear)
        new_width, new_height = _FT._compute_affine_output_size(affine_vector, width, height)
        spatial_size = (new_height, new_width)

    return out_bboxes.to(bounding_box.dtype), spatial_size


def affine_bounding_box(
    bounding_box: torch.Tensor,
    format: features.BoundingBoxFormat,
    spatial_size: Tuple[int, int],
    angle: Union[int, float],
    translate: List[float],
    scale: float,
    shear: List[float],
    center: Optional[List[float]] = None,
) -> torch.Tensor:
    original_shape = bounding_box.shape
    bounding_box = convert_format_bounding_box(
        bounding_box, old_format=format, new_format=features.BoundingBoxFormat.XYXY
    ).view(-1, 4)

    out_bboxes, _ = _affine_bounding_box_xyxy(bounding_box, spatial_size, angle, translate, scale, shear, center)

    # out_bboxes should be of shape [N boxes, 4]

    return convert_format_bounding_box(
        out_bboxes, old_format=features.BoundingBoxFormat.XYXY, new_format=format, copy=False
    ).view(original_shape)


def affine_mask(
    mask: torch.Tensor,
    angle: Union[int, float],
    translate: List[float],
    scale: float,
    shear: List[float],
    fill: features.FillTypeJIT = None,
    center: Optional[List[float]] = None,
) -> torch.Tensor:
    if mask.ndim < 3:
        mask = mask.unsqueeze(0)
        needs_squeeze = True
    else:
        needs_squeeze = False

    output = affine_image_tensor(
        mask,
        angle=angle,
        translate=translate,
        scale=scale,
        shear=shear,
        interpolation=InterpolationMode.NEAREST,
        fill=fill,
        center=center,
    )

    if needs_squeeze:
        output = output.squeeze(0)

    return output


def affine_video(
    video: torch.Tensor,
    angle: Union[int, float],
    translate: List[float],
    scale: float,
    shear: List[float],
    interpolation: InterpolationMode = InterpolationMode.NEAREST,
    fill: features.FillTypeJIT = None,
    center: Optional[List[float]] = None,
) -> torch.Tensor:
    return affine_image_tensor(
        video,
        angle=angle,
        translate=translate,
        scale=scale,
        shear=shear,
        interpolation=interpolation,
        fill=fill,
        center=center,
    )


def _convert_fill_arg(fill: features.FillType) -> features.FillTypeJIT:
    # Fill = 0 is not equivalent to None, https://github.com/pytorch/vision/issues/6517
    # So, we can't reassign fill to 0
    # if fill is None:
    #     fill = 0
    if fill is None:
        return fill

    # This cast does Sequence -> List[float] to please mypy and torch.jit.script
    if not isinstance(fill, (int, float)):
        fill = [float(v) for v in list(fill)]
    return fill


def affine(
    inpt: features.InputTypeJIT,
    angle: Union[int, float],
    translate: List[float],
    scale: float,
    shear: List[float],
    interpolation: InterpolationMode = InterpolationMode.NEAREST,
    fill: features.FillTypeJIT = None,
    center: Optional[List[float]] = None,
) -> features.InputTypeJIT:
    # TODO: consider deprecating integers from angle and shear on the future
    if isinstance(inpt, torch.Tensor) and (torch.jit.is_scripting() or not isinstance(inpt, features._Feature)):
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
    elif isinstance(inpt, features._Feature):
        return inpt.affine(
            angle, translate=translate, scale=scale, shear=shear, interpolation=interpolation, fill=fill, center=center
        )
    else:
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


def rotate_image_tensor(
    image: torch.Tensor,
    angle: float,
    interpolation: InterpolationMode = InterpolationMode.NEAREST,
    expand: bool = False,
    fill: features.FillTypeJIT = None,
    center: Optional[List[float]] = None,
) -> torch.Tensor:
    num_channels, height, width = image.shape[-3:]
    extra_dims = image.shape[:-3]

    center_f = [0.0, 0.0]
    if center is not None:
        if expand:
            warnings.warn("The provided center argument has no effect on the result if expand is True")
        else:
            # Center values should be in pixel coordinates but translated such that (0, 0) corresponds to image center.
            center_f = [(c - s * 0.5) for c, s in zip(center, [width, height])]

    # due to current incoherence of rotation angle direction between affine and rotate implementations
    # we need to set -angle.
    matrix = _get_inverse_affine_matrix(center_f, -angle, [0.0, 0.0], 1.0, [0.0, 0.0])

    if image.numel() > 0:
        image = _FT.rotate(
            image.view(-1, num_channels, height, width),
            matrix,
            interpolation=interpolation.value,
            expand=expand,
            fill=fill,
        )
        new_height, new_width = image.shape[-2:]
    else:
        new_width, new_height = _FT._compute_affine_output_size(matrix, width, height) if expand else (width, height)

    return image.view(extra_dims + (num_channels, new_height, new_width))


@torch.jit.unused
def rotate_image_pil(
    image: PIL.Image.Image,
    angle: float,
    interpolation: InterpolationMode = InterpolationMode.NEAREST,
    expand: bool = False,
    fill: features.FillTypeJIT = None,
    center: Optional[List[float]] = None,
) -> PIL.Image.Image:
    if center is not None and expand:
        warnings.warn("The provided center argument has no effect on the result if expand is True")
        center = None

    return _FP.rotate(
        image, angle, interpolation=pil_modes_mapping[interpolation], expand=expand, fill=fill, center=center
    )


def rotate_bounding_box(
    bounding_box: torch.Tensor,
    format: features.BoundingBoxFormat,
    spatial_size: Tuple[int, int],
    angle: float,
    expand: bool = False,
    center: Optional[List[float]] = None,
) -> Tuple[torch.Tensor, Tuple[int, int]]:
    if center is not None and expand:
        warnings.warn("The provided center argument has no effect on the result if expand is True")
        center = None

    original_shape = bounding_box.shape
    bounding_box = convert_format_bounding_box(
        bounding_box, old_format=format, new_format=features.BoundingBoxFormat.XYXY
    ).view(-1, 4)

    out_bboxes, spatial_size = _affine_bounding_box_xyxy(
        bounding_box,
        spatial_size,
        angle=-angle,
        translate=[0.0, 0.0],
        scale=1.0,
        shear=[0.0, 0.0],
        center=center,
        expand=expand,
    )

    return (
        convert_format_bounding_box(
            out_bboxes, old_format=features.BoundingBoxFormat.XYXY, new_format=format, copy=False
        ).view(original_shape),
        spatial_size,
    )


def rotate_mask(
    mask: torch.Tensor,
    angle: float,
    expand: bool = False,
    fill: features.FillTypeJIT = None,
    center: Optional[List[float]] = None,
) -> torch.Tensor:
    if mask.ndim < 3:
        mask = mask.unsqueeze(0)
        needs_squeeze = True
    else:
        needs_squeeze = False

    output = rotate_image_tensor(
        mask,
        angle=angle,
        expand=expand,
        interpolation=InterpolationMode.NEAREST,
        fill=fill,
        center=center,
    )

    if needs_squeeze:
        output = output.squeeze(0)

    return output


def rotate_video(
    video: torch.Tensor,
    angle: float,
    interpolation: InterpolationMode = InterpolationMode.NEAREST,
    expand: bool = False,
    fill: features.FillTypeJIT = None,
    center: Optional[List[float]] = None,
) -> torch.Tensor:
    return rotate_image_tensor(video, angle, interpolation=interpolation, expand=expand, fill=fill, center=center)


def rotate(
    inpt: features.InputTypeJIT,
    angle: float,
    interpolation: InterpolationMode = InterpolationMode.NEAREST,
    expand: bool = False,
    fill: features.FillTypeJIT = None,
    center: Optional[List[float]] = None,
) -> features.InputTypeJIT:
    if isinstance(inpt, torch.Tensor) and (torch.jit.is_scripting() or not isinstance(inpt, features._Feature)):
        return rotate_image_tensor(inpt, angle, interpolation=interpolation, expand=expand, fill=fill, center=center)
    elif isinstance(inpt, features._Feature):
        return inpt.rotate(angle, interpolation=interpolation, expand=expand, fill=fill, center=center)
    else:
        return rotate_image_pil(inpt, angle, interpolation=interpolation, expand=expand, fill=fill, center=center)


pad_image_pil = _FP.pad


def pad_image_tensor(
    image: torch.Tensor,
    padding: Union[int, List[int]],
    fill: features.FillTypeJIT = None,
    padding_mode: str = "constant",
) -> torch.Tensor:
    if fill is None:
        # This is a JIT workaround
        return _pad_with_scalar_fill(image, padding, fill=None, padding_mode=padding_mode)
    elif isinstance(fill, (int, float)) or len(fill) == 1:
        fill_number = fill[0] if isinstance(fill, list) else fill
        return _pad_with_scalar_fill(image, padding, fill=fill_number, padding_mode=padding_mode)
    else:
        return _pad_with_vector_fill(image, padding, fill=fill, padding_mode=padding_mode)


def _pad_with_scalar_fill(
    image: torch.Tensor,
    padding: Union[int, List[int]],
    fill: Union[int, float, None],
    padding_mode: str = "constant",
) -> torch.Tensor:
    num_channels, height, width = image.shape[-3:]
    extra_dims = image.shape[:-3]

    if image.numel() > 0:
        image = _FT.pad(
            img=image.view(-1, num_channels, height, width), padding=padding, fill=fill, padding_mode=padding_mode
        )
        new_height, new_width = image.shape[-2:]
    else:
        left, right, top, bottom = _FT._parse_pad_padding(padding)
        new_height = height + top + bottom
        new_width = width + left + right

    return image.view(extra_dims + (num_channels, new_height, new_width))


# TODO: This should be removed once pytorch pad supports non-scalar padding values
def _pad_with_vector_fill(
    image: torch.Tensor,
    padding: Union[int, List[int]],
    fill: List[float],
    padding_mode: str = "constant",
) -> torch.Tensor:
    if padding_mode != "constant":
        raise ValueError(f"Padding mode '{padding_mode}' is not supported if fill is not scalar")

    output = _pad_with_scalar_fill(image, padding, fill=0, padding_mode="constant")
    left, right, top, bottom = _parse_pad_padding(padding)
    fill = torch.tensor(fill, dtype=image.dtype, device=image.device).view(-1, 1, 1)

    if top > 0:
        output[..., :top, :] = fill
    if left > 0:
        output[..., :, :left] = fill
    if bottom > 0:
        output[..., -bottom:, :] = fill
    if right > 0:
        output[..., :, -right:] = fill
    return output


def pad_mask(
    mask: torch.Tensor,
    padding: Union[int, List[int]],
    padding_mode: str = "constant",
    fill: features.FillTypeJIT = None,
) -> torch.Tensor:
    if fill is None:
        fill = 0

    if isinstance(fill, list):
        raise ValueError("Non-scalar fill value is not supported")

    if mask.ndim < 3:
        mask = mask.unsqueeze(0)
        needs_squeeze = True
    else:
        needs_squeeze = False

    output = pad_image_tensor(mask, padding=padding, fill=fill, padding_mode=padding_mode)

    if needs_squeeze:
        output = output.squeeze(0)

    return output


def pad_bounding_box(
    bounding_box: torch.Tensor,
    format: features.BoundingBoxFormat,
    spatial_size: Tuple[int, int],
    padding: Union[int, List[int]],
    padding_mode: str = "constant",
) -> Tuple[torch.Tensor, Tuple[int, int]]:
    if padding_mode not in ["constant"]:
        # TODO: add support of other padding modes
        raise ValueError(f"Padding mode '{padding_mode}' is not supported with bounding boxes")

    left, right, top, bottom = _parse_pad_padding(padding)

    bounding_box = bounding_box.clone()

    # this works without conversion since padding only affects xy coordinates
    bounding_box[..., 0] += left
    bounding_box[..., 1] += top
    if format == features.BoundingBoxFormat.XYXY:
        bounding_box[..., 2] += left
        bounding_box[..., 3] += top

    height, width = spatial_size
    height += top + bottom
    width += left + right

    return bounding_box, (height, width)


def pad_video(
    video: torch.Tensor,
    padding: Union[int, List[int]],
    fill: features.FillTypeJIT = None,
    padding_mode: str = "constant",
) -> torch.Tensor:
    return pad_image_tensor(video, padding, fill=fill, padding_mode=padding_mode)


def pad(
    inpt: features.InputTypeJIT,
    padding: Union[int, List[int]],
    fill: features.FillTypeJIT = None,
    padding_mode: str = "constant",
) -> features.InputTypeJIT:
    if isinstance(inpt, torch.Tensor) and (torch.jit.is_scripting() or not isinstance(inpt, features._Feature)):
        return pad_image_tensor(inpt, padding, fill=fill, padding_mode=padding_mode)

    elif isinstance(inpt, features._Feature):
        return inpt.pad(padding, fill=fill, padding_mode=padding_mode)
    else:
        return pad_image_pil(inpt, padding, fill=fill, padding_mode=padding_mode)


crop_image_tensor = _FT.crop
crop_image_pil = _FP.crop


def crop_bounding_box(
    bounding_box: torch.Tensor,
    format: features.BoundingBoxFormat,
    top: int,
    left: int,
    height: int,
    width: int,
) -> Tuple[torch.Tensor, Tuple[int, int]]:
    bounding_box = convert_format_bounding_box(
        bounding_box, old_format=format, new_format=features.BoundingBoxFormat.XYXY
    )

    # Crop or implicit pad if left and/or top have negative values:
    bounding_box[..., 0::2] -= left
    bounding_box[..., 1::2] -= top

    return (
        convert_format_bounding_box(
            bounding_box, old_format=features.BoundingBoxFormat.XYXY, new_format=format, copy=False
        ),
        (height, width),
    )


def crop_mask(mask: torch.Tensor, top: int, left: int, height: int, width: int) -> torch.Tensor:
    return crop_image_tensor(mask, top, left, height, width)


def crop_video(video: torch.Tensor, top: int, left: int, height: int, width: int) -> torch.Tensor:
    return crop_image_tensor(video, top, left, height, width)


def crop(inpt: features.InputTypeJIT, top: int, left: int, height: int, width: int) -> features.InputTypeJIT:
    if isinstance(inpt, torch.Tensor) and (torch.jit.is_scripting() or not isinstance(inpt, features._Feature)):
        return crop_image_tensor(inpt, top, left, height, width)
    elif isinstance(inpt, features._Feature):
        return inpt.crop(top, left, height, width)
    else:
        return crop_image_pil(inpt, top, left, height, width)


def perspective_image_tensor(
    image: torch.Tensor,
    perspective_coeffs: List[float],
    interpolation: InterpolationMode = InterpolationMode.BILINEAR,
    fill: features.FillTypeJIT = None,
) -> torch.Tensor:
    if image.numel() == 0:
        return image

    shape = image.shape

    if image.ndim > 4:
        image = image.view((-1,) + shape[-3:])
        needs_unsquash = True
    else:
        needs_unsquash = False

    output = _FT.perspective(image, perspective_coeffs, interpolation=interpolation.value, fill=fill)

    if needs_unsquash:
        output = output.view(shape)

    return output


@torch.jit.unused
def perspective_image_pil(
    image: PIL.Image.Image,
    perspective_coeffs: List[float],
    interpolation: InterpolationMode = InterpolationMode.BICUBIC,
    fill: features.FillTypeJIT = None,
) -> PIL.Image.Image:
    return _FP.perspective(image, perspective_coeffs, interpolation=pil_modes_mapping[interpolation], fill=fill)


def perspective_bounding_box(
    bounding_box: torch.Tensor,
    format: features.BoundingBoxFormat,
    perspective_coeffs: List[float],
) -> torch.Tensor:

    if len(perspective_coeffs) != 8:
        raise ValueError("Argument perspective_coeffs should have 8 float values")

    original_shape = bounding_box.shape
    bounding_box = convert_format_bounding_box(
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
    out_bboxes = torch.cat([out_bbox_mins, out_bbox_maxs], dim=1).to(bounding_box.dtype)

    # out_bboxes should be of shape [N boxes, 4]

    return convert_format_bounding_box(
        out_bboxes, old_format=features.BoundingBoxFormat.XYXY, new_format=format, copy=False
    ).view(original_shape)


def perspective_mask(
    mask: torch.Tensor,
    perspective_coeffs: List[float],
    fill: features.FillTypeJIT = None,
) -> torch.Tensor:
    if mask.ndim < 3:
        mask = mask.unsqueeze(0)
        needs_squeeze = True
    else:
        needs_squeeze = False

    output = perspective_image_tensor(
        mask, perspective_coeffs=perspective_coeffs, interpolation=InterpolationMode.NEAREST, fill=fill
    )

    if needs_squeeze:
        output = output.squeeze(0)

    return output


def perspective_video(
    video: torch.Tensor,
    perspective_coeffs: List[float],
    interpolation: InterpolationMode = InterpolationMode.BILINEAR,
    fill: features.FillTypeJIT = None,
) -> torch.Tensor:
    return perspective_image_tensor(video, perspective_coeffs, interpolation=interpolation, fill=fill)


def perspective(
    inpt: features.InputTypeJIT,
    perspective_coeffs: List[float],
    interpolation: InterpolationMode = InterpolationMode.BILINEAR,
    fill: features.FillTypeJIT = None,
) -> features.InputTypeJIT:
    if isinstance(inpt, torch.Tensor) and (torch.jit.is_scripting() or not isinstance(inpt, features._Feature)):
        return perspective_image_tensor(inpt, perspective_coeffs, interpolation=interpolation, fill=fill)
    elif isinstance(inpt, features._Feature):
        return inpt.perspective(perspective_coeffs, interpolation=interpolation, fill=fill)
    else:
        return perspective_image_pil(inpt, perspective_coeffs, interpolation=interpolation, fill=fill)


def elastic_image_tensor(
    image: torch.Tensor,
    displacement: torch.Tensor,
    interpolation: InterpolationMode = InterpolationMode.BILINEAR,
    fill: features.FillTypeJIT = None,
) -> torch.Tensor:
    if image.numel() == 0:
        return image

    shape = image.shape

    if image.ndim > 4:
        image = image.view((-1,) + shape[-3:])
        needs_unsquash = True
    else:
        needs_unsquash = False

    output = _FT.elastic_transform(image, displacement, interpolation=interpolation.value, fill=fill)

    if needs_unsquash:
        output = output.view(shape)

    return output


@torch.jit.unused
def elastic_image_pil(
    image: PIL.Image.Image,
    displacement: torch.Tensor,
    interpolation: InterpolationMode = InterpolationMode.BILINEAR,
    fill: features.FillTypeJIT = None,
) -> PIL.Image.Image:
    t_img = pil_to_tensor(image)
    output = elastic_image_tensor(t_img, displacement, interpolation=interpolation, fill=fill)
    return to_pil_image(output, mode=image.mode)


def elastic_bounding_box(
    bounding_box: torch.Tensor,
    format: features.BoundingBoxFormat,
    displacement: torch.Tensor,
) -> torch.Tensor:
    # TODO: add in docstring about approximation we are doing for grid inversion
    displacement = displacement.to(bounding_box.device)

    original_shape = bounding_box.shape
    bounding_box = convert_format_bounding_box(
        bounding_box, old_format=format, new_format=features.BoundingBoxFormat.XYXY
    ).view(-1, 4)

    # Question (vfdev-5): should we rely on good displacement shape and fetch image size from it
    # Or add spatial_size arg and check displacement shape
    spatial_size = displacement.shape[-3], displacement.shape[-2]

    id_grid = _FT._create_identity_grid(list(spatial_size)).to(bounding_box.device)
    # We construct an approximation of inverse grid as inv_grid = id_grid - displacement
    # This is not an exact inverse of the grid
    inv_grid = id_grid - displacement

    # Get points from bboxes
    points = bounding_box[:, [[0, 1], [2, 1], [2, 3], [0, 3]]].view(-1, 2)
    index_x = torch.floor(points[:, 0] + 0.5).to(dtype=torch.long)
    index_y = torch.floor(points[:, 1] + 0.5).to(dtype=torch.long)
    # Transform points:
    t_size = torch.tensor(spatial_size[::-1], device=displacement.device, dtype=displacement.dtype)
    transformed_points = (inv_grid[0, index_y, index_x, :] + 1) * 0.5 * t_size - 0.5

    transformed_points = transformed_points.view(-1, 4, 2)
    out_bbox_mins, _ = torch.min(transformed_points, dim=1)
    out_bbox_maxs, _ = torch.max(transformed_points, dim=1)
    out_bboxes = torch.cat([out_bbox_mins, out_bbox_maxs], dim=1).to(bounding_box.dtype)

    return convert_format_bounding_box(
        out_bboxes, old_format=features.BoundingBoxFormat.XYXY, new_format=format, copy=False
    ).view(original_shape)


def elastic_mask(
    mask: torch.Tensor,
    displacement: torch.Tensor,
    fill: features.FillTypeJIT = None,
) -> torch.Tensor:
    if mask.ndim < 3:
        mask = mask.unsqueeze(0)
        needs_squeeze = True
    else:
        needs_squeeze = False

    output = elastic_image_tensor(mask, displacement=displacement, interpolation=InterpolationMode.NEAREST, fill=fill)

    if needs_squeeze:
        output = output.squeeze(0)

    return output


def elastic_video(
    video: torch.Tensor,
    displacement: torch.Tensor,
    interpolation: InterpolationMode = InterpolationMode.BILINEAR,
    fill: features.FillTypeJIT = None,
) -> torch.Tensor:
    return elastic_image_tensor(video, displacement, interpolation=interpolation, fill=fill)


def elastic(
    inpt: features.InputTypeJIT,
    displacement: torch.Tensor,
    interpolation: InterpolationMode = InterpolationMode.BILINEAR,
    fill: features.FillTypeJIT = None,
) -> features.InputTypeJIT:
    if isinstance(inpt, torch.Tensor) and (torch.jit.is_scripting() or not isinstance(inpt, features._Feature)):
        return elastic_image_tensor(inpt, displacement, interpolation=interpolation, fill=fill)
    elif isinstance(inpt, features._Feature):
        return inpt.elastic(displacement, interpolation=interpolation, fill=fill)
    else:
        return elastic_image_pil(inpt, displacement, interpolation=interpolation, fill=fill)


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


def center_crop_image_tensor(image: torch.Tensor, output_size: List[int]) -> torch.Tensor:
    crop_height, crop_width = _center_crop_parse_output_size(output_size)
    image_height, image_width = get_spatial_size_image_tensor(image)

    if crop_height > image_height or crop_width > image_width:
        padding_ltrb = _center_crop_compute_padding(crop_height, crop_width, image_height, image_width)
        image = pad_image_tensor(image, padding_ltrb, fill=0)

        image_height, image_width = get_spatial_size_image_tensor(image)
        if crop_width == image_width and crop_height == image_height:
            return image

    crop_top, crop_left = _center_crop_compute_crop_anchor(crop_height, crop_width, image_height, image_width)
    return crop_image_tensor(image, crop_top, crop_left, crop_height, crop_width)


@torch.jit.unused
def center_crop_image_pil(image: PIL.Image.Image, output_size: List[int]) -> PIL.Image.Image:
    crop_height, crop_width = _center_crop_parse_output_size(output_size)
    image_height, image_width = get_spatial_size_image_pil(image)

    if crop_height > image_height or crop_width > image_width:
        padding_ltrb = _center_crop_compute_padding(crop_height, crop_width, image_height, image_width)
        image = pad_image_pil(image, padding_ltrb, fill=0)

        image_height, image_width = get_spatial_size_image_pil(image)
        if crop_width == image_width and crop_height == image_height:
            return image

    crop_top, crop_left = _center_crop_compute_crop_anchor(crop_height, crop_width, image_height, image_width)
    return crop_image_pil(image, crop_top, crop_left, crop_height, crop_width)


def center_crop_bounding_box(
    bounding_box: torch.Tensor,
    format: features.BoundingBoxFormat,
    spatial_size: Tuple[int, int],
    output_size: List[int],
) -> Tuple[torch.Tensor, Tuple[int, int]]:
    crop_height, crop_width = _center_crop_parse_output_size(output_size)
    crop_top, crop_left = _center_crop_compute_crop_anchor(crop_height, crop_width, *spatial_size)
    return crop_bounding_box(bounding_box, format, top=crop_top, left=crop_left, height=crop_height, width=crop_width)


def center_crop_mask(mask: torch.Tensor, output_size: List[int]) -> torch.Tensor:
    if mask.ndim < 3:
        mask = mask.unsqueeze(0)
        needs_squeeze = True
    else:
        needs_squeeze = False

    output = center_crop_image_tensor(image=mask, output_size=output_size)

    if needs_squeeze:
        output = output.squeeze(0)

    return output


def center_crop_video(video: torch.Tensor, output_size: List[int]) -> torch.Tensor:
    return center_crop_image_tensor(video, output_size)


def center_crop(inpt: features.InputTypeJIT, output_size: List[int]) -> features.InputTypeJIT:
    if isinstance(inpt, torch.Tensor) and (torch.jit.is_scripting() or not isinstance(inpt, features._Feature)):
        return center_crop_image_tensor(inpt, output_size)
    elif isinstance(inpt, features._Feature):
        return inpt.center_crop(output_size)
    else:
        return center_crop_image_pil(inpt, output_size)


def resized_crop_image_tensor(
    image: torch.Tensor,
    top: int,
    left: int,
    height: int,
    width: int,
    size: List[int],
    interpolation: InterpolationMode = InterpolationMode.BILINEAR,
    antialias: bool = False,
) -> torch.Tensor:
    image = crop_image_tensor(image, top, left, height, width)
    return resize_image_tensor(image, size, interpolation=interpolation, antialias=antialias)


@torch.jit.unused
def resized_crop_image_pil(
    image: PIL.Image.Image,
    top: int,
    left: int,
    height: int,
    width: int,
    size: List[int],
    interpolation: InterpolationMode = InterpolationMode.BILINEAR,
) -> PIL.Image.Image:
    image = crop_image_pil(image, top, left, height, width)
    return resize_image_pil(image, size, interpolation=interpolation)


def resized_crop_bounding_box(
    bounding_box: torch.Tensor,
    format: features.BoundingBoxFormat,
    top: int,
    left: int,
    height: int,
    width: int,
    size: List[int],
) -> Tuple[torch.Tensor, Tuple[int, int]]:
    bounding_box, _ = crop_bounding_box(bounding_box, format, top, left, height, width)
    return resize_bounding_box(bounding_box, (height, width), size)


def resized_crop_mask(
    mask: torch.Tensor,
    top: int,
    left: int,
    height: int,
    width: int,
    size: List[int],
) -> torch.Tensor:
    mask = crop_mask(mask, top, left, height, width)
    return resize_mask(mask, size)


def resized_crop_video(
    video: torch.Tensor,
    top: int,
    left: int,
    height: int,
    width: int,
    size: List[int],
    interpolation: InterpolationMode = InterpolationMode.BILINEAR,
    antialias: bool = False,
) -> torch.Tensor:
    return resized_crop_image_tensor(
        video, top, left, height, width, antialias=antialias, size=size, interpolation=interpolation
    )


def resized_crop(
    inpt: features.InputTypeJIT,
    top: int,
    left: int,
    height: int,
    width: int,
    size: List[int],
    interpolation: InterpolationMode = InterpolationMode.BILINEAR,
    antialias: Optional[bool] = None,
) -> features.InputTypeJIT:
    if isinstance(inpt, torch.Tensor) and (torch.jit.is_scripting() or not isinstance(inpt, features._Feature)):
        antialias = False if antialias is None else antialias
        return resized_crop_image_tensor(
            inpt, top, left, height, width, antialias=antialias, size=size, interpolation=interpolation
        )
    elif isinstance(inpt, features._Feature):
        antialias = False if antialias is None else antialias
        return inpt.resized_crop(top, left, height, width, antialias=antialias, size=size, interpolation=interpolation)
    else:
        return resized_crop_image_pil(inpt, top, left, height, width, size=size, interpolation=interpolation)


def _parse_five_crop_size(size: List[int]) -> List[int]:
    if isinstance(size, numbers.Number):
        size = [int(size), int(size)]
    elif isinstance(size, (tuple, list)) and len(size) == 1:
        size = [size[0], size[0]]

    if len(size) != 2:
        raise ValueError("Please provide only two dimensions (h, w) for size.")

    return size


def five_crop_image_tensor(
    image: torch.Tensor, size: List[int]
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    crop_height, crop_width = _parse_five_crop_size(size)
    image_height, image_width = get_spatial_size_image_tensor(image)

    if crop_width > image_width or crop_height > image_height:
        msg = "Requested crop size {} is bigger than input size {}"
        raise ValueError(msg.format(size, (image_height, image_width)))

    tl = crop_image_tensor(image, 0, 0, crop_height, crop_width)
    tr = crop_image_tensor(image, 0, image_width - crop_width, crop_height, crop_width)
    bl = crop_image_tensor(image, image_height - crop_height, 0, crop_height, crop_width)
    br = crop_image_tensor(image, image_height - crop_height, image_width - crop_width, crop_height, crop_width)
    center = center_crop_image_tensor(image, [crop_height, crop_width])

    return tl, tr, bl, br, center


@torch.jit.unused
def five_crop_image_pil(
    image: PIL.Image.Image, size: List[int]
) -> Tuple[PIL.Image.Image, PIL.Image.Image, PIL.Image.Image, PIL.Image.Image, PIL.Image.Image]:
    crop_height, crop_width = _parse_five_crop_size(size)
    image_height, image_width = get_spatial_size_image_pil(image)

    if crop_width > image_width or crop_height > image_height:
        msg = "Requested crop size {} is bigger than input size {}"
        raise ValueError(msg.format(size, (image_height, image_width)))

    tl = crop_image_pil(image, 0, 0, crop_height, crop_width)
    tr = crop_image_pil(image, 0, image_width - crop_width, crop_height, crop_width)
    bl = crop_image_pil(image, image_height - crop_height, 0, crop_height, crop_width)
    br = crop_image_pil(image, image_height - crop_height, image_width - crop_width, crop_height, crop_width)
    center = center_crop_image_pil(image, [crop_height, crop_width])

    return tl, tr, bl, br, center


def five_crop_video(
    video: torch.Tensor, size: List[int]
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    return five_crop_image_tensor(video, size)


def five_crop(
    inpt: features.ImageOrVideoTypeJIT, size: List[int]
) -> Tuple[
    features.ImageOrVideoTypeJIT,
    features.ImageOrVideoTypeJIT,
    features.ImageOrVideoTypeJIT,
    features.ImageOrVideoTypeJIT,
    features.ImageOrVideoTypeJIT,
]:
    # TODO: consider breaking BC here to return List[features.ImageOrVideoTypeJIT] to align this op with `ten_crop`
    if isinstance(inpt, torch.Tensor):
        output = five_crop_image_tensor(inpt, size)
        if not torch.jit.is_scripting() and isinstance(inpt, (features.Image, features.Video)):
            tmp = tuple(inpt.wrap_like(inpt, item) for item in output)  # type: ignore[arg-type]
            output = tmp  # type: ignore[assignment]
        return output
    else:  # isinstance(inpt, PIL.Image.Image):
        return five_crop_image_pil(inpt, size)


def ten_crop_image_tensor(image: torch.Tensor, size: List[int], vertical_flip: bool = False) -> List[torch.Tensor]:
    tl, tr, bl, br, center = five_crop_image_tensor(image, size)

    if vertical_flip:
        image = vertical_flip_image_tensor(image)
    else:
        image = horizontal_flip_image_tensor(image)

    tl_flip, tr_flip, bl_flip, br_flip, center_flip = five_crop_image_tensor(image, size)

    return [tl, tr, bl, br, center, tl_flip, tr_flip, bl_flip, br_flip, center_flip]


@torch.jit.unused
def ten_crop_image_pil(image: PIL.Image.Image, size: List[int], vertical_flip: bool = False) -> List[PIL.Image.Image]:
    tl, tr, bl, br, center = five_crop_image_pil(image, size)

    if vertical_flip:
        image = vertical_flip_image_pil(image)
    else:
        image = horizontal_flip_image_pil(image)

    tl_flip, tr_flip, bl_flip, br_flip, center_flip = five_crop_image_pil(image, size)

    return [tl, tr, bl, br, center, tl_flip, tr_flip, bl_flip, br_flip, center_flip]


def ten_crop_video(video: torch.Tensor, size: List[int], vertical_flip: bool = False) -> List[torch.Tensor]:
    return ten_crop_image_tensor(video, size, vertical_flip=vertical_flip)


def ten_crop(
    inpt: features.ImageOrVideoTypeJIT, size: List[int], vertical_flip: bool = False
) -> List[features.ImageOrVideoTypeJIT]:
    if isinstance(inpt, torch.Tensor):
        output = ten_crop_image_tensor(inpt, size, vertical_flip=vertical_flip)
        if not torch.jit.is_scripting() and isinstance(inpt, (features.Image, features.Video)):
            output = [inpt.wrap_like(inpt, item) for item in output]  # type: ignore[arg-type]
        return output
    else:  # isinstance(inpt, PIL.Image.Image):
        return ten_crop_image_pil(inpt, size, vertical_flip=vertical_flip)
