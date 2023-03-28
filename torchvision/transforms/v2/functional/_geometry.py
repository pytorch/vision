import math
import numbers
import warnings
from typing import List, Optional, Sequence, Tuple, Union

import PIL.Image
import torch
from torch.nn.functional import grid_sample, interpolate, pad as torch_pad

from torchvision import datapoints
from torchvision.transforms import _functional_pil as _FP
from torchvision.transforms._functional_tensor import _pad_symmetric
from torchvision.transforms.functional import (
    _check_antialias,
    _compute_resized_output_size as __compute_resized_output_size,
    _get_perspective_coeffs,
    _interpolation_modes_from_int,
    InterpolationMode,
    pil_modes_mapping,
    pil_to_tensor,
    to_pil_image,
)

from torchvision.utils import _log_api_usage_once

from ._meta import clamp_bounding_box, convert_format_bounding_box, get_spatial_size_image_pil

from ._utils import is_simple_tensor


def _check_interpolation(interpolation: Union[InterpolationMode, int]) -> InterpolationMode:
    if isinstance(interpolation, int):
        interpolation = _interpolation_modes_from_int(interpolation)
    elif not isinstance(interpolation, InterpolationMode):
        raise ValueError(
            f"Argument interpolation should be an `InterpolationMode` or a corresponding Pillow integer constant, "
            f"but got {interpolation}."
        )
    return interpolation


def horizontal_flip_image_tensor(image: torch.Tensor) -> torch.Tensor:
    return image.flip(-1)


horizontal_flip_image_pil = _FP.hflip


def horizontal_flip_mask(mask: torch.Tensor) -> torch.Tensor:
    return horizontal_flip_image_tensor(mask)


def horizontal_flip_bounding_box(
    bounding_box: torch.Tensor, format: datapoints.BoundingBoxFormat, spatial_size: Tuple[int, int]
) -> torch.Tensor:
    shape = bounding_box.shape

    bounding_box = bounding_box.clone().reshape(-1, 4)

    if format == datapoints.BoundingBoxFormat.XYXY:
        bounding_box[:, [2, 0]] = bounding_box[:, [0, 2]].sub_(spatial_size[1]).neg_()
    elif format == datapoints.BoundingBoxFormat.XYWH:
        bounding_box[:, 0].add_(bounding_box[:, 2]).sub_(spatial_size[1]).neg_()
    else:  # format == datapoints.BoundingBoxFormat.CXCYWH:
        bounding_box[:, 0].sub_(spatial_size[1]).neg_()

    return bounding_box.reshape(shape)


def horizontal_flip_video(video: torch.Tensor) -> torch.Tensor:
    return horizontal_flip_image_tensor(video)


def horizontal_flip(inpt: datapoints._InputTypeJIT) -> datapoints._InputTypeJIT:
    if not torch.jit.is_scripting():
        _log_api_usage_once(horizontal_flip)

    if torch.jit.is_scripting() or is_simple_tensor(inpt):
        return horizontal_flip_image_tensor(inpt)
    elif isinstance(inpt, datapoints._datapoint.Datapoint):
        return inpt.horizontal_flip()
    elif isinstance(inpt, PIL.Image.Image):
        return horizontal_flip_image_pil(inpt)
    else:
        raise TypeError(
            f"Input can either be a plain tensor, any TorchVision datapoint, or a PIL image, "
            f"but got {type(inpt)} instead."
        )


def vertical_flip_image_tensor(image: torch.Tensor) -> torch.Tensor:
    return image.flip(-2)


vertical_flip_image_pil = _FP.vflip


def vertical_flip_mask(mask: torch.Tensor) -> torch.Tensor:
    return vertical_flip_image_tensor(mask)


def vertical_flip_bounding_box(
    bounding_box: torch.Tensor, format: datapoints.BoundingBoxFormat, spatial_size: Tuple[int, int]
) -> torch.Tensor:
    shape = bounding_box.shape

    bounding_box = bounding_box.clone().reshape(-1, 4)

    if format == datapoints.BoundingBoxFormat.XYXY:
        bounding_box[:, [1, 3]] = bounding_box[:, [3, 1]].sub_(spatial_size[0]).neg_()
    elif format == datapoints.BoundingBoxFormat.XYWH:
        bounding_box[:, 1].add_(bounding_box[:, 3]).sub_(spatial_size[0]).neg_()
    else:  # format == datapoints.BoundingBoxFormat.CXCYWH:
        bounding_box[:, 1].sub_(spatial_size[0]).neg_()

    return bounding_box.reshape(shape)


def vertical_flip_video(video: torch.Tensor) -> torch.Tensor:
    return vertical_flip_image_tensor(video)


def vertical_flip(inpt: datapoints._InputTypeJIT) -> datapoints._InputTypeJIT:
    if not torch.jit.is_scripting():
        _log_api_usage_once(vertical_flip)

    if torch.jit.is_scripting() or is_simple_tensor(inpt):
        return vertical_flip_image_tensor(inpt)
    elif isinstance(inpt, datapoints._datapoint.Datapoint):
        return inpt.vertical_flip()
    elif isinstance(inpt, PIL.Image.Image):
        return vertical_flip_image_pil(inpt)
    else:
        raise TypeError(
            f"Input can either be a plain tensor, any TorchVision datapoint, or a PIL image, "
            f"but got {type(inpt)} instead."
        )


# We changed the names to align them with the transforms, i.e. `RandomHorizontalFlip`. Still, `hflip` and `vflip` are
# prevalent and well understood. Thus, we just alias them without deprecating the old names.
hflip = horizontal_flip
vflip = vertical_flip


def _compute_resized_output_size(
    spatial_size: Tuple[int, int], size: List[int], max_size: Optional[int] = None
) -> List[int]:
    if isinstance(size, int):
        size = [size]
    elif max_size is not None and len(size) != 1:
        raise ValueError(
            "max_size should only be passed if size specifies the length of the smaller edge, "
            "i.e. size should be an int or a sequence of length 1 in torchscript mode."
        )
    return __compute_resized_output_size(spatial_size, size=size, max_size=max_size)


def resize_image_tensor(
    image: torch.Tensor,
    size: List[int],
    interpolation: Union[InterpolationMode, int] = InterpolationMode.BILINEAR,
    max_size: Optional[int] = None,
    antialias: Optional[Union[str, bool]] = "warn",
) -> torch.Tensor:
    interpolation = _check_interpolation(interpolation)
    antialias = _check_antialias(img=image, antialias=antialias, interpolation=interpolation)
    assert not isinstance(antialias, str)
    antialias = False if antialias is None else antialias
    align_corners: Optional[bool] = None
    if interpolation == InterpolationMode.BILINEAR or interpolation == InterpolationMode.BICUBIC:
        align_corners = False
    else:
        # The default of antialias should be True from 0.17, so we don't warn or
        # error if other interpolation modes are used. This is documented.
        antialias = False

    shape = image.shape
    num_channels, old_height, old_width = shape[-3:]
    new_height, new_width = _compute_resized_output_size((old_height, old_width), size=size, max_size=max_size)

    if image.numel() > 0:
        image = image.reshape(-1, num_channels, old_height, old_width)

        dtype = image.dtype
        need_cast = dtype not in (torch.float32, torch.float64)
        if need_cast:
            image = image.to(dtype=torch.float32)

        image = interpolate(
            image,
            size=[new_height, new_width],
            mode=interpolation.value,
            align_corners=align_corners,
            antialias=antialias,
        )

        if need_cast:
            if interpolation == InterpolationMode.BICUBIC and dtype == torch.uint8:
                image = image.clamp_(min=0, max=255)
            image = image.round_().to(dtype=dtype)

    return image.reshape(shape[:-3] + (num_channels, new_height, new_width))


@torch.jit.unused
def resize_image_pil(
    image: PIL.Image.Image,
    size: Union[Sequence[int], int],
    interpolation: Union[InterpolationMode, int] = InterpolationMode.BILINEAR,
    max_size: Optional[int] = None,
) -> PIL.Image.Image:
    interpolation = _check_interpolation(interpolation)
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
    w_ratio = new_width / old_width
    h_ratio = new_height / old_height
    ratios = torch.tensor([w_ratio, h_ratio, w_ratio, h_ratio], device=bounding_box.device)
    return (
        bounding_box.mul(ratios).to(bounding_box.dtype),
        (new_height, new_width),
    )


def resize_video(
    video: torch.Tensor,
    size: List[int],
    interpolation: Union[InterpolationMode, int] = InterpolationMode.BILINEAR,
    max_size: Optional[int] = None,
    antialias: Optional[Union[str, bool]] = "warn",
) -> torch.Tensor:
    return resize_image_tensor(video, size=size, interpolation=interpolation, max_size=max_size, antialias=antialias)


def resize(
    inpt: datapoints._InputTypeJIT,
    size: List[int],
    interpolation: Union[InterpolationMode, int] = InterpolationMode.BILINEAR,
    max_size: Optional[int] = None,
    antialias: Optional[Union[str, bool]] = "warn",
) -> datapoints._InputTypeJIT:
    if not torch.jit.is_scripting():
        _log_api_usage_once(resize)
    if torch.jit.is_scripting() or is_simple_tensor(inpt):
        return resize_image_tensor(inpt, size, interpolation=interpolation, max_size=max_size, antialias=antialias)
    elif isinstance(inpt, datapoints._datapoint.Datapoint):
        return inpt.resize(size, interpolation=interpolation, max_size=max_size, antialias=antialias)
    elif isinstance(inpt, PIL.Image.Image):
        if antialias is False:
            warnings.warn("Anti-alias option is always applied for PIL Image input. Argument antialias is ignored.")
        return resize_image_pil(inpt, size, interpolation=interpolation, max_size=max_size)
    else:
        raise TypeError(
            f"Input can either be a plain tensor, any TorchVision datapoint, or a PIL image, "
            f"but got {type(inpt)} instead."
        )


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


def _get_inverse_affine_matrix(
    center: List[float], angle: float, translate: List[float], scale: float, shear: List[float], inverted: bool = True
) -> List[float]:
    # Helper method to compute inverse matrix for affine transformation

    # Pillow requires inverse affine transformation matrix:
    # Affine matrix is : M = T * C * RotateScaleShear * C^-1
    #
    # where T is translation matrix: [1, 0, tx | 0, 1, ty | 0, 0, 1]
    #       C is translation matrix to keep center: [1, 0, cx | 0, 1, cy | 0, 0, 1]
    #       RotateScaleShear is rotation with scale and shear matrix
    #
    #       RotateScaleShear(a, s, (sx, sy)) =
    #       = R(a) * S(s) * SHy(sy) * SHx(sx)
    #       = [ s*cos(a - sy)/cos(sy), s*(-cos(a - sy)*tan(sx)/cos(sy) - sin(a)), 0 ]
    #         [ s*sin(a - sy)/cos(sy), s*(-sin(a - sy)*tan(sx)/cos(sy) + cos(a)), 0 ]
    #         [ 0                    , 0                                      , 1 ]
    # where R is a rotation matrix, S is a scaling matrix, and SHx and SHy are the shears:
    # SHx(s) = [1, -tan(s)] and SHy(s) = [1      , 0]
    #          [0, 1      ]              [-tan(s), 1]
    #
    # Thus, the inverse is M^-1 = C * RotateScaleShear^-1 * C^-1 * T^-1

    rot = math.radians(angle)
    sx = math.radians(shear[0])
    sy = math.radians(shear[1])

    cx, cy = center
    tx, ty = translate

    # Cached results
    cos_sy = math.cos(sy)
    tan_sx = math.tan(sx)
    rot_minus_sy = rot - sy
    cx_plus_tx = cx + tx
    cy_plus_ty = cy + ty

    # Rotate Scale Shear (RSS) without scaling
    a = math.cos(rot_minus_sy) / cos_sy
    b = -(a * tan_sx + math.sin(rot))
    c = math.sin(rot_minus_sy) / cos_sy
    d = math.cos(rot) - c * tan_sx

    if inverted:
        # Inverted rotation matrix with scale and shear
        # det([[a, b], [c, d]]) == 1, since det(rotation) = 1 and det(shear) = 1
        matrix = [d / scale, -b / scale, 0.0, -c / scale, a / scale, 0.0]
        # Apply inverse of translation and of center translation: RSS^-1 * C^-1 * T^-1
        # and then apply center translation: C * RSS^-1 * C^-1 * T^-1
        matrix[2] += cx - matrix[0] * cx_plus_tx - matrix[1] * cy_plus_ty
        matrix[5] += cy - matrix[3] * cx_plus_tx - matrix[4] * cy_plus_ty
    else:
        matrix = [a * scale, b * scale, 0.0, c * scale, d * scale, 0.0]
        # Apply inverse of center translation: RSS * C^-1
        # and then apply translation and center : T * C * RSS * C^-1
        matrix[2] += cx_plus_tx - matrix[0] * cx - matrix[1] * cy
        matrix[5] += cy_plus_ty - matrix[3] * cx - matrix[4] * cy

    return matrix


def _compute_affine_output_size(matrix: List[float], w: int, h: int) -> Tuple[int, int]:
    # Inspired of PIL implementation:
    # https://github.com/python-pillow/Pillow/blob/11de3318867e4398057373ee9f12dcb33db7335c/src/PIL/Image.py#L2054

    # pts are Top-Left, Top-Right, Bottom-Left, Bottom-Right points.
    # Points are shifted due to affine matrix torch convention about
    # the center point. Center is (0, 0) for image center pivot point (w * 0.5, h * 0.5)
    half_w = 0.5 * w
    half_h = 0.5 * h
    pts = torch.tensor(
        [
            [-half_w, -half_h, 1.0],
            [-half_w, half_h, 1.0],
            [half_w, half_h, 1.0],
            [half_w, -half_h, 1.0],
        ]
    )
    theta = torch.tensor(matrix, dtype=torch.float).view(2, 3)
    new_pts = torch.matmul(pts, theta.T)
    min_vals, max_vals = new_pts.aminmax(dim=0)

    # shift points to [0, w] and [0, h] interval to match PIL results
    halfs = torch.tensor((half_w, half_h))
    min_vals.add_(halfs)
    max_vals.add_(halfs)

    # Truncate precision to 1e-4 to avoid ceil of Xe-15 to 1.0
    tol = 1e-4
    inv_tol = 1.0 / tol
    cmax = max_vals.mul_(inv_tol).trunc_().mul_(tol).ceil_()
    cmin = min_vals.mul_(inv_tol).trunc_().mul_(tol).floor_()
    size = cmax.sub_(cmin)
    return int(size[0]), int(size[1])  # w, h


def _apply_grid_transform(
    img: torch.Tensor, grid: torch.Tensor, mode: str, fill: datapoints._FillTypeJIT
) -> torch.Tensor:

    # We are using context knowledge that grid should have float dtype
    fp = img.dtype == grid.dtype
    float_img = img if fp else img.to(grid.dtype)

    shape = float_img.shape
    if shape[0] > 1:
        # Apply same grid to a batch of images
        grid = grid.expand(shape[0], -1, -1, -1)

    # Append a dummy mask for customized fill colors, should be faster than grid_sample() twice
    if fill is not None:
        mask = torch.ones((shape[0], 1, shape[2], shape[3]), dtype=float_img.dtype, device=float_img.device)
        float_img = torch.cat((float_img, mask), dim=1)

    float_img = grid_sample(float_img, grid, mode=mode, padding_mode="zeros", align_corners=False)

    # Fill with required color
    if fill is not None:
        float_img, mask = torch.tensor_split(float_img, indices=(-1,), dim=-3)
        mask = mask.expand_as(float_img)
        fill_list = fill if isinstance(fill, (tuple, list)) else [float(fill)]  # type: ignore[arg-type]
        fill_img = torch.tensor(fill_list, dtype=float_img.dtype, device=float_img.device).view(1, -1, 1, 1)
        if mode == "nearest":
            bool_mask = mask < 0.5
            float_img[bool_mask] = fill_img.expand_as(float_img)[bool_mask]
        else:  # 'bilinear'
            # The following is mathematically equivalent to:
            # img * mask + (1.0 - mask) * fill = img * mask - fill * mask + fill = mask * (img - fill) + fill
            float_img = float_img.sub_(fill_img).mul_(mask).add_(fill_img)

    img = float_img.round_().to(img.dtype) if not fp else float_img

    return img


def _assert_grid_transform_inputs(
    image: torch.Tensor,
    matrix: Optional[List[float]],
    interpolation: str,
    fill: datapoints._FillTypeJIT,
    supported_interpolation_modes: List[str],
    coeffs: Optional[List[float]] = None,
) -> None:
    if matrix is not None:
        if not isinstance(matrix, list):
            raise TypeError("Argument matrix should be a list")
        elif len(matrix) != 6:
            raise ValueError("Argument matrix should have 6 float values")

    if coeffs is not None and len(coeffs) != 8:
        raise ValueError("Argument coeffs should have 8 float values")

    if fill is not None:
        if isinstance(fill, (tuple, list)):
            length = len(fill)
            num_channels = image.shape[-3]
            if length > 1 and length != num_channels:
                raise ValueError(
                    "The number of elements in 'fill' cannot broadcast to match the number of "
                    f"channels of the image ({length} != {num_channels})"
                )
        elif not isinstance(fill, (int, float)):
            raise ValueError("Argument fill should be either int, float, tuple or list")

    if interpolation not in supported_interpolation_modes:
        raise ValueError(f"Interpolation mode '{interpolation}' is unsupported with Tensor input")


def _affine_grid(
    theta: torch.Tensor,
    w: int,
    h: int,
    ow: int,
    oh: int,
) -> torch.Tensor:
    # https://github.com/pytorch/pytorch/blob/74b65c32be68b15dc7c9e8bb62459efbfbde33d8/aten/src/ATen/native/
    # AffineGridGenerator.cpp#L18
    # Difference with AffineGridGenerator is that:
    # 1) we normalize grid values after applying theta
    # 2) we can normalize by other image size, such that it covers "extend" option like in PIL.Image.rotate
    dtype = theta.dtype
    device = theta.device

    base_grid = torch.empty(1, oh, ow, 3, dtype=dtype, device=device)
    x_grid = torch.linspace((1.0 - ow) * 0.5, (ow - 1.0) * 0.5, steps=ow, device=device)
    base_grid[..., 0].copy_(x_grid)
    y_grid = torch.linspace((1.0 - oh) * 0.5, (oh - 1.0) * 0.5, steps=oh, device=device).unsqueeze_(-1)
    base_grid[..., 1].copy_(y_grid)
    base_grid[..., 2].fill_(1)

    rescaled_theta = theta.transpose(1, 2).div_(torch.tensor([0.5 * w, 0.5 * h], dtype=dtype, device=device))
    output_grid = base_grid.view(1, oh * ow, 3).bmm(rescaled_theta)
    return output_grid.view(1, oh, ow, 2)


def affine_image_tensor(
    image: torch.Tensor,
    angle: Union[int, float],
    translate: List[float],
    scale: float,
    shear: List[float],
    interpolation: Union[InterpolationMode, int] = InterpolationMode.NEAREST,
    fill: datapoints._FillTypeJIT = None,
    center: Optional[List[float]] = None,
) -> torch.Tensor:
    interpolation = _check_interpolation(interpolation)

    if image.numel() == 0:
        return image

    shape = image.shape
    ndim = image.ndim

    if ndim > 4:
        image = image.reshape((-1,) + shape[-3:])
        needs_unsquash = True
    elif ndim == 3:
        image = image.unsqueeze(0)
        needs_unsquash = True
    else:
        needs_unsquash = False

    height, width = shape[-2:]
    angle, translate, shear, center = _affine_parse_args(angle, translate, scale, shear, interpolation, center)

    center_f = [0.0, 0.0]
    if center is not None:
        # Center values should be in pixel coordinates but translated such that (0, 0) corresponds to image center.
        center_f = [(c - s * 0.5) for c, s in zip(center, [width, height])]

    translate_f = [float(t) for t in translate]
    matrix = _get_inverse_affine_matrix(center_f, angle, translate_f, scale, shear)

    _assert_grid_transform_inputs(image, matrix, interpolation.value, fill, ["nearest", "bilinear"])

    dtype = image.dtype if torch.is_floating_point(image) else torch.float32
    theta = torch.tensor(matrix, dtype=dtype, device=image.device).reshape(1, 2, 3)
    grid = _affine_grid(theta, w=width, h=height, ow=width, oh=height)
    output = _apply_grid_transform(image, grid, interpolation.value, fill=fill)

    if needs_unsquash:
        output = output.reshape(shape)

    return output


@torch.jit.unused
def affine_image_pil(
    image: PIL.Image.Image,
    angle: Union[int, float],
    translate: List[float],
    scale: float,
    shear: List[float],
    interpolation: Union[InterpolationMode, int] = InterpolationMode.NEAREST,
    fill: datapoints._FillTypeJIT = None,
    center: Optional[List[float]] = None,
) -> PIL.Image.Image:
    interpolation = _check_interpolation(interpolation)
    angle, translate, shear, center = _affine_parse_args(angle, translate, scale, shear, interpolation, center)

    # center = (img_size[0] * 0.5 + 0.5, img_size[1] * 0.5 + 0.5)
    # it is visually better to estimate the center without 0.5 offset
    # otherwise image rotated by 90 degrees is shifted vs output image of torch.rot90 or F_t.affine
    if center is None:
        height, width = get_spatial_size_image_pil(image)
        center = [width * 0.5, height * 0.5]
    matrix = _get_inverse_affine_matrix(center, angle, translate, scale, shear)

    return _FP.affine(image, matrix, interpolation=pil_modes_mapping[interpolation], fill=fill)


def _affine_bounding_box_with_expand(
    bounding_box: torch.Tensor,
    format: datapoints.BoundingBoxFormat,
    spatial_size: Tuple[int, int],
    angle: Union[int, float],
    translate: List[float],
    scale: float,
    shear: List[float],
    center: Optional[List[float]] = None,
    expand: bool = False,
) -> Tuple[torch.Tensor, Tuple[int, int]]:
    if bounding_box.numel() == 0:
        return bounding_box, spatial_size

    original_shape = bounding_box.shape
    original_dtype = bounding_box.dtype
    bounding_box = bounding_box.clone() if bounding_box.is_floating_point() else bounding_box.float()
    dtype = bounding_box.dtype
    device = bounding_box.device
    bounding_box = (
        convert_format_bounding_box(
            bounding_box, old_format=format, new_format=datapoints.BoundingBoxFormat.XYXY, inplace=True
        )
    ).reshape(-1, 4)

    angle, translate, shear, center = _affine_parse_args(
        angle, translate, scale, shear, InterpolationMode.NEAREST, center
    )

    if center is None:
        height, width = spatial_size
        center = [width * 0.5, height * 0.5]

    affine_vector = _get_inverse_affine_matrix(center, angle, translate, scale, shear, inverted=False)
    transposed_affine_matrix = (
        torch.tensor(
            affine_vector,
            dtype=dtype,
            device=device,
        )
        .reshape(2, 3)
        .T
    )
    # 1) Let's transform bboxes into a tensor of 4 points (top-left, top-right, bottom-left, bottom-right corners).
    # Tensor of points has shape (N * 4, 3), where N is the number of bboxes
    # Single point structure is similar to
    # [(xmin, ymin, 1), (xmax, ymin, 1), (xmax, ymax, 1), (xmin, ymax, 1)]
    points = bounding_box[:, [[0, 1], [2, 1], [2, 3], [0, 3]]].reshape(-1, 2)
    points = torch.cat([points, torch.ones(points.shape[0], 1, device=device, dtype=dtype)], dim=-1)
    # 2) Now let's transform the points using affine matrix
    transformed_points = torch.matmul(points, transposed_affine_matrix)
    # 3) Reshape transformed points to [N boxes, 4 points, x/y coords]
    # and compute bounding box from 4 transformed points:
    transformed_points = transformed_points.reshape(-1, 4, 2)
    out_bbox_mins, out_bbox_maxs = torch.aminmax(transformed_points, dim=1)
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
        tr = torch.amin(new_points, dim=0, keepdim=True)
        # Translate bounding boxes
        out_bboxes.sub_(tr.repeat((1, 2)))
        # Estimate meta-data for image with inverted=True and with center=[0,0]
        affine_vector = _get_inverse_affine_matrix([0.0, 0.0], angle, translate, scale, shear)
        new_width, new_height = _compute_affine_output_size(affine_vector, width, height)
        spatial_size = (new_height, new_width)

    out_bboxes = clamp_bounding_box(out_bboxes, format=datapoints.BoundingBoxFormat.XYXY, spatial_size=spatial_size)
    out_bboxes = convert_format_bounding_box(
        out_bboxes, old_format=datapoints.BoundingBoxFormat.XYXY, new_format=format, inplace=True
    ).reshape(original_shape)

    out_bboxes = out_bboxes.to(original_dtype)
    return out_bboxes, spatial_size


def affine_bounding_box(
    bounding_box: torch.Tensor,
    format: datapoints.BoundingBoxFormat,
    spatial_size: Tuple[int, int],
    angle: Union[int, float],
    translate: List[float],
    scale: float,
    shear: List[float],
    center: Optional[List[float]] = None,
) -> torch.Tensor:
    out_box, _ = _affine_bounding_box_with_expand(
        bounding_box,
        format=format,
        spatial_size=spatial_size,
        angle=angle,
        translate=translate,
        scale=scale,
        shear=shear,
        center=center,
        expand=False,
    )
    return out_box


def affine_mask(
    mask: torch.Tensor,
    angle: Union[int, float],
    translate: List[float],
    scale: float,
    shear: List[float],
    fill: datapoints._FillTypeJIT = None,
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
    interpolation: Union[InterpolationMode, int] = InterpolationMode.NEAREST,
    fill: datapoints._FillTypeJIT = None,
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


def affine(
    inpt: datapoints._InputTypeJIT,
    angle: Union[int, float],
    translate: List[float],
    scale: float,
    shear: List[float],
    interpolation: Union[InterpolationMode, int] = InterpolationMode.NEAREST,
    fill: datapoints._FillTypeJIT = None,
    center: Optional[List[float]] = None,
) -> datapoints._InputTypeJIT:
    if not torch.jit.is_scripting():
        _log_api_usage_once(affine)

    # TODO: consider deprecating integers from angle and shear on the future
    if torch.jit.is_scripting() or is_simple_tensor(inpt):
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
    elif isinstance(inpt, datapoints._datapoint.Datapoint):
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
        raise TypeError(
            f"Input can either be a plain tensor, any TorchVision datapoint, or a PIL image, "
            f"but got {type(inpt)} instead."
        )


def rotate_image_tensor(
    image: torch.Tensor,
    angle: float,
    interpolation: Union[InterpolationMode, int] = InterpolationMode.NEAREST,
    expand: bool = False,
    center: Optional[List[float]] = None,
    fill: datapoints._FillTypeJIT = None,
) -> torch.Tensor:
    interpolation = _check_interpolation(interpolation)

    shape = image.shape
    num_channels, height, width = shape[-3:]

    center_f = [0.0, 0.0]
    if center is not None:
        if expand:
            # TODO: Do we actually want to warn, or just document this?
            warnings.warn("The provided center argument has no effect on the result if expand is True")
        # Center values should be in pixel coordinates but translated such that (0, 0) corresponds to image center.
        center_f = [(c - s * 0.5) for c, s in zip(center, [width, height])]

    # due to current incoherence of rotation angle direction between affine and rotate implementations
    # we need to set -angle.
    matrix = _get_inverse_affine_matrix(center_f, -angle, [0.0, 0.0], 1.0, [0.0, 0.0])

    if image.numel() > 0:
        image = image.reshape(-1, num_channels, height, width)

        _assert_grid_transform_inputs(image, matrix, interpolation.value, fill, ["nearest", "bilinear"])

        ow, oh = _compute_affine_output_size(matrix, width, height) if expand else (width, height)
        dtype = image.dtype if torch.is_floating_point(image) else torch.float32
        theta = torch.tensor(matrix, dtype=dtype, device=image.device).reshape(1, 2, 3)
        grid = _affine_grid(theta, w=width, h=height, ow=ow, oh=oh)
        output = _apply_grid_transform(image, grid, interpolation.value, fill=fill)

        new_height, new_width = output.shape[-2:]
    else:
        output = image
        new_width, new_height = _compute_affine_output_size(matrix, width, height) if expand else (width, height)

    return output.reshape(shape[:-3] + (num_channels, new_height, new_width))


@torch.jit.unused
def rotate_image_pil(
    image: PIL.Image.Image,
    angle: float,
    interpolation: Union[InterpolationMode, int] = InterpolationMode.NEAREST,
    expand: bool = False,
    center: Optional[List[float]] = None,
    fill: datapoints._FillTypeJIT = None,
) -> PIL.Image.Image:
    interpolation = _check_interpolation(interpolation)

    if center is not None and expand:
        warnings.warn("The provided center argument has no effect on the result if expand is True")
        center = None

    return _FP.rotate(
        image, angle, interpolation=pil_modes_mapping[interpolation], expand=expand, fill=fill, center=center
    )


def rotate_bounding_box(
    bounding_box: torch.Tensor,
    format: datapoints.BoundingBoxFormat,
    spatial_size: Tuple[int, int],
    angle: float,
    expand: bool = False,
    center: Optional[List[float]] = None,
) -> Tuple[torch.Tensor, Tuple[int, int]]:
    if center is not None and expand:
        warnings.warn("The provided center argument has no effect on the result if expand is True")
        center = None

    return _affine_bounding_box_with_expand(
        bounding_box,
        format=format,
        spatial_size=spatial_size,
        angle=-angle,
        translate=[0.0, 0.0],
        scale=1.0,
        shear=[0.0, 0.0],
        center=center,
        expand=expand,
    )


def rotate_mask(
    mask: torch.Tensor,
    angle: float,
    expand: bool = False,
    center: Optional[List[float]] = None,
    fill: datapoints._FillTypeJIT = None,
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
    interpolation: Union[InterpolationMode, int] = InterpolationMode.NEAREST,
    expand: bool = False,
    center: Optional[List[float]] = None,
    fill: datapoints._FillTypeJIT = None,
) -> torch.Tensor:
    return rotate_image_tensor(video, angle, interpolation=interpolation, expand=expand, fill=fill, center=center)


def rotate(
    inpt: datapoints._InputTypeJIT,
    angle: float,
    interpolation: Union[InterpolationMode, int] = InterpolationMode.NEAREST,
    expand: bool = False,
    center: Optional[List[float]] = None,
    fill: datapoints._FillTypeJIT = None,
) -> datapoints._InputTypeJIT:
    if not torch.jit.is_scripting():
        _log_api_usage_once(rotate)

    if torch.jit.is_scripting() or is_simple_tensor(inpt):
        return rotate_image_tensor(inpt, angle, interpolation=interpolation, expand=expand, fill=fill, center=center)
    elif isinstance(inpt, datapoints._datapoint.Datapoint):
        return inpt.rotate(angle, interpolation=interpolation, expand=expand, fill=fill, center=center)
    elif isinstance(inpt, PIL.Image.Image):
        return rotate_image_pil(inpt, angle, interpolation=interpolation, expand=expand, fill=fill, center=center)
    else:
        raise TypeError(
            f"Input can either be a plain tensor, any TorchVision datapoint, or a PIL image, "
            f"but got {type(inpt)} instead."
        )


def _parse_pad_padding(padding: Union[int, List[int]]) -> List[int]:
    if isinstance(padding, int):
        pad_left = pad_right = pad_top = pad_bottom = padding
    elif isinstance(padding, (tuple, list)):
        if len(padding) == 1:
            pad_left = pad_right = pad_top = pad_bottom = padding[0]
        elif len(padding) == 2:
            pad_left = pad_right = padding[0]
            pad_top = pad_bottom = padding[1]
        elif len(padding) == 4:
            pad_left = padding[0]
            pad_top = padding[1]
            pad_right = padding[2]
            pad_bottom = padding[3]
        else:
            raise ValueError(
                f"Padding must be an int or a 1, 2, or 4 element tuple, not a {len(padding)} element tuple"
            )
    else:
        raise TypeError(f"`padding` should be an integer or tuple or list of integers, but got {padding}")

    return [pad_left, pad_right, pad_top, pad_bottom]


def pad_image_tensor(
    image: torch.Tensor,
    padding: List[int],
    fill: Optional[Union[int, float, List[float]]] = None,
    padding_mode: str = "constant",
) -> torch.Tensor:
    # Be aware that while `padding` has order `[left, top, right, bottom]` has order, `torch_padding` uses
    # `[left, right, top, bottom]`. This stems from the fact that we align our API with PIL, but need to use `torch_pad`
    # internally.
    torch_padding = _parse_pad_padding(padding)

    if padding_mode not in ("constant", "edge", "reflect", "symmetric"):
        raise ValueError(
            f"`padding_mode` should be either `'constant'`, `'edge'`, `'reflect'` or `'symmetric'`, "
            f"but got `'{padding_mode}'`."
        )

    if fill is None:
        fill = 0

    if isinstance(fill, (int, float)):
        return _pad_with_scalar_fill(image, torch_padding, fill=fill, padding_mode=padding_mode)
    elif len(fill) == 1:
        return _pad_with_scalar_fill(image, torch_padding, fill=fill[0], padding_mode=padding_mode)
    else:
        return _pad_with_vector_fill(image, torch_padding, fill=fill, padding_mode=padding_mode)


def _pad_with_scalar_fill(
    image: torch.Tensor,
    torch_padding: List[int],
    fill: Union[int, float],
    padding_mode: str,
) -> torch.Tensor:
    shape = image.shape
    num_channels, height, width = shape[-3:]

    batch_size = 1
    for s in shape[:-3]:
        batch_size *= s

    image = image.reshape(batch_size, num_channels, height, width)

    if padding_mode == "edge":
        # Similar to the padding order, `torch_pad`'s PIL's padding modes don't have the same names. Thus, we map
        # the PIL name for the padding mode, which we are also using for our API, to the corresponding `torch_pad`
        # name.
        padding_mode = "replicate"

    if padding_mode == "constant":
        image = torch_pad(image, torch_padding, mode=padding_mode, value=float(fill))
    elif padding_mode in ("reflect", "replicate"):
        # `torch_pad` only supports `"reflect"` or `"replicate"` padding for floating point inputs.
        # TODO: See https://github.com/pytorch/pytorch/issues/40763
        dtype = image.dtype
        if not image.is_floating_point():
            needs_cast = True
            image = image.to(torch.float32)
        else:
            needs_cast = False

        image = torch_pad(image, torch_padding, mode=padding_mode)

        if needs_cast:
            image = image.to(dtype)
    else:  # padding_mode == "symmetric"
        image = _pad_symmetric(image, torch_padding)

    new_height, new_width = image.shape[-2:]

    return image.reshape(shape[:-3] + (num_channels, new_height, new_width))


# TODO: This should be removed once torch_pad supports non-scalar padding values
def _pad_with_vector_fill(
    image: torch.Tensor,
    torch_padding: List[int],
    fill: List[float],
    padding_mode: str,
) -> torch.Tensor:
    if padding_mode != "constant":
        raise ValueError(f"Padding mode '{padding_mode}' is not supported if fill is not scalar")

    output = _pad_with_scalar_fill(image, torch_padding, fill=0, padding_mode="constant")
    left, right, top, bottom = torch_padding
    fill = torch.tensor(fill, dtype=image.dtype, device=image.device).reshape(-1, 1, 1)

    if top > 0:
        output[..., :top, :] = fill
    if left > 0:
        output[..., :, :left] = fill
    if bottom > 0:
        output[..., -bottom:, :] = fill
    if right > 0:
        output[..., :, -right:] = fill
    return output


pad_image_pil = _FP.pad


def pad_mask(
    mask: torch.Tensor,
    padding: List[int],
    fill: Optional[Union[int, float, List[float]]] = None,
    padding_mode: str = "constant",
) -> torch.Tensor:
    if fill is None:
        fill = 0

    if isinstance(fill, (tuple, list)):
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
    format: datapoints.BoundingBoxFormat,
    spatial_size: Tuple[int, int],
    padding: List[int],
    padding_mode: str = "constant",
) -> Tuple[torch.Tensor, Tuple[int, int]]:
    if padding_mode not in ["constant"]:
        # TODO: add support of other padding modes
        raise ValueError(f"Padding mode '{padding_mode}' is not supported with bounding boxes")

    left, right, top, bottom = _parse_pad_padding(padding)

    if format == datapoints.BoundingBoxFormat.XYXY:
        pad = [left, top, left, top]
    else:
        pad = [left, top, 0, 0]
    bounding_box = bounding_box + torch.tensor(pad, dtype=bounding_box.dtype, device=bounding_box.device)

    height, width = spatial_size
    height += top + bottom
    width += left + right
    spatial_size = (height, width)

    return clamp_bounding_box(bounding_box, format=format, spatial_size=spatial_size), spatial_size


def pad_video(
    video: torch.Tensor,
    padding: List[int],
    fill: Optional[Union[int, float, List[float]]] = None,
    padding_mode: str = "constant",
) -> torch.Tensor:
    return pad_image_tensor(video, padding, fill=fill, padding_mode=padding_mode)


def pad(
    inpt: datapoints._InputTypeJIT,
    padding: List[int],
    fill: Optional[Union[int, float, List[float]]] = None,
    padding_mode: str = "constant",
) -> datapoints._InputTypeJIT:
    if not torch.jit.is_scripting():
        _log_api_usage_once(pad)

    if torch.jit.is_scripting() or is_simple_tensor(inpt):
        return pad_image_tensor(inpt, padding, fill=fill, padding_mode=padding_mode)

    elif isinstance(inpt, datapoints._datapoint.Datapoint):
        return inpt.pad(padding, fill=fill, padding_mode=padding_mode)
    elif isinstance(inpt, PIL.Image.Image):
        return pad_image_pil(inpt, padding, fill=fill, padding_mode=padding_mode)
    else:
        raise TypeError(
            f"Input can either be a plain tensor, any TorchVision datapoint, or a PIL image, "
            f"but got {type(inpt)} instead."
        )


def crop_image_tensor(image: torch.Tensor, top: int, left: int, height: int, width: int) -> torch.Tensor:
    h, w = image.shape[-2:]

    right = left + width
    bottom = top + height

    if left < 0 or top < 0 or right > w or bottom > h:
        image = image[..., max(top, 0) : bottom, max(left, 0) : right]
        torch_padding = [
            max(min(right, 0) - left, 0),
            max(right - max(w, left), 0),
            max(min(bottom, 0) - top, 0),
            max(bottom - max(h, top), 0),
        ]
        return _pad_with_scalar_fill(image, torch_padding, fill=0, padding_mode="constant")
    return image[..., top:bottom, left:right]


crop_image_pil = _FP.crop


def crop_bounding_box(
    bounding_box: torch.Tensor,
    format: datapoints.BoundingBoxFormat,
    top: int,
    left: int,
    height: int,
    width: int,
) -> Tuple[torch.Tensor, Tuple[int, int]]:

    # Crop or implicit pad if left and/or top have negative values:
    if format == datapoints.BoundingBoxFormat.XYXY:
        sub = [left, top, left, top]
    else:
        sub = [left, top, 0, 0]

    bounding_box = bounding_box - torch.tensor(sub, dtype=bounding_box.dtype, device=bounding_box.device)
    spatial_size = (height, width)

    return clamp_bounding_box(bounding_box, format=format, spatial_size=spatial_size), spatial_size


def crop_mask(mask: torch.Tensor, top: int, left: int, height: int, width: int) -> torch.Tensor:
    if mask.ndim < 3:
        mask = mask.unsqueeze(0)
        needs_squeeze = True
    else:
        needs_squeeze = False

    output = crop_image_tensor(mask, top, left, height, width)

    if needs_squeeze:
        output = output.squeeze(0)

    return output


def crop_video(video: torch.Tensor, top: int, left: int, height: int, width: int) -> torch.Tensor:
    return crop_image_tensor(video, top, left, height, width)


def crop(inpt: datapoints._InputTypeJIT, top: int, left: int, height: int, width: int) -> datapoints._InputTypeJIT:
    if not torch.jit.is_scripting():
        _log_api_usage_once(crop)

    if torch.jit.is_scripting() or is_simple_tensor(inpt):
        return crop_image_tensor(inpt, top, left, height, width)
    elif isinstance(inpt, datapoints._datapoint.Datapoint):
        return inpt.crop(top, left, height, width)
    elif isinstance(inpt, PIL.Image.Image):
        return crop_image_pil(inpt, top, left, height, width)
    else:
        raise TypeError(
            f"Input can either be a plain tensor, any TorchVision datapoint, or a PIL image, "
            f"but got {type(inpt)} instead."
        )


def _perspective_grid(coeffs: List[float], ow: int, oh: int, dtype: torch.dtype, device: torch.device) -> torch.Tensor:
    # https://github.com/python-pillow/Pillow/blob/4634eafe3c695a014267eefdce830b4a825beed7/
    # src/libImaging/Geometry.c#L394

    #
    # x_out = (coeffs[0] * x + coeffs[1] * y + coeffs[2]) / (coeffs[6] * x + coeffs[7] * y + 1)
    # y_out = (coeffs[3] * x + coeffs[4] * y + coeffs[5]) / (coeffs[6] * x + coeffs[7] * y + 1)
    #
    theta1 = torch.tensor(
        [[[coeffs[0], coeffs[1], coeffs[2]], [coeffs[3], coeffs[4], coeffs[5]]]], dtype=dtype, device=device
    )
    theta2 = torch.tensor([[[coeffs[6], coeffs[7], 1.0], [coeffs[6], coeffs[7], 1.0]]], dtype=dtype, device=device)

    d = 0.5
    base_grid = torch.empty(1, oh, ow, 3, dtype=dtype, device=device)
    x_grid = torch.linspace(d, ow + d - 1.0, steps=ow, device=device, dtype=dtype)
    base_grid[..., 0].copy_(x_grid)
    y_grid = torch.linspace(d, oh + d - 1.0, steps=oh, device=device, dtype=dtype).unsqueeze_(-1)
    base_grid[..., 1].copy_(y_grid)
    base_grid[..., 2].fill_(1)

    rescaled_theta1 = theta1.transpose(1, 2).div_(torch.tensor([0.5 * ow, 0.5 * oh], dtype=dtype, device=device))
    shape = (1, oh * ow, 3)
    output_grid1 = base_grid.view(shape).bmm(rescaled_theta1)
    output_grid2 = base_grid.view(shape).bmm(theta2.transpose(1, 2))

    output_grid = output_grid1.div_(output_grid2).sub_(1.0)
    return output_grid.view(1, oh, ow, 2)


def _perspective_coefficients(
    startpoints: Optional[List[List[int]]],
    endpoints: Optional[List[List[int]]],
    coefficients: Optional[List[float]],
) -> List[float]:
    if coefficients is not None:
        if startpoints is not None and endpoints is not None:
            raise ValueError("The startpoints/endpoints and the coefficients shouldn't be defined concurrently.")
        elif len(coefficients) != 8:
            raise ValueError("Argument coefficients should have 8 float values")
        return coefficients
    elif startpoints is not None and endpoints is not None:
        return _get_perspective_coeffs(startpoints, endpoints)
    else:
        raise ValueError("Either the startpoints/endpoints or the coefficients must have non `None` values.")


def perspective_image_tensor(
    image: torch.Tensor,
    startpoints: Optional[List[List[int]]],
    endpoints: Optional[List[List[int]]],
    interpolation: Union[InterpolationMode, int] = InterpolationMode.BILINEAR,
    fill: datapoints._FillTypeJIT = None,
    coefficients: Optional[List[float]] = None,
) -> torch.Tensor:
    perspective_coeffs = _perspective_coefficients(startpoints, endpoints, coefficients)
    interpolation = _check_interpolation(interpolation)

    if image.numel() == 0:
        return image

    shape = image.shape
    ndim = image.ndim

    if ndim > 4:
        image = image.reshape((-1,) + shape[-3:])
        needs_unsquash = True
    elif ndim == 3:
        image = image.unsqueeze(0)
        needs_unsquash = True
    else:
        needs_unsquash = False

    _assert_grid_transform_inputs(
        image,
        matrix=None,
        interpolation=interpolation.value,
        fill=fill,
        supported_interpolation_modes=["nearest", "bilinear"],
        coeffs=perspective_coeffs,
    )

    oh, ow = shape[-2:]
    dtype = image.dtype if torch.is_floating_point(image) else torch.float32
    grid = _perspective_grid(perspective_coeffs, ow=ow, oh=oh, dtype=dtype, device=image.device)
    output = _apply_grid_transform(image, grid, interpolation.value, fill=fill)

    if needs_unsquash:
        output = output.reshape(shape)

    return output


@torch.jit.unused
def perspective_image_pil(
    image: PIL.Image.Image,
    startpoints: Optional[List[List[int]]],
    endpoints: Optional[List[List[int]]],
    interpolation: Union[InterpolationMode, int] = InterpolationMode.BICUBIC,
    fill: datapoints._FillTypeJIT = None,
    coefficients: Optional[List[float]] = None,
) -> PIL.Image.Image:
    perspective_coeffs = _perspective_coefficients(startpoints, endpoints, coefficients)
    interpolation = _check_interpolation(interpolation)
    return _FP.perspective(image, perspective_coeffs, interpolation=pil_modes_mapping[interpolation], fill=fill)


def perspective_bounding_box(
    bounding_box: torch.Tensor,
    format: datapoints.BoundingBoxFormat,
    spatial_size: Tuple[int, int],
    startpoints: Optional[List[List[int]]],
    endpoints: Optional[List[List[int]]],
    coefficients: Optional[List[float]] = None,
) -> torch.Tensor:
    if bounding_box.numel() == 0:
        return bounding_box

    perspective_coeffs = _perspective_coefficients(startpoints, endpoints, coefficients)

    original_shape = bounding_box.shape
    # TODO: first cast to float if bbox is int64 before convert_format_bounding_box
    bounding_box = (
        convert_format_bounding_box(bounding_box, old_format=format, new_format=datapoints.BoundingBoxFormat.XYXY)
    ).reshape(-1, 4)

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
    points = bounding_box[:, [[0, 1], [2, 1], [2, 3], [0, 3]]].reshape(-1, 2)
    points = torch.cat([points, torch.ones(points.shape[0], 1, device=points.device)], dim=-1)
    # 2) Now let's transform the points using perspective matrices
    #   x_out = (coeffs[0] * x + coeffs[1] * y + coeffs[2]) / (coeffs[6] * x + coeffs[7] * y + 1)
    #   y_out = (coeffs[3] * x + coeffs[4] * y + coeffs[5]) / (coeffs[6] * x + coeffs[7] * y + 1)

    numer_points = torch.matmul(points, theta1.T)
    denom_points = torch.matmul(points, theta2.T)
    transformed_points = numer_points.div_(denom_points)

    # 3) Reshape transformed points to [N boxes, 4 points, x/y coords]
    # and compute bounding box from 4 transformed points:
    transformed_points = transformed_points.reshape(-1, 4, 2)
    out_bbox_mins, out_bbox_maxs = torch.aminmax(transformed_points, dim=1)

    out_bboxes = clamp_bounding_box(
        torch.cat([out_bbox_mins, out_bbox_maxs], dim=1).to(bounding_box.dtype),
        format=datapoints.BoundingBoxFormat.XYXY,
        spatial_size=spatial_size,
    )

    # out_bboxes should be of shape [N boxes, 4]

    return convert_format_bounding_box(
        out_bboxes, old_format=datapoints.BoundingBoxFormat.XYXY, new_format=format, inplace=True
    ).reshape(original_shape)


def perspective_mask(
    mask: torch.Tensor,
    startpoints: Optional[List[List[int]]],
    endpoints: Optional[List[List[int]]],
    fill: datapoints._FillTypeJIT = None,
    coefficients: Optional[List[float]] = None,
) -> torch.Tensor:
    if mask.ndim < 3:
        mask = mask.unsqueeze(0)
        needs_squeeze = True
    else:
        needs_squeeze = False

    output = perspective_image_tensor(
        mask, startpoints, endpoints, interpolation=InterpolationMode.NEAREST, fill=fill, coefficients=coefficients
    )

    if needs_squeeze:
        output = output.squeeze(0)

    return output


def perspective_video(
    video: torch.Tensor,
    startpoints: Optional[List[List[int]]],
    endpoints: Optional[List[List[int]]],
    interpolation: Union[InterpolationMode, int] = InterpolationMode.BILINEAR,
    fill: datapoints._FillTypeJIT = None,
    coefficients: Optional[List[float]] = None,
) -> torch.Tensor:
    return perspective_image_tensor(
        video, startpoints, endpoints, interpolation=interpolation, fill=fill, coefficients=coefficients
    )


def perspective(
    inpt: datapoints._InputTypeJIT,
    startpoints: Optional[List[List[int]]],
    endpoints: Optional[List[List[int]]],
    interpolation: Union[InterpolationMode, int] = InterpolationMode.BILINEAR,
    fill: datapoints._FillTypeJIT = None,
    coefficients: Optional[List[float]] = None,
) -> datapoints._InputTypeJIT:
    if not torch.jit.is_scripting():
        _log_api_usage_once(perspective)
    if torch.jit.is_scripting() or is_simple_tensor(inpt):
        return perspective_image_tensor(
            inpt, startpoints, endpoints, interpolation=interpolation, fill=fill, coefficients=coefficients
        )
    elif isinstance(inpt, datapoints._datapoint.Datapoint):
        return inpt.perspective(
            startpoints, endpoints, interpolation=interpolation, fill=fill, coefficients=coefficients
        )
    elif isinstance(inpt, PIL.Image.Image):
        return perspective_image_pil(
            inpt, startpoints, endpoints, interpolation=interpolation, fill=fill, coefficients=coefficients
        )
    else:
        raise TypeError(
            f"Input can either be a plain tensor, any TorchVision datapoint, or a PIL image, "
            f"but got {type(inpt)} instead."
        )


def elastic_image_tensor(
    image: torch.Tensor,
    displacement: torch.Tensor,
    interpolation: Union[InterpolationMode, int] = InterpolationMode.BILINEAR,
    fill: datapoints._FillTypeJIT = None,
) -> torch.Tensor:
    interpolation = _check_interpolation(interpolation)

    if image.numel() == 0:
        return image

    shape = image.shape
    ndim = image.ndim

    device = image.device
    dtype = image.dtype if torch.is_floating_point(image) else torch.float32

    # Patch: elastic transform should support (cpu,f16) input
    is_cpu_half = device.type == "cpu" and dtype == torch.float16
    if is_cpu_half:
        image = image.to(torch.float32)
        dtype = torch.float32

    # We are aware that if input image dtype is uint8 and displacement is float64 then
    # displacement will be casted to float32 and all computations will be done with float32
    # We can fix this later if needed

    expected_shape = (1,) + shape[-2:] + (2,)
    if expected_shape != displacement.shape:
        raise ValueError(f"Argument displacement shape should be {expected_shape}, but given {displacement.shape}")

    if ndim > 4:
        image = image.reshape((-1,) + shape[-3:])
        needs_unsquash = True
    elif ndim == 3:
        image = image.unsqueeze(0)
        needs_unsquash = True
    else:
        needs_unsquash = False

    if displacement.dtype != dtype or displacement.device != device:
        displacement = displacement.to(dtype=dtype, device=device)

    image_height, image_width = shape[-2:]
    grid = _create_identity_grid((image_height, image_width), device=device, dtype=dtype).add_(displacement)
    output = _apply_grid_transform(image, grid, interpolation.value, fill=fill)

    if needs_unsquash:
        output = output.reshape(shape)

    if is_cpu_half:
        output = output.to(torch.float16)

    return output


@torch.jit.unused
def elastic_image_pil(
    image: PIL.Image.Image,
    displacement: torch.Tensor,
    interpolation: Union[InterpolationMode, int] = InterpolationMode.BILINEAR,
    fill: datapoints._FillTypeJIT = None,
) -> PIL.Image.Image:
    t_img = pil_to_tensor(image)
    output = elastic_image_tensor(t_img, displacement, interpolation=interpolation, fill=fill)
    return to_pil_image(output, mode=image.mode)


def _create_identity_grid(size: Tuple[int, int], device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    sy, sx = size
    base_grid = torch.empty(1, sy, sx, 2, device=device, dtype=dtype)
    x_grid = torch.linspace((-sx + 1) / sx, (sx - 1) / sx, sx, device=device, dtype=dtype)
    base_grid[..., 0].copy_(x_grid)

    y_grid = torch.linspace((-sy + 1) / sy, (sy - 1) / sy, sy, device=device, dtype=dtype).unsqueeze_(-1)
    base_grid[..., 1].copy_(y_grid)

    return base_grid


def elastic_bounding_box(
    bounding_box: torch.Tensor,
    format: datapoints.BoundingBoxFormat,
    spatial_size: Tuple[int, int],
    displacement: torch.Tensor,
) -> torch.Tensor:
    if bounding_box.numel() == 0:
        return bounding_box

    # TODO: add in docstring about approximation we are doing for grid inversion
    device = bounding_box.device
    dtype = bounding_box.dtype if torch.is_floating_point(bounding_box) else torch.float32

    if displacement.dtype != dtype or displacement.device != device:
        displacement = displacement.to(dtype=dtype, device=device)

    original_shape = bounding_box.shape
    # TODO: first cast to float if bbox is int64 before convert_format_bounding_box
    bounding_box = (
        convert_format_bounding_box(bounding_box, old_format=format, new_format=datapoints.BoundingBoxFormat.XYXY)
    ).reshape(-1, 4)

    id_grid = _create_identity_grid(spatial_size, device=device, dtype=dtype)
    # We construct an approximation of inverse grid as inv_grid = id_grid - displacement
    # This is not an exact inverse of the grid
    inv_grid = id_grid.sub_(displacement)

    # Get points from bboxes
    points = bounding_box[:, [[0, 1], [2, 1], [2, 3], [0, 3]]].reshape(-1, 2)
    if points.is_floating_point():
        points = points.ceil_()
    index_xy = points.to(dtype=torch.long)
    index_x, index_y = index_xy[:, 0], index_xy[:, 1]

    # Transform points:
    t_size = torch.tensor(spatial_size[::-1], device=displacement.device, dtype=displacement.dtype)
    transformed_points = inv_grid[0, index_y, index_x, :].add_(1).mul_(0.5 * t_size).sub_(0.5)

    transformed_points = transformed_points.reshape(-1, 4, 2)
    out_bbox_mins, out_bbox_maxs = torch.aminmax(transformed_points, dim=1)
    out_bboxes = clamp_bounding_box(
        torch.cat([out_bbox_mins, out_bbox_maxs], dim=1).to(bounding_box.dtype),
        format=datapoints.BoundingBoxFormat.XYXY,
        spatial_size=spatial_size,
    )

    return convert_format_bounding_box(
        out_bboxes, old_format=datapoints.BoundingBoxFormat.XYXY, new_format=format, inplace=True
    ).reshape(original_shape)


def elastic_mask(
    mask: torch.Tensor,
    displacement: torch.Tensor,
    fill: datapoints._FillTypeJIT = None,
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
    interpolation: Union[InterpolationMode, int] = InterpolationMode.BILINEAR,
    fill: datapoints._FillTypeJIT = None,
) -> torch.Tensor:
    return elastic_image_tensor(video, displacement, interpolation=interpolation, fill=fill)


def elastic(
    inpt: datapoints._InputTypeJIT,
    displacement: torch.Tensor,
    interpolation: Union[InterpolationMode, int] = InterpolationMode.BILINEAR,
    fill: datapoints._FillTypeJIT = None,
) -> datapoints._InputTypeJIT:
    if not torch.jit.is_scripting():
        _log_api_usage_once(elastic)

    if not isinstance(displacement, torch.Tensor):
        raise TypeError("Argument displacement should be a Tensor")

    if torch.jit.is_scripting() or is_simple_tensor(inpt):
        return elastic_image_tensor(inpt, displacement, interpolation=interpolation, fill=fill)
    elif isinstance(inpt, datapoints._datapoint.Datapoint):
        return inpt.elastic(displacement, interpolation=interpolation, fill=fill)
    elif isinstance(inpt, PIL.Image.Image):
        return elastic_image_pil(inpt, displacement, interpolation=interpolation, fill=fill)
    else:
        raise TypeError(
            f"Input can either be a plain tensor, any TorchVision datapoint, or a PIL image, "
            f"but got {type(inpt)} instead."
        )


elastic_transform = elastic


def _center_crop_parse_output_size(output_size: List[int]) -> List[int]:
    if isinstance(output_size, numbers.Number):
        s = int(output_size)
        return [s, s]
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
    shape = image.shape
    if image.numel() == 0:
        return image.reshape(shape[:-2] + (crop_height, crop_width))
    image_height, image_width = shape[-2:]

    if crop_height > image_height or crop_width > image_width:
        padding_ltrb = _center_crop_compute_padding(crop_height, crop_width, image_height, image_width)
        image = torch_pad(image, _parse_pad_padding(padding_ltrb), value=0.0)

        image_height, image_width = image.shape[-2:]
        if crop_width == image_width and crop_height == image_height:
            return image

    crop_top, crop_left = _center_crop_compute_crop_anchor(crop_height, crop_width, image_height, image_width)
    return image[..., crop_top : (crop_top + crop_height), crop_left : (crop_left + crop_width)]


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
    format: datapoints.BoundingBoxFormat,
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


def center_crop(inpt: datapoints._InputTypeJIT, output_size: List[int]) -> datapoints._InputTypeJIT:
    if not torch.jit.is_scripting():
        _log_api_usage_once(center_crop)

    if torch.jit.is_scripting() or is_simple_tensor(inpt):
        return center_crop_image_tensor(inpt, output_size)
    elif isinstance(inpt, datapoints._datapoint.Datapoint):
        return inpt.center_crop(output_size)
    elif isinstance(inpt, PIL.Image.Image):
        return center_crop_image_pil(inpt, output_size)
    else:
        raise TypeError(
            f"Input can either be a plain tensor, any TorchVision datapoint, or a PIL image, "
            f"but got {type(inpt)} instead."
        )


def resized_crop_image_tensor(
    image: torch.Tensor,
    top: int,
    left: int,
    height: int,
    width: int,
    size: List[int],
    interpolation: Union[InterpolationMode, int] = InterpolationMode.BILINEAR,
    antialias: Optional[Union[str, bool]] = "warn",
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
    interpolation: Union[InterpolationMode, int] = InterpolationMode.BILINEAR,
) -> PIL.Image.Image:
    image = crop_image_pil(image, top, left, height, width)
    return resize_image_pil(image, size, interpolation=interpolation)


def resized_crop_bounding_box(
    bounding_box: torch.Tensor,
    format: datapoints.BoundingBoxFormat,
    top: int,
    left: int,
    height: int,
    width: int,
    size: List[int],
) -> Tuple[torch.Tensor, Tuple[int, int]]:
    bounding_box, _ = crop_bounding_box(bounding_box, format, top, left, height, width)
    return resize_bounding_box(bounding_box, spatial_size=(height, width), size=size)


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
    interpolation: Union[InterpolationMode, int] = InterpolationMode.BILINEAR,
    antialias: Optional[Union[str, bool]] = "warn",
) -> torch.Tensor:
    return resized_crop_image_tensor(
        video, top, left, height, width, antialias=antialias, size=size, interpolation=interpolation
    )


def resized_crop(
    inpt: datapoints._InputTypeJIT,
    top: int,
    left: int,
    height: int,
    width: int,
    size: List[int],
    interpolation: Union[InterpolationMode, int] = InterpolationMode.BILINEAR,
    antialias: Optional[Union[str, bool]] = "warn",
) -> datapoints._InputTypeJIT:
    if not torch.jit.is_scripting():
        _log_api_usage_once(resized_crop)

    if torch.jit.is_scripting() or is_simple_tensor(inpt):
        return resized_crop_image_tensor(
            inpt, top, left, height, width, antialias=antialias, size=size, interpolation=interpolation
        )
    elif isinstance(inpt, datapoints._datapoint.Datapoint):
        return inpt.resized_crop(top, left, height, width, antialias=antialias, size=size, interpolation=interpolation)
    elif isinstance(inpt, PIL.Image.Image):
        return resized_crop_image_pil(inpt, top, left, height, width, size=size, interpolation=interpolation)
    else:
        raise TypeError(
            f"Input can either be a plain tensor, any TorchVision datapoint, or a PIL image, "
            f"but got {type(inpt)} instead."
        )


def _parse_five_crop_size(size: List[int]) -> List[int]:
    if isinstance(size, numbers.Number):
        s = int(size)
        size = [s, s]
    elif isinstance(size, (tuple, list)) and len(size) == 1:
        s = size[0]
        size = [s, s]

    if len(size) != 2:
        raise ValueError("Please provide only two dimensions (h, w) for size.")

    return size


def five_crop_image_tensor(
    image: torch.Tensor, size: List[int]
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    crop_height, crop_width = _parse_five_crop_size(size)
    image_height, image_width = image.shape[-2:]

    if crop_width > image_width or crop_height > image_height:
        raise ValueError(f"Requested crop size {size} is bigger than input size {(image_height, image_width)}")

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
        raise ValueError(f"Requested crop size {size} is bigger than input size {(image_height, image_width)}")

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


ImageOrVideoTypeJIT = Union[datapoints._ImageTypeJIT, datapoints._VideoTypeJIT]


def five_crop(
    inpt: ImageOrVideoTypeJIT, size: List[int]
) -> Tuple[ImageOrVideoTypeJIT, ImageOrVideoTypeJIT, ImageOrVideoTypeJIT, ImageOrVideoTypeJIT, ImageOrVideoTypeJIT]:
    if not torch.jit.is_scripting():
        _log_api_usage_once(five_crop)

    if torch.jit.is_scripting() or is_simple_tensor(inpt):
        return five_crop_image_tensor(inpt, size)
    elif isinstance(inpt, datapoints.Image):
        output = five_crop_image_tensor(inpt.as_subclass(torch.Tensor), size)
        return tuple(datapoints.Image.wrap_like(inpt, item) for item in output)  # type: ignore[return-value]
    elif isinstance(inpt, datapoints.Video):
        output = five_crop_video(inpt.as_subclass(torch.Tensor), size)
        return tuple(datapoints.Video.wrap_like(inpt, item) for item in output)  # type: ignore[return-value]
    elif isinstance(inpt, PIL.Image.Image):
        return five_crop_image_pil(inpt, size)
    else:
        raise TypeError(
            f"Input can either be a plain tensor, an `Image` or `Video` datapoint, or a PIL image, "
            f"but got {type(inpt)} instead."
        )


def ten_crop_image_tensor(
    image: torch.Tensor, size: List[int], vertical_flip: bool = False
) -> Tuple[
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
]:
    non_flipped = five_crop_image_tensor(image, size)

    if vertical_flip:
        image = vertical_flip_image_tensor(image)
    else:
        image = horizontal_flip_image_tensor(image)

    flipped = five_crop_image_tensor(image, size)

    return non_flipped + flipped


@torch.jit.unused
def ten_crop_image_pil(
    image: PIL.Image.Image, size: List[int], vertical_flip: bool = False
) -> Tuple[
    PIL.Image.Image,
    PIL.Image.Image,
    PIL.Image.Image,
    PIL.Image.Image,
    PIL.Image.Image,
    PIL.Image.Image,
    PIL.Image.Image,
    PIL.Image.Image,
    PIL.Image.Image,
    PIL.Image.Image,
]:
    non_flipped = five_crop_image_pil(image, size)

    if vertical_flip:
        image = vertical_flip_image_pil(image)
    else:
        image = horizontal_flip_image_pil(image)

    flipped = five_crop_image_pil(image, size)

    return non_flipped + flipped


def ten_crop_video(
    video: torch.Tensor, size: List[int], vertical_flip: bool = False
) -> Tuple[
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
]:
    return ten_crop_image_tensor(video, size, vertical_flip=vertical_flip)


def ten_crop(
    inpt: Union[datapoints._ImageTypeJIT, datapoints._VideoTypeJIT], size: List[int], vertical_flip: bool = False
) -> Tuple[
    ImageOrVideoTypeJIT,
    ImageOrVideoTypeJIT,
    ImageOrVideoTypeJIT,
    ImageOrVideoTypeJIT,
    ImageOrVideoTypeJIT,
    ImageOrVideoTypeJIT,
    ImageOrVideoTypeJIT,
    ImageOrVideoTypeJIT,
    ImageOrVideoTypeJIT,
    ImageOrVideoTypeJIT,
]:
    if not torch.jit.is_scripting():
        _log_api_usage_once(ten_crop)

    if torch.jit.is_scripting() or is_simple_tensor(inpt):
        return ten_crop_image_tensor(inpt, size, vertical_flip=vertical_flip)
    elif isinstance(inpt, datapoints.Image):
        output = ten_crop_image_tensor(inpt.as_subclass(torch.Tensor), size, vertical_flip=vertical_flip)
        return tuple(datapoints.Image.wrap_like(inpt, item) for item in output)  # type: ignore[return-value]
    elif isinstance(inpt, datapoints.Video):
        output = ten_crop_video(inpt.as_subclass(torch.Tensor), size, vertical_flip=vertical_flip)
        return tuple(datapoints.Video.wrap_like(inpt, item) for item in output)  # type: ignore[return-value]
    elif isinstance(inpt, PIL.Image.Image):
        return ten_crop_image_pil(inpt, size, vertical_flip=vertical_flip)
    else:
        raise TypeError(
            f"Input can either be a plain tensor, an `Image` or `Video` datapoint, or a PIL image, "
            f"but got {type(inpt)} instead."
        )
