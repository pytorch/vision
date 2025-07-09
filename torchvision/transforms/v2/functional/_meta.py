from typing import Optional, Union

import PIL.Image
import torch
from torchvision import tv_tensors
from torchvision.transforms import _functional_pil as _FP
from torchvision.tv_tensors import BoundingBoxFormat
from torchvision.tv_tensors._bounding_boxes import CLAMPING_MODE_TYPE

from torchvision.utils import _log_api_usage_once

from ._utils import _get_kernel, _register_kernel_internal, is_pure_tensor


def get_dimensions(inpt: torch.Tensor) -> list[int]:
    if torch.jit.is_scripting():
        return get_dimensions_image(inpt)

    _log_api_usage_once(get_dimensions)

    kernel = _get_kernel(get_dimensions, type(inpt))
    return kernel(inpt)


@_register_kernel_internal(get_dimensions, torch.Tensor)
@_register_kernel_internal(get_dimensions, tv_tensors.Image, tv_tensor_wrapper=False)
def get_dimensions_image(image: torch.Tensor) -> list[int]:
    chw = list(image.shape[-3:])
    ndims = len(chw)
    if ndims == 3:
        return chw
    elif ndims == 2:
        chw.insert(0, 1)
        return chw
    else:
        raise TypeError(f"Input tensor should have at least two dimensions, but got {ndims}")


_get_dimensions_image_pil = _register_kernel_internal(get_dimensions, PIL.Image.Image)(_FP.get_dimensions)


@_register_kernel_internal(get_dimensions, tv_tensors.Video, tv_tensor_wrapper=False)
def get_dimensions_video(video: torch.Tensor) -> list[int]:
    return get_dimensions_image(video)


def get_num_channels(inpt: torch.Tensor) -> int:
    if torch.jit.is_scripting():
        return get_num_channels_image(inpt)

    _log_api_usage_once(get_num_channels)

    kernel = _get_kernel(get_num_channels, type(inpt))
    return kernel(inpt)


@_register_kernel_internal(get_num_channels, torch.Tensor)
@_register_kernel_internal(get_num_channels, tv_tensors.Image, tv_tensor_wrapper=False)
def get_num_channels_image(image: torch.Tensor) -> int:
    chw = image.shape[-3:]
    ndims = len(chw)
    if ndims == 3:
        return chw[0]
    elif ndims == 2:
        return 1
    else:
        raise TypeError(f"Input tensor should have at least two dimensions, but got {ndims}")


_get_num_channels_image_pil = _register_kernel_internal(get_num_channels, PIL.Image.Image)(_FP.get_image_num_channels)


@_register_kernel_internal(get_num_channels, tv_tensors.Video, tv_tensor_wrapper=False)
def get_num_channels_video(video: torch.Tensor) -> int:
    return get_num_channels_image(video)


# We changed the names to ensure it can be used not only for images but also videos. Thus, we just alias it without
# deprecating the old names.
get_image_num_channels = get_num_channels


def get_size(inpt: torch.Tensor) -> list[int]:
    if torch.jit.is_scripting():
        return get_size_image(inpt)

    _log_api_usage_once(get_size)

    kernel = _get_kernel(get_size, type(inpt))
    return kernel(inpt)


@_register_kernel_internal(get_size, torch.Tensor)
@_register_kernel_internal(get_size, tv_tensors.Image, tv_tensor_wrapper=False)
def get_size_image(image: torch.Tensor) -> list[int]:
    hw = list(image.shape[-2:])
    ndims = len(hw)
    if ndims == 2:
        return hw
    else:
        raise TypeError(f"Input tensor should have at least two dimensions, but got {ndims}")


@_register_kernel_internal(get_size, PIL.Image.Image)
def _get_size_image_pil(image: PIL.Image.Image) -> list[int]:
    width, height = _FP.get_image_size(image)
    return [height, width]


@_register_kernel_internal(get_size, tv_tensors.Video, tv_tensor_wrapper=False)
def get_size_video(video: torch.Tensor) -> list[int]:
    return get_size_image(video)


@_register_kernel_internal(get_size, tv_tensors.Mask, tv_tensor_wrapper=False)
def get_size_mask(mask: torch.Tensor) -> list[int]:
    return get_size_image(mask)


@_register_kernel_internal(get_size, tv_tensors.BoundingBoxes, tv_tensor_wrapper=False)
def get_size_bounding_boxes(bounding_box: tv_tensors.BoundingBoxes) -> list[int]:
    return list(bounding_box.canvas_size)


@_register_kernel_internal(get_size, tv_tensors.KeyPoints, tv_tensor_wrapper=False)
def get_size_keypoints(keypoints: tv_tensors.KeyPoints) -> list[int]:
    return list(keypoints.canvas_size)


def get_num_frames(inpt: torch.Tensor) -> int:
    if torch.jit.is_scripting():
        return get_num_frames_video(inpt)

    _log_api_usage_once(get_num_frames)

    kernel = _get_kernel(get_num_frames, type(inpt))
    return kernel(inpt)


@_register_kernel_internal(get_num_frames, torch.Tensor)
@_register_kernel_internal(get_num_frames, tv_tensors.Video, tv_tensor_wrapper=False)
def get_num_frames_video(video: torch.Tensor) -> int:
    return video.shape[-4]


def _xywh_to_xyxy(xywh: torch.Tensor, inplace: bool) -> torch.Tensor:
    xyxy = xywh if inplace else xywh.clone()
    xyxy[..., 2:] += xyxy[..., :2]
    return xyxy


def _xyxy_to_xywh(xyxy: torch.Tensor, inplace: bool) -> torch.Tensor:
    xywh = xyxy if inplace else xyxy.clone()
    xywh[..., 2:] -= xywh[..., :2]
    return xywh


def _cxcywh_to_xyxy(cxcywh: torch.Tensor, inplace: bool) -> torch.Tensor:
    if not inplace:
        cxcywh = cxcywh.clone()

    # Trick to do fast division by 2 and ceil, without casting. It produces the same result as
    # `torchvision.ops._box_convert._box_cxcywh_to_xyxy`.
    half_wh = cxcywh[..., 2:].div(-2, rounding_mode=None if cxcywh.is_floating_point() else "floor").abs_()
    # (cx - width / 2) = x1, same for y1
    cxcywh[..., :2].sub_(half_wh)
    # (x1 + width) = x2, same for y2
    cxcywh[..., 2:].add_(cxcywh[..., :2])

    return cxcywh


def _xyxy_to_cxcywh(xyxy: torch.Tensor, inplace: bool) -> torch.Tensor:
    if not inplace:
        xyxy = xyxy.clone()

    # (x2 - x1) = width, same for height
    xyxy[..., 2:].sub_(xyxy[..., :2])
    # (x1 * 2 + width) / 2 = x1 + width / 2 = x1 + (x2-x1)/2 = (x1 + x2)/2 = cx, same for cy
    xyxy[..., :2].mul_(2).add_(xyxy[..., 2:]).div_(2, rounding_mode=None if xyxy.is_floating_point() else "floor")

    return xyxy


def _xyxy_to_keypoints(bounding_boxes: torch.Tensor) -> torch.Tensor:
    return bounding_boxes[:, [[0, 1], [2, 1], [2, 3], [0, 3]]]


def _xyxyxyxy_to_keypoints(bounding_boxes: torch.Tensor) -> torch.Tensor:
    return bounding_boxes[:, [[0, 1], [2, 3], [4, 5], [6, 7]]]


def _cxcywhr_to_xywhr(cxcywhr: torch.Tensor, inplace: bool) -> torch.Tensor:
    if not inplace:
        cxcywhr = cxcywhr.clone()

    dtype = cxcywhr.dtype
    need_cast = not cxcywhr.is_floating_point()
    if need_cast:
        cxcywhr = cxcywhr.float()

    half_wh = cxcywhr[..., 2:-1].div(-2, rounding_mode=None if cxcywhr.is_floating_point() else "floor").abs_()
    r_rad = cxcywhr[..., 4].mul(torch.pi).div(180.0)
    cos, sin = r_rad.cos(), r_rad.sin()
    # (cx - width / 2 * cos - height / 2 * sin) = x1
    cxcywhr[..., 0].sub_(half_wh[..., 0].mul(cos)).sub_(half_wh[..., 1].mul(sin))
    # (cy + width / 2 * sin - height / 2 * cos) = y1
    cxcywhr[..., 1].add_(half_wh[..., 0].mul(sin)).sub_(half_wh[..., 1].mul(cos))

    if need_cast:
        cxcywhr.round_()
        cxcywhr = cxcywhr.to(dtype)

    return cxcywhr


def _xywhr_to_cxcywhr(xywhr: torch.Tensor, inplace: bool) -> torch.Tensor:
    if not inplace:
        xywhr = xywhr.clone()

    dtype = xywhr.dtype
    need_cast = not xywhr.is_floating_point()
    if need_cast:
        xywhr = xywhr.float()

    half_wh = xywhr[..., 2:-1].div(-2, rounding_mode=None if xywhr.is_floating_point() else "floor").abs_()
    r_rad = xywhr[..., 4].mul(torch.pi).div(180.0)
    cos, sin = r_rad.cos(), r_rad.sin()
    # (x1 + width / 2 * cos + height / 2 * sin) = cx
    xywhr[..., 0].add_(half_wh[..., 0].mul(cos)).add_(half_wh[..., 1].mul(sin))
    # (y1 - width / 2 * sin + height / 2 * cos) = cy
    xywhr[..., 1].sub_(half_wh[..., 0].mul(sin)).add_(half_wh[..., 1].mul(cos))

    if need_cast:
        xywhr.round_()
        xywhr = xywhr.to(dtype)

    return xywhr


def _xywhr_to_xyxyxyxy(xywhr: torch.Tensor, inplace: bool) -> torch.Tensor:
    # NOTE: This function cannot modify the input tensor inplace as it requires a dimension change.
    if not inplace:
        xywhr = xywhr.clone()

    dtype = xywhr.dtype
    need_cast = not xywhr.is_floating_point()
    if need_cast:
        xywhr = xywhr.float()

    wh = xywhr[..., 2:-1]
    r_rad = xywhr[..., 4].mul(torch.pi).div(180.0)
    cos, sin = r_rad.cos(), r_rad.sin()
    xywhr = xywhr[..., :2].tile((1, 4))
    # x1 + w * cos = x2
    xywhr[..., 2].add_(wh[..., 0].mul(cos))
    # y1 - w * sin = y2
    xywhr[..., 3].sub_(wh[..., 0].mul(sin))
    # x1 + w * cos + h * sin = x3
    xywhr[..., 4].add_(wh[..., 0].mul(cos).add(wh[..., 1].mul(sin)))
    # y1 - w * sin + h * cos = y3
    xywhr[..., 5].sub_(wh[..., 0].mul(sin).sub(wh[..., 1].mul(cos)))
    # x1 + h * sin = x4
    xywhr[..., 6].add_(wh[..., 1].mul(sin))
    # y1 + h * cos = y4
    xywhr[..., 7].add_(wh[..., 1].mul(cos))

    if need_cast:
        xywhr.round_()
        xywhr = xywhr.to(dtype)

    return xywhr


def _xyxyxyxy_to_xywhr(xyxyxyxy: torch.Tensor, inplace: bool) -> torch.Tensor:
    # NOTE: This function cannot modify the input tensor inplace as it requires a dimension change.
    if not inplace:
        xyxyxyxy = xyxyxyxy.clone()

    dtype = xyxyxyxy.dtype
    need_cast = not xyxyxyxy.is_floating_point()
    if need_cast:
        xyxyxyxy = xyxyxyxy.float()

    r_rad = torch.atan2(xyxyxyxy[..., 1].sub(xyxyxyxy[..., 3]), xyxyxyxy[..., 2].sub(xyxyxyxy[..., 0]))
    # x1, y1, (x2 - x1), (y2 - y1), (x3 - x2), (y3 - y2) x4, y4
    xyxyxyxy[..., 4:6].sub_(xyxyxyxy[..., 2:4])
    xyxyxyxy[..., 2:4].sub_(xyxyxyxy[..., :2])
    # sqrt((x2 - x1) ** 2 + (y1 - y2) ** 2) = w
    xyxyxyxy[..., 2] = xyxyxyxy[..., 2].pow(2).add(xyxyxyxy[..., 3].pow(2)).sqrt()
    # sqrt((x2 - x3) ** 2 + (y2 - y3) ** 2) = h
    xyxyxyxy[..., 3] = xyxyxyxy[..., 4].pow(2).add(xyxyxyxy[..., 5].pow(2)).sqrt()
    xyxyxyxy[..., 4] = r_rad.div_(torch.pi).mul_(180.0)

    if need_cast:
        xyxyxyxy.round_()
        xyxyxyxy = xyxyxyxy.to(dtype)

    return xyxyxyxy[..., :5]


def _convert_bounding_box_format(
    bounding_boxes: torch.Tensor, old_format: BoundingBoxFormat, new_format: BoundingBoxFormat, inplace: bool = False
) -> torch.Tensor:

    if new_format == old_format:
        return bounding_boxes

    if tv_tensors.is_rotated_bounding_format(old_format) ^ tv_tensors.is_rotated_bounding_format(new_format):
        raise ValueError("Cannot convert between rotated and unrotated bounding boxes.")

    # TODO: Add _xywh_to_cxcywh and _cxcywh_to_xywh to improve performance
    if old_format == BoundingBoxFormat.XYWH:
        bounding_boxes = _xywh_to_xyxy(bounding_boxes, inplace)
    elif old_format == BoundingBoxFormat.CXCYWH:
        bounding_boxes = _cxcywh_to_xyxy(bounding_boxes, inplace)
    elif old_format == BoundingBoxFormat.CXCYWHR:
        bounding_boxes = _cxcywhr_to_xywhr(bounding_boxes, inplace)
    elif old_format == BoundingBoxFormat.XYXYXYXY:
        bounding_boxes = _xyxyxyxy_to_xywhr(bounding_boxes, inplace)

    if new_format == BoundingBoxFormat.XYWH:
        bounding_boxes = _xyxy_to_xywh(bounding_boxes, inplace)
    elif new_format == BoundingBoxFormat.CXCYWH:
        bounding_boxes = _xyxy_to_cxcywh(bounding_boxes, inplace)
    elif new_format == BoundingBoxFormat.CXCYWHR:
        bounding_boxes = _xywhr_to_cxcywhr(bounding_boxes, inplace)
    elif new_format == BoundingBoxFormat.XYXYXYXY:
        bounding_boxes = _xywhr_to_xyxyxyxy(bounding_boxes, inplace)

    return bounding_boxes


def convert_bounding_box_format(
    inpt: torch.Tensor,
    old_format: Optional[BoundingBoxFormat] = None,
    new_format: Optional[BoundingBoxFormat] = None,
    inplace: bool = False,
) -> torch.Tensor:
    """See :func:`~torchvision.transforms.v2.ConvertBoundingBoxFormat` for details."""
    # This being a kernel / functional hybrid, we need an option to pass `old_format` explicitly for pure tensor
    # inputs as well as extract it from `tv_tensors.BoundingBoxes` inputs. However, putting a default value on
    # `old_format` means we also need to put one on `new_format` to have syntactically correct Python. Here we mimic the
    # default error that would be thrown if `new_format` had no default value.
    if new_format is None:
        raise TypeError("convert_bounding_box_format() missing 1 required argument: 'new_format'")

    if not torch.jit.is_scripting():
        _log_api_usage_once(convert_bounding_box_format)

    if isinstance(old_format, str):
        old_format = BoundingBoxFormat[old_format.upper()]
    if isinstance(new_format, str):
        new_format = BoundingBoxFormat[new_format.upper()]

    if torch.jit.is_scripting() or is_pure_tensor(inpt):
        if old_format is None:
            raise ValueError("For pure tensor inputs, `old_format` has to be passed.")
        return _convert_bounding_box_format(inpt, old_format=old_format, new_format=new_format, inplace=inplace)
    elif isinstance(inpt, tv_tensors.BoundingBoxes):
        if old_format is not None:
            raise ValueError("For bounding box tv_tensor inputs, `old_format` must not be passed.")
        output = _convert_bounding_box_format(
            inpt.as_subclass(torch.Tensor), old_format=inpt.format, new_format=new_format, inplace=inplace
        )
        return tv_tensors.wrap(output, like=inpt, format=new_format)
    else:
        raise TypeError(
            f"Input can either be a plain tensor or a bounding box tv_tensor, but got {type(inpt)} instead."
        )


def _clamp_bounding_boxes(
    bounding_boxes: torch.Tensor,
    format: BoundingBoxFormat,
    canvas_size: tuple[int, int],
    clamping_mode: CLAMPING_MODE_TYPE,
) -> torch.Tensor:
    if clamping_mode is None:
        return bounding_boxes.clone()
    # TODO: Investigate if it makes sense from a performance perspective to have an implementation for every
    #  BoundingBoxFormat instead of converting back and forth
    in_dtype = bounding_boxes.dtype
    bounding_boxes = bounding_boxes.clone() if bounding_boxes.is_floating_point() else bounding_boxes.float()
    xyxy_boxes = convert_bounding_box_format(
        bounding_boxes, old_format=format, new_format=tv_tensors.BoundingBoxFormat.XYXY, inplace=True
    )
    # hard and soft modes are equivalent for non-rotated boxes
    xyxy_boxes[..., 0::2].clamp_(min=0, max=canvas_size[1])
    xyxy_boxes[..., 1::2].clamp_(min=0, max=canvas_size[0])
    out_boxes = convert_bounding_box_format(
        xyxy_boxes, old_format=BoundingBoxFormat.XYXY, new_format=format, inplace=True
    )
    return out_boxes.to(in_dtype)


def _order_bounding_boxes_points(
    bounding_boxes: torch.Tensor, indices: Optional[torch.Tensor] = None
) -> tuple[torch.Tensor, torch.Tensor]:
    """Re-order points in bounding boxes based on specific criteria or provided indices.

    This function reorders the points of bounding boxes either according to provided indices or
    by a default ordering strategy. In the default strategy, (x1, y1) corresponds to the point
    with the lowest x value. If multiple points have the same lowest x value, the point with the
    lowest y value is chosen.

    Args:
        bounding_boxes (torch.Tensor): A tensor containing bounding box coordinates in format [x1, y1, x2, y2, x3, y3, x4, y4].
        indices (torch.Tensor | None): Optional tensor containing indices for reordering. If None, default ordering is applied.

    Returns:
        tuple[torch.Tensor, torch.Tensor]: A tuple containing:
            - indices: The indices used for reordering
            - reordered_boxes: The bounding boxes with reordered points
    """
    if indices is None:
        output_xyxyxyxy = bounding_boxes.reshape(-1, 8)
        x, y = output_xyxyxyxy[..., 0::2], output_xyxyxyxy[..., 1::2]
        y_max = torch.max(y.abs(), dim=1, keepdim=True)[0]
        x_max = torch.max(x.abs(), dim=1, keepdim=True)[0]
        _, x1 = (y / y_max + (x / x_max) * 100).min(dim=1)
        indices = torch.ones_like(output_xyxyxyxy)
        indices[..., 0] = x1.mul(2)
        indices.cumsum_(1).remainder_(8)
    return indices, bounding_boxes.gather(1, indices.to(torch.int64))


def _get_slope_and_intercept(box: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Calculate the slope and y-intercept of the lines defined by consecutive vertices in a bounding box.
    This function computes the slope (a) and y-intercept (b) for each line segment in a bounding box,
    where each line is defined by two consecutive vertices.
    """
    x, y = box[..., ::2], box[..., 1::2]
    a = y.diff(append=y[..., 0:1]) / x.diff(append=x[..., 0:1])
    b = y - a * x
    return a, b


def _get_intersection_point(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """
    Calculate the intersection point of two lines defined by their slopes and y-intercepts.
    This function computes the intersection points between pairs of lines, where each line
    is defined by the equation y = ax + b (slope and y-intercept form).
    """
    batch_size = a.shape[0]
    x = b.diff(prepend=b[..., 3:4]).neg() / a.diff(prepend=a[..., 3:4])
    y = a * x + b
    return torch.cat((x.unsqueeze(-1), y.unsqueeze(-1)), dim=-1).view(batch_size, 8)


def _clamp_y_intercept(
    bounding_boxes: torch.Tensor,
    original_bounding_boxes: torch.Tensor,
    canvas_size: tuple[int, int],
    clamping_mode: CLAMPING_MODE_TYPE,
) -> torch.Tensor:
    """
    Apply clamping to bounding box y-intercepts. This function handles two clamping strategies:
    - Hard clamping: Ensures all box vertices stay within canvas boundaries, finding the largest
      angle-preserving box enclosed within the original box and the image canvas.
    - Soft clamping: Allows some vertices to extend beyond the canvas, finding the smallest
      angle-preserving box that encloses the intersection of the original box and the image canvas.

    The function first calculates the slopes and y-intercepts of the lines forming the bounding box,
    then applies various constraints to ensure the clamping conditions are respected.
    """

    # Calculate slopes and y-intercepts for bounding boxes
    a, b = _get_slope_and_intercept(bounding_boxes)
    a1, a2, a3, a4 = a.unbind(-1)
    b1, b2, b3, b4 = b.unbind(-1)

    # Get y-intercepts from original bounding boxes
    _, bm = _get_slope_and_intercept(original_bounding_boxes)
    b1m, b2m, b3m, b4m = bm.unbind(-1)

    # Soft clamping: Clamp y-intercepts within canvas boundaries
    b1 = b2.clamp(b1, b3).clamp(0, canvas_size[0])
    b4 = b3.clamp(b2, b4).clamp(0, canvas_size[0])

    if clamping_mode is not None and clamping_mode == "hard":
        # Hard clamping: Average b1 and b4, and adjust b2 and b3 for maximum area
        b1 = b4 = (b1 + b4) / 2

        # Calculate candidate values for b2 based on geometric constraints
        b2_candidates = torch.stack(
            [
                b1 * a2 / a1,  # Constraint at y=0
                b3 * a2 / a3,  # Constraint at y=0
                (a1 - a2) * canvas_size[1] + b1,  # Constraint at x=canvas_width
                (a3 - a2) * canvas_size[1] + b3,  # Constraint at x=canvas_width
            ],
            dim=1,
        )
        # Take maximum value that doesn't exceed original b2
        b2 = torch.max(b2_candidates, dim=1)[0].clamp(max=b2)

        # Calculate candidate values for b3 based on geometric constraints
        b3_candidates = torch.stack(
            [
                canvas_size[0] * (1 - a3 / a4) + b4 * a3 / a4,  # Constraint at y=canvas_height
                canvas_size[0] * (1 - a3 / a2) + b2 * a3 / a2,  # Constraint at y=canvas_height
                (a2 - a3) * canvas_size[1] + b2,  # Constraint at x=canvas_width
                (a4 - a3) * canvas_size[1] + b4,  # Constraint at x=canvas_width
            ],
            dim=1,
        )
        # Take minimum value that doesn't go below original b3
        b3 = torch.min(b3_candidates, dim=1)[0].clamp(min=b3)

    # Final clamping to ensure y-intercepts are within original box bounds
    b1.clamp_(b1m, b3m)
    b3.clamp_(b1m, b3m)
    b2.clamp_(b2m, b4m)
    b4.clamp_(b2m, b4m)

    return torch.stack([b1, b2, b3, b4], dim=-1)


def _clamp_along_y_axis(
    bounding_boxes: torch.Tensor,
    original_bounding_boxes: torch.Tensor,
    canvas_size: tuple[int, int],
    clamping_mode: CLAMPING_MODE_TYPE,
) -> torch.Tensor:
    """
    Adjusts bounding boxes along the y-axis based on specific conditions.

    This function modifies the bounding boxes by evaluating different cases
    and applying the appropriate transformation to ensure the bounding boxes
    are clamped correctly along the y-axis.

    Args:
        bounding_boxes (torch.Tensor): A tensor containing bounding box coordinates.
        original_bounding_boxes (torch.Tensor): The original bounding boxes before any clamping is applied.
        canvas_size (tuple[int, int]): The size of the canvas as (height, width).
        clamping_mode (str, optional): The clamping strategy to use.

    Returns:
        torch.Tensor: The adjusted bounding boxes.
    """
    original_shape = bounding_boxes.shape
    bounding_boxes = bounding_boxes.reshape(-1, 8)
    original_bounding_boxes = original_bounding_boxes.reshape(-1, 8)

    # Calculate slopes (a) and y-intercepts (b) for all lines in the bounding boxes
    a, b = _get_slope_and_intercept(bounding_boxes)
    x1, y1, x2, y2, x3, y3, x4, y4 = bounding_boxes.unbind(-1)
    b = _clamp_y_intercept(bounding_boxes, original_bounding_boxes, canvas_size, clamping_mode)

    case_a = _get_intersection_point(a, b)
    case_b = bounding_boxes.clone()
    case_b[..., 0].clamp_(0)  # Clamp x1 to 0
    case_b[..., 6].clamp_(0)  # Clamp x4 to 0
    case_c = torch.zeros_like(case_b)

    cond_a = (x1 < 0) & ~case_a.isnan().any(-1)  # First point is outside left boundary
    cond_b = y1.isclose(y2) | y3.isclose(y4)  # First line is nearly vertical
    cond_c = (x1 <= 0) & (x2 <= 0) & (x3 <= 0) & (x4 <= 0)  # All points outside left boundary
    cond_c = cond_c | y1.isclose(y4) | y2.isclose(y3) | (cond_b & x1.isclose(x2))  # First line is nearly horizontal

    for (cond, case) in zip(
        [cond_a, cond_b, cond_c],
        [case_a, case_b, case_c],
    ):
        bounding_boxes = torch.where(cond.unsqueeze(1).repeat(1, 8), case.reshape(-1, 8), bounding_boxes)

    return bounding_boxes.reshape(original_shape)


def _clamp_rotated_bounding_boxes(
    bounding_boxes: torch.Tensor,
    format: BoundingBoxFormat,
    canvas_size: tuple[int, int],
    clamping_mode: CLAMPING_MODE_TYPE,
) -> torch.Tensor:
    """
    Clamp rotated bounding boxes to ensure they stay within the canvas boundaries.

    This function handles rotated bounding boxes by:
    1. Converting them to XYXYXYXY format (8 coordinates representing 4 corners).
    2. Re-ordering the points in the bounding boxes to ensure (x1, y1) corresponds to the point with the lowest x value
    2. Translates the points (x1, y1), (x2, y2) and (x3, y3)
        to ensure the bounding box does not go out beyond the left boundary of the canvas.
    3. Rotate the bounding box four times and apply the same transformation to each vertex to ensure
        the box does not go beyond the top, right, and bottom boundaries.
    3. Converting back to the original format and re-order the points as in the original input.

    Args:
        bounding_boxes (torch.Tensor): Tensor containing rotated bounding box coordinates
        format (BoundingBoxFormat): The format of the input bounding boxes
        canvas_size (tuple[int, int]): The size of the canvas as (height, width)

    Returns:
        torch.Tensor: Clamped bounding boxes in the original format and shape
    """
    if clamping_mode is None:
        return bounding_boxes.clone()
    original_shape = bounding_boxes.shape
    bounding_boxes = bounding_boxes.clone()
    out_boxes = (
        convert_bounding_box_format(
            bounding_boxes, old_format=format, new_format=tv_tensors.BoundingBoxFormat.XYXYXYXY, inplace=True
        )
    ).reshape(-1, 8)

    original_boxes = out_boxes.clone()
    for _ in range(4):  # Iterate over the 4 vertices.
        indices, out_boxes = _order_bounding_boxes_points(out_boxes)
        _, original_boxes = _order_bounding_boxes_points(original_boxes, indices)
        out_boxes = _clamp_along_y_axis(out_boxes, original_boxes, canvas_size, clamping_mode)
        _, out_boxes = _order_bounding_boxes_points(out_boxes, indices)
        _, original_boxes = _order_bounding_boxes_points(original_boxes, indices)
        # rotate 90 degrees counter clock wise
        out_boxes[:, ::2], out_boxes[:, 1::2] = (
            out_boxes[:, 1::2].clone(),
            canvas_size[1] - out_boxes[:, ::2].clone(),
        )
        original_boxes[:, ::2], original_boxes[:, 1::2] = (
            original_boxes[:, 1::2].clone(),
            canvas_size[1] - original_boxes[:, ::2].clone(),
        )
        canvas_size = (canvas_size[1], canvas_size[0])

    out_boxes = convert_bounding_box_format(
        out_boxes, old_format=tv_tensors.BoundingBoxFormat.XYXYXYXY, new_format=format, inplace=True
    ).reshape(original_shape)

    return out_boxes


def clamp_bounding_boxes(
    inpt: torch.Tensor,
    format: Optional[BoundingBoxFormat] = None,
    canvas_size: Optional[tuple[int, int]] = None,
    clamping_mode: Union[CLAMPING_MODE_TYPE, str] = "auto",
) -> torch.Tensor:
    """See :func:`~torchvision.transforms.v2.ClampBoundingBoxes` for details."""
    if not torch.jit.is_scripting():
        _log_api_usage_once(clamp_bounding_boxes)

    if clamping_mode is not None and clamping_mode not in ("soft", "hard", "auto"):
        raise ValueError(f"clamping_mode must be soft, hard, auto or None, got {clamping_mode}")

    if torch.jit.is_scripting() or is_pure_tensor(inpt):

        if format is None or canvas_size is None or (clamping_mode is not None and clamping_mode == "auto"):
            raise ValueError("For pure tensor inputs, `format`, `canvas_size` and `clamping_mode` have to be passed.")
        if tv_tensors.is_rotated_bounding_format(format):
            return _clamp_rotated_bounding_boxes(
                inpt, format=format, canvas_size=canvas_size, clamping_mode=clamping_mode
            )
        else:
            return _clamp_bounding_boxes(inpt, format=format, canvas_size=canvas_size, clamping_mode=clamping_mode)
    elif isinstance(inpt, tv_tensors.BoundingBoxes):
        if format is not None or canvas_size is not None:
            raise ValueError("For bounding box tv_tensor inputs, `format` and `canvas_size` must not be passed.")
        if clamping_mode is not None and clamping_mode == "auto":
            clamping_mode = inpt.clamping_mode
        if tv_tensors.is_rotated_bounding_format(inpt.format):
            output = _clamp_rotated_bounding_boxes(
                inpt.as_subclass(torch.Tensor),
                format=inpt.format,
                canvas_size=inpt.canvas_size,
                clamping_mode=clamping_mode,
            )
        else:
            output = _clamp_bounding_boxes(
                inpt.as_subclass(torch.Tensor),
                format=inpt.format,
                canvas_size=inpt.canvas_size,
                clamping_mode=clamping_mode,
            )
        return tv_tensors.wrap(output, like=inpt)
    else:
        raise TypeError(
            f"Input can either be a plain tensor or a bounding box tv_tensor, but got {type(inpt)} instead."
        )


def _clamp_keypoints(keypoints: torch.Tensor, canvas_size: tuple[int, int]) -> torch.Tensor:
    dtype = keypoints.dtype
    keypoints = keypoints.clone() if keypoints.is_floating_point() else keypoints.float()
    # Note that max is canvas_size[i] - 1 and not can canvas_size[i] like for
    # bounding boxes.
    keypoints[..., 0].clamp_(min=0, max=canvas_size[1] - 1)
    keypoints[..., 1].clamp_(min=0, max=canvas_size[0] - 1)
    return keypoints.to(dtype=dtype)


def clamp_keypoints(
    inpt: torch.Tensor,
    canvas_size: Optional[tuple[int, int]] = None,
) -> torch.Tensor:
    """See :func:`~torchvision.transforms.v2.ClampKeyPoints` for details."""
    if not torch.jit.is_scripting():
        _log_api_usage_once(clamp_keypoints)

    if torch.jit.is_scripting() or is_pure_tensor(inpt):

        if canvas_size is None:
            raise ValueError("For pure tensor inputs, `canvas_size` has to be passed.")
        return _clamp_keypoints(inpt, canvas_size=canvas_size)
    elif isinstance(inpt, tv_tensors.KeyPoints):
        if canvas_size is not None:
            raise ValueError("For keypoints tv_tensor inputs, `canvas_size` must not be passed.")
        output = _clamp_keypoints(inpt.as_subclass(torch.Tensor), canvas_size=inpt.canvas_size)
        return tv_tensors.wrap(output, like=inpt)
    else:
        raise TypeError(f"Input can either be a plain tensor or a keypoints tv_tensor, but got {type(inpt)} instead.")
