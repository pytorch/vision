from typing import List, Optional, Tuple, Union

import PIL.Image
import torch
from torchvision import datapoints
from torchvision.datapoints import BoundingBoxFormat
from torchvision.transforms import _functional_pil as _FP
from torchvision.transforms._functional_tensor import _max_value

from torchvision.utils import _log_api_usage_once

from ._utils import is_simple_tensor


def get_dimensions_image_tensor(image: torch.Tensor) -> List[int]:
    chw = list(image.shape[-3:])
    ndims = len(chw)
    if ndims == 3:
        return chw
    elif ndims == 2:
        chw.insert(0, 1)
        return chw
    else:
        raise TypeError(f"Input tensor should have at least two dimensions, but got {ndims}")


get_dimensions_image_pil = _FP.get_dimensions


def get_dimensions(inpt: Union[datapoints._ImageTypeJIT, datapoints._VideoTypeJIT]) -> List[int]:
    if not torch.jit.is_scripting():
        _log_api_usage_once(get_dimensions)

    if torch.jit.is_scripting() or is_simple_tensor(inpt):
        return get_dimensions_image_tensor(inpt)
    elif isinstance(inpt, (datapoints.Image, datapoints.Video)):
        channels = inpt.num_channels
        height, width = inpt.spatial_size
        return [channels, height, width]
    elif isinstance(inpt, PIL.Image.Image):
        return get_dimensions_image_pil(inpt)
    else:
        raise TypeError(
            f"Input can either be a plain tensor, an `Image` or `Video` datapoint, or a PIL image, "
            f"but got {type(inpt)} instead."
        )


def get_num_channels_image_tensor(image: torch.Tensor) -> int:
    chw = image.shape[-3:]
    ndims = len(chw)
    if ndims == 3:
        return chw[0]
    elif ndims == 2:
        return 1
    else:
        raise TypeError(f"Input tensor should have at least two dimensions, but got {ndims}")


get_num_channels_image_pil = _FP.get_image_num_channels


def get_num_channels_video(video: torch.Tensor) -> int:
    return get_num_channels_image_tensor(video)


def get_num_channels(inpt: Union[datapoints._ImageTypeJIT, datapoints._VideoTypeJIT]) -> int:
    if not torch.jit.is_scripting():
        _log_api_usage_once(get_num_channels)

    if torch.jit.is_scripting() or is_simple_tensor(inpt):
        return get_num_channels_image_tensor(inpt)
    elif isinstance(inpt, (datapoints.Image, datapoints.Video)):
        return inpt.num_channels
    elif isinstance(inpt, PIL.Image.Image):
        return get_num_channels_image_pil(inpt)
    else:
        raise TypeError(
            f"Input can either be a plain tensor, an `Image` or `Video` datapoint, or a PIL image, "
            f"but got {type(inpt)} instead."
        )


# We changed the names to ensure it can be used not only for images but also videos. Thus, we just alias it without
# deprecating the old names.
get_image_num_channels = get_num_channels


def get_spatial_size_image_tensor(image: torch.Tensor) -> List[int]:
    hw = list(image.shape[-2:])
    ndims = len(hw)
    if ndims == 2:
        return hw
    else:
        raise TypeError(f"Input tensor should have at least two dimensions, but got {ndims}")


@torch.jit.unused
def get_spatial_size_image_pil(image: PIL.Image.Image) -> List[int]:
    width, height = _FP.get_image_size(image)
    return [height, width]


def get_spatial_size_video(video: torch.Tensor) -> List[int]:
    return get_spatial_size_image_tensor(video)


def get_spatial_size_mask(mask: torch.Tensor) -> List[int]:
    return get_spatial_size_image_tensor(mask)


@torch.jit.unused
def get_spatial_size_bounding_box(bounding_box: datapoints.BoundingBox) -> List[int]:
    return list(bounding_box.spatial_size)


def get_spatial_size(inpt: datapoints._InputTypeJIT) -> List[int]:
    if not torch.jit.is_scripting():
        _log_api_usage_once(get_spatial_size)

    if torch.jit.is_scripting() or is_simple_tensor(inpt):
        return get_spatial_size_image_tensor(inpt)
    elif isinstance(inpt, (datapoints.Image, datapoints.Video, datapoints.BoundingBox, datapoints.Mask)):
        return list(inpt.spatial_size)
    elif isinstance(inpt, PIL.Image.Image):
        return get_spatial_size_image_pil(inpt)
    else:
        raise TypeError(
            f"Input can either be a plain tensor, any TorchVision datapoint, or a PIL image, "
            f"but got {type(inpt)} instead."
        )


def get_num_frames_video(video: torch.Tensor) -> int:
    return video.shape[-4]


def get_num_frames(inpt: datapoints._VideoTypeJIT) -> int:
    if not torch.jit.is_scripting():
        _log_api_usage_once(get_num_frames)

    if torch.jit.is_scripting() or is_simple_tensor(inpt):
        return get_num_frames_video(inpt)
    elif isinstance(inpt, datapoints.Video):
        return inpt.num_frames
    else:
        raise TypeError(f"Input can either be a plain tensor or a `Video` datapoint, but got {type(inpt)} instead.")


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


def _convert_format_bounding_box(
    bounding_box: torch.Tensor, old_format: BoundingBoxFormat, new_format: BoundingBoxFormat, inplace: bool = False
) -> torch.Tensor:

    if new_format == old_format:
        return bounding_box

    # TODO: Add _xywh_to_cxcywh and _cxcywh_to_xywh to improve performance
    if old_format == BoundingBoxFormat.XYWH:
        bounding_box = _xywh_to_xyxy(bounding_box, inplace)
    elif old_format == BoundingBoxFormat.CXCYWH:
        bounding_box = _cxcywh_to_xyxy(bounding_box, inplace)

    if new_format == BoundingBoxFormat.XYWH:
        bounding_box = _xyxy_to_xywh(bounding_box, inplace)
    elif new_format == BoundingBoxFormat.CXCYWH:
        bounding_box = _xyxy_to_cxcywh(bounding_box, inplace)

    return bounding_box


def convert_format_bounding_box(
    inpt: datapoints._InputTypeJIT,
    old_format: Optional[BoundingBoxFormat] = None,
    new_format: Optional[BoundingBoxFormat] = None,
    inplace: bool = False,
) -> datapoints._InputTypeJIT:
    # This being a kernel / dispatcher hybrid, we need an option to pass `old_format` explicitly for simple tensor
    # inputs as well as extract it from `datapoints.BoundingBox` inputs. However, putting a default value on
    # `old_format` means we also need to put one on `new_format` to have syntactically correct Python. Here we mimic the
    # default error that would be thrown if `new_format` had no default value.
    if new_format is None:
        raise TypeError("convert_format_bounding_box() missing 1 required argument: 'new_format'")

    if not torch.jit.is_scripting():
        _log_api_usage_once(convert_format_bounding_box)

    if torch.jit.is_scripting() or is_simple_tensor(inpt):
        if old_format is None:
            raise ValueError("For simple tensor inputs, `old_format` has to be passed.")
        return _convert_format_bounding_box(inpt, old_format=old_format, new_format=new_format, inplace=inplace)
    elif isinstance(inpt, datapoints.BoundingBox):
        if old_format is not None:
            raise ValueError("For bounding box datapoint inputs, `old_format` must not be passed.")
        output = _convert_format_bounding_box(
            inpt.as_subclass(torch.Tensor), old_format=inpt.format, new_format=new_format, inplace=inplace
        )
        return datapoints.BoundingBox.wrap_like(inpt, output, format=new_format)
    else:
        raise TypeError(
            f"Input can either be a plain tensor or a bounding box datapoint, but got {type(inpt)} instead."
        )


def _clamp_bounding_box(
    bounding_box: torch.Tensor, format: BoundingBoxFormat, spatial_size: Tuple[int, int]
) -> torch.Tensor:
    # TODO: Investigate if it makes sense from a performance perspective to have an implementation for every
    #  BoundingBoxFormat instead of converting back and forth
    in_dtype = bounding_box.dtype
    bounding_box = bounding_box.clone() if bounding_box.is_floating_point() else bounding_box.float()
    xyxy_boxes = convert_format_bounding_box(
        bounding_box, old_format=format, new_format=datapoints.BoundingBoxFormat.XYXY, inplace=True
    )
    xyxy_boxes[..., 0::2].clamp_(min=0, max=spatial_size[1])
    xyxy_boxes[..., 1::2].clamp_(min=0, max=spatial_size[0])
    out_boxes = convert_format_bounding_box(
        xyxy_boxes, old_format=BoundingBoxFormat.XYXY, new_format=format, inplace=True
    )
    return out_boxes.to(in_dtype)


def clamp_bounding_box(
    inpt: datapoints._InputTypeJIT,
    format: Optional[BoundingBoxFormat] = None,
    spatial_size: Optional[Tuple[int, int]] = None,
) -> datapoints._InputTypeJIT:
    if not torch.jit.is_scripting():
        _log_api_usage_once(clamp_bounding_box)

    if torch.jit.is_scripting() or is_simple_tensor(inpt):
        if format is None or spatial_size is None:
            raise ValueError("For simple tensor inputs, `format` and `spatial_size` has to be passed.")
        return _clamp_bounding_box(inpt, format=format, spatial_size=spatial_size)
    elif isinstance(inpt, datapoints.BoundingBox):
        if format is not None or spatial_size is not None:
            raise ValueError("For bounding box datapoint inputs, `format` and `spatial_size` must not be passed.")
        output = _clamp_bounding_box(inpt.as_subclass(torch.Tensor), format=inpt.format, spatial_size=inpt.spatial_size)
        return datapoints.BoundingBox.wrap_like(inpt, output)
    else:
        raise TypeError(
            f"Input can either be a plain tensor or a bounding box datapoint, but got {type(inpt)} instead."
        )


def _num_value_bits(dtype: torch.dtype) -> int:
    if dtype == torch.uint8:
        return 8
    elif dtype == torch.int8:
        return 7
    elif dtype == torch.int16:
        return 15
    elif dtype == torch.int32:
        return 31
    elif dtype == torch.int64:
        return 63
    else:
        raise TypeError(f"Number of value bits is only defined for integer dtypes, but got {dtype}.")


def convert_dtype_image_tensor(image: torch.Tensor, dtype: torch.dtype = torch.float) -> torch.Tensor:
    if image.dtype == dtype:
        return image

    float_input = image.is_floating_point()
    if torch.jit.is_scripting():
        # TODO: remove this branch as soon as `dtype.is_floating_point` is supported by JIT
        float_output = torch.tensor(0, dtype=dtype).is_floating_point()
    else:
        float_output = dtype.is_floating_point

    if float_input:
        # float to float
        if float_output:
            return image.to(dtype)

        # float to int
        if (image.dtype == torch.float32 and dtype in (torch.int32, torch.int64)) or (
            image.dtype == torch.float64 and dtype == torch.int64
        ):
            raise RuntimeError(f"The conversion from {image.dtype} to {dtype} cannot be performed safely.")

        # For data in the range `[0.0, 1.0]`, just multiplying by the maximum value of the integer range and converting
        # to the integer dtype  is not sufficient. For example, `torch.rand(...).mul(255).to(torch.uint8)` will only
        # be `255` if the input is exactly `1.0`. See https://github.com/pytorch/vision/pull/2078#issuecomment-612045321
        # for a detailed analysis.
        # To mitigate this, we could round before we convert to the integer dtype, but this is an extra operation.
        # Instead, we can also multiply by the maximum value plus something close to `1`. See
        # https://github.com/pytorch/vision/pull/2078#issuecomment-613524965 for details.
        eps = 1e-3
        max_value = float(_max_value(dtype))
        # We need to scale first since the conversion would otherwise turn the input range `[0.0, 1.0]` into the
        # discrete set `{0, 1}`.
        return image.mul(max_value + 1.0 - eps).to(dtype)
    else:
        # int to float
        if float_output:
            return image.to(dtype).mul_(1.0 / _max_value(image.dtype))

        # int to int
        num_value_bits_input = _num_value_bits(image.dtype)
        num_value_bits_output = _num_value_bits(dtype)

        if num_value_bits_input > num_value_bits_output:
            return image.bitwise_right_shift(num_value_bits_input - num_value_bits_output).to(dtype)
        else:
            return image.to(dtype).bitwise_left_shift_(num_value_bits_output - num_value_bits_input)


# We changed the name to align it with the new naming scheme. Still, `convert_image_dtype` is
# prevalent and well understood. Thus, we just alias it without deprecating the old name.
convert_image_dtype = convert_dtype_image_tensor


def convert_dtype_video(video: torch.Tensor, dtype: torch.dtype = torch.float) -> torch.Tensor:
    return convert_dtype_image_tensor(video, dtype)


def convert_dtype(
    inpt: Union[datapoints._ImageTypeJIT, datapoints._VideoTypeJIT], dtype: torch.dtype = torch.float
) -> torch.Tensor:
    if not torch.jit.is_scripting():
        _log_api_usage_once(convert_dtype)

    if torch.jit.is_scripting() or is_simple_tensor(inpt):
        return convert_dtype_image_tensor(inpt, dtype)
    elif isinstance(inpt, datapoints.Image):
        output = convert_dtype_image_tensor(inpt.as_subclass(torch.Tensor), dtype)
        return datapoints.Image.wrap_like(inpt, output)
    elif isinstance(inpt, datapoints.Video):
        output = convert_dtype_video(inpt.as_subclass(torch.Tensor), dtype)
        return datapoints.Video.wrap_like(inpt, output)
    else:
        raise TypeError(
            f"Input can either be a plain tensor or an `Image` or `Video` datapoint, " f"but got {type(inpt)} instead."
        )
