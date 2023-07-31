from typing import List, Optional, Tuple, Union

import PIL.Image
import torch
from torchvision import datapoints
from torchvision.datapoints import BBoxFormat
from torchvision.transforms import _functional_pil as _FP

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
def get_spatial_size_bboxes(bboxes: datapoints.BBoxes) -> List[int]:
    return list(bboxes.spatial_size)


def get_spatial_size(inpt: datapoints._InputTypeJIT) -> List[int]:
    if not torch.jit.is_scripting():
        _log_api_usage_once(get_spatial_size)

    if torch.jit.is_scripting() or is_simple_tensor(inpt):
        return get_spatial_size_image_tensor(inpt)
    elif isinstance(inpt, (datapoints.Image, datapoints.Video, datapoints.BBoxes, datapoints.Mask)):
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


def _convert_format_bboxes(
    bboxes: torch.Tensor, old_format: BBoxFormat, new_format: BBoxFormat, inplace: bool = False
) -> torch.Tensor:

    if new_format == old_format:
        return bboxes

    # TODO: Add _xywh_to_cxcywh and _cxcywh_to_xywh to improve performance
    if old_format == BBoxFormat.XYWH:
        bboxes = _xywh_to_xyxy(bboxes, inplace)
    elif old_format == BBoxFormat.CXCYWH:
        bboxes = _cxcywh_to_xyxy(bboxes, inplace)

    if new_format == BBoxFormat.XYWH:
        bboxes = _xyxy_to_xywh(bboxes, inplace)
    elif new_format == BBoxFormat.CXCYWH:
        bboxes = _xyxy_to_cxcywh(bboxes, inplace)

    return bboxes


def convert_format_bboxes(
    inpt: datapoints._InputTypeJIT,
    old_format: Optional[BBoxFormat] = None,
    new_format: Optional[BBoxFormat] = None,
    inplace: bool = False,
) -> datapoints._InputTypeJIT:
    # This being a kernel / dispatcher hybrid, we need an option to pass `old_format` explicitly for simple tensor
    # inputs as well as extract it from `datapoints.BBoxes` inputs. However, putting a default value on
    # `old_format` means we also need to put one on `new_format` to have syntactically correct Python. Here we mimic the
    # default error that would be thrown if `new_format` had no default value.
    if new_format is None:
        raise TypeError("convert_format_bboxes() missing 1 required argument: 'new_format'")

    if not torch.jit.is_scripting():
        _log_api_usage_once(convert_format_bboxes)

    if torch.jit.is_scripting() or is_simple_tensor(inpt):
        if old_format is None:
            raise ValueError("For simple tensor inputs, `old_format` has to be passed.")
        return _convert_format_bboxes(inpt, old_format=old_format, new_format=new_format, inplace=inplace)
    elif isinstance(inpt, datapoints.BBoxes):
        if old_format is not None:
            raise ValueError("For bounding box datapoint inputs, `old_format` must not be passed.")
        output = _convert_format_bboxes(
            inpt.as_subclass(torch.Tensor), old_format=inpt.format, new_format=new_format, inplace=inplace
        )
        return datapoints.BBoxes.wrap_like(inpt, output, format=new_format)
    else:
        raise TypeError(
            f"Input can either be a plain tensor or a bounding box datapoint, but got {type(inpt)} instead."
        )


def _clamp_bboxes(bboxes: torch.Tensor, format: BBoxFormat, spatial_size: Tuple[int, int]) -> torch.Tensor:
    # TODO: Investigate if it makes sense from a performance perspective to have an implementation for every
    #  BBoxFormat instead of converting back and forth
    in_dtype = bboxes.dtype
    bboxes = bboxes.clone() if bboxes.is_floating_point() else bboxes.float()
    xyxy_boxes = convert_format_bboxes(bboxes, old_format=format, new_format=datapoints.BBoxFormat.XYXY, inplace=True)
    xyxy_boxes[..., 0::2].clamp_(min=0, max=spatial_size[1])
    xyxy_boxes[..., 1::2].clamp_(min=0, max=spatial_size[0])
    out_boxes = convert_format_bboxes(xyxy_boxes, old_format=BBoxFormat.XYXY, new_format=format, inplace=True)
    return out_boxes.to(in_dtype)


def clamp_bboxes(
    inpt: datapoints._InputTypeJIT,
    format: Optional[BBoxFormat] = None,
    spatial_size: Optional[Tuple[int, int]] = None,
) -> datapoints._InputTypeJIT:
    if not torch.jit.is_scripting():
        _log_api_usage_once(clamp_bboxes)

    if torch.jit.is_scripting() or is_simple_tensor(inpt):
        if format is None or spatial_size is None:
            raise ValueError("For simple tensor inputs, `format` and `spatial_size` has to be passed.")
        return _clamp_bboxes(inpt, format=format, spatial_size=spatial_size)
    elif isinstance(inpt, datapoints.BBoxes):
        if format is not None or spatial_size is not None:
            raise ValueError("For bounding box datapoint inputs, `format` and `spatial_size` must not be passed.")
        output = _clamp_bboxes(inpt.as_subclass(torch.Tensor), format=inpt.format, spatial_size=inpt.spatial_size)
        return datapoints.BBoxes.wrap_like(inpt, output)
    else:
        raise TypeError(
            f"Input can either be a plain tensor or a bounding box datapoint, but got {type(inpt)} instead."
        )
