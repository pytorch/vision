from typing import List, Optional, Tuple, Union

import PIL.Image
import torch
from torchvision.prototype import datapoints
from torchvision.prototype.datapoints import BoundingBoxFormat, ColorSpace
from torchvision.transforms import functional_pil as _FP
from torchvision.transforms.functional_tensor import _max_value

from torchvision.utils import _log_api_usage_once


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


def get_dimensions(inpt: Union[datapoints.ImageTypeJIT, datapoints.VideoTypeJIT]) -> List[int]:
    if not torch.jit.is_scripting():
        _log_api_usage_once(get_dimensions)

    if isinstance(inpt, torch.Tensor) and (
        torch.jit.is_scripting() or not isinstance(inpt, (datapoints.Image, datapoints.Video))
    ):
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


def get_num_channels(inpt: Union[datapoints.ImageTypeJIT, datapoints.VideoTypeJIT]) -> int:
    if not torch.jit.is_scripting():
        _log_api_usage_once(get_num_channels)

    if isinstance(inpt, torch.Tensor) and (
        torch.jit.is_scripting() or not isinstance(inpt, (datapoints.Image, datapoints.Video))
    ):
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


def get_spatial_size(inpt: datapoints.InputTypeJIT) -> List[int]:
    if not torch.jit.is_scripting():
        _log_api_usage_once(get_spatial_size)

    if isinstance(inpt, torch.Tensor) and (
        torch.jit.is_scripting() or not isinstance(inpt, datapoints._datapoint.Datapoint)
    ):
        return get_spatial_size_image_tensor(inpt)
    elif isinstance(inpt, (datapoints.Image, datapoints.Video, datapoints.BoundingBox, datapoints.Mask)):
        return list(inpt.spatial_size)
    elif isinstance(inpt, PIL.Image.Image):
        return get_spatial_size_image_pil(inpt)  # type: ignore[no-any-return]
    else:
        raise TypeError(
            f"Input can either be a plain tensor, any TorchVision datapoint, or a PIL image, "
            f"but got {type(inpt)} instead."
        )


def get_num_frames_video(video: torch.Tensor) -> int:
    return video.shape[-4]


def get_num_frames(inpt: datapoints.VideoTypeJIT) -> int:
    if not torch.jit.is_scripting():
        _log_api_usage_once(get_num_frames)

    if isinstance(inpt, torch.Tensor) and (torch.jit.is_scripting() or not isinstance(inpt, datapoints.Video)):
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


def convert_format_bounding_box(
    bounding_box: torch.Tensor, old_format: BoundingBoxFormat, new_format: BoundingBoxFormat, inplace: bool = False
) -> torch.Tensor:
    if not torch.jit.is_scripting():
        _log_api_usage_once(convert_format_bounding_box)

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


def clamp_bounding_box(
    bounding_box: torch.Tensor, format: BoundingBoxFormat, spatial_size: Tuple[int, int]
) -> torch.Tensor:
    if not torch.jit.is_scripting():
        _log_api_usage_once(clamp_bounding_box)

    # TODO: Investigate if it makes sense from a performance perspective to have an implementation for every
    #  BoundingBoxFormat instead of converting back and forth
    xyxy_boxes = convert_format_bounding_box(
        bounding_box.clone(), old_format=format, new_format=datapoints.BoundingBoxFormat.XYXY, inplace=True
    )
    xyxy_boxes[..., 0::2].clamp_(min=0, max=spatial_size[1])
    xyxy_boxes[..., 1::2].clamp_(min=0, max=spatial_size[0])
    return convert_format_bounding_box(xyxy_boxes, old_format=BoundingBoxFormat.XYXY, new_format=format, inplace=True)


def _strip_alpha(image: torch.Tensor) -> torch.Tensor:
    image, alpha = torch.tensor_split(image, indices=(-1,), dim=-3)
    if not torch.all(alpha == _max_value(alpha.dtype)):
        raise RuntimeError(
            "Stripping the alpha channel if it contains values other than the max value is not supported."
        )
    return image


def _add_alpha(image: torch.Tensor, alpha: Optional[torch.Tensor] = None) -> torch.Tensor:
    if alpha is None:
        shape = list(image.shape)
        shape[-3] = 1
        alpha = torch.full(shape, _max_value(image.dtype), dtype=image.dtype, device=image.device)
    return torch.cat((image, alpha), dim=-3)


def _gray_to_rgb(grayscale: torch.Tensor) -> torch.Tensor:
    repeats = [1] * grayscale.ndim
    repeats[-3] = 3
    return grayscale.repeat(repeats)


def _rgb_to_gray(image: torch.Tensor, cast: bool = True) -> torch.Tensor:
    r, g, b = image.unbind(dim=-3)
    l_img = r.mul(0.2989).add_(g, alpha=0.587).add_(b, alpha=0.114)
    if cast:
        l_img = l_img.to(image.dtype)
    l_img = l_img.unsqueeze(dim=-3)
    return l_img


def convert_color_space_image_tensor(
    image: torch.Tensor, old_color_space: ColorSpace, new_color_space: ColorSpace
) -> torch.Tensor:
    if new_color_space == old_color_space:
        return image

    if old_color_space == ColorSpace.OTHER or new_color_space == ColorSpace.OTHER:
        raise RuntimeError(f"Conversion to or from {ColorSpace.OTHER} is not supported.")

    if old_color_space == ColorSpace.GRAY and new_color_space == ColorSpace.GRAY_ALPHA:
        return _add_alpha(image)
    elif old_color_space == ColorSpace.GRAY and new_color_space == ColorSpace.RGB:
        return _gray_to_rgb(image)
    elif old_color_space == ColorSpace.GRAY and new_color_space == ColorSpace.RGB_ALPHA:
        return _add_alpha(_gray_to_rgb(image))
    elif old_color_space == ColorSpace.GRAY_ALPHA and new_color_space == ColorSpace.GRAY:
        return _strip_alpha(image)
    elif old_color_space == ColorSpace.GRAY_ALPHA and new_color_space == ColorSpace.RGB:
        return _gray_to_rgb(_strip_alpha(image))
    elif old_color_space == ColorSpace.GRAY_ALPHA and new_color_space == ColorSpace.RGB_ALPHA:
        image, alpha = torch.tensor_split(image, indices=(-1,), dim=-3)
        return _add_alpha(_gray_to_rgb(image), alpha)
    elif old_color_space == ColorSpace.RGB and new_color_space == ColorSpace.GRAY:
        return _rgb_to_gray(image)
    elif old_color_space == ColorSpace.RGB and new_color_space == ColorSpace.GRAY_ALPHA:
        return _add_alpha(_rgb_to_gray(image))
    elif old_color_space == ColorSpace.RGB and new_color_space == ColorSpace.RGB_ALPHA:
        return _add_alpha(image)
    elif old_color_space == ColorSpace.RGB_ALPHA and new_color_space == ColorSpace.GRAY:
        return _rgb_to_gray(_strip_alpha(image))
    elif old_color_space == ColorSpace.RGB_ALPHA and new_color_space == ColorSpace.GRAY_ALPHA:
        image, alpha = torch.tensor_split(image, indices=(-1,), dim=-3)
        return _add_alpha(_rgb_to_gray(image), alpha)
    elif old_color_space == ColorSpace.RGB_ALPHA and new_color_space == ColorSpace.RGB:
        return _strip_alpha(image)
    else:
        raise RuntimeError(f"Conversion from {old_color_space} to {new_color_space} is not supported.")


_COLOR_SPACE_TO_PIL_MODE = {
    ColorSpace.GRAY: "L",
    ColorSpace.GRAY_ALPHA: "LA",
    ColorSpace.RGB: "RGB",
    ColorSpace.RGB_ALPHA: "RGBA",
}


@torch.jit.unused
def convert_color_space_image_pil(image: PIL.Image.Image, color_space: ColorSpace) -> PIL.Image.Image:
    old_mode = image.mode
    try:
        new_mode = _COLOR_SPACE_TO_PIL_MODE[color_space]
    except KeyError:
        raise ValueError(f"Conversion from {ColorSpace.from_pil_mode(old_mode)} to {color_space} is not supported.")

    if image.mode == new_mode:
        return image

    return image.convert(new_mode)


def convert_color_space_video(
    video: torch.Tensor, old_color_space: ColorSpace, new_color_space: ColorSpace
) -> torch.Tensor:
    return convert_color_space_image_tensor(video, old_color_space=old_color_space, new_color_space=new_color_space)


def convert_color_space(
    inpt: Union[datapoints.ImageTypeJIT, datapoints.VideoTypeJIT],
    color_space: ColorSpace,
    old_color_space: Optional[ColorSpace] = None,
) -> Union[datapoints.ImageTypeJIT, datapoints.VideoTypeJIT]:
    if not torch.jit.is_scripting():
        _log_api_usage_once(convert_color_space)

    if isinstance(inpt, torch.Tensor) and (
        torch.jit.is_scripting() or not isinstance(inpt, (datapoints.Image, datapoints.Video))
    ):
        if old_color_space is None:
            raise RuntimeError(
                "In order to convert the color space of simple tensors, "
                "the `old_color_space=...` parameter needs to be passed."
            )
        return convert_color_space_image_tensor(inpt, old_color_space=old_color_space, new_color_space=color_space)
    elif isinstance(inpt, datapoints.Image):
        output = convert_color_space_image_tensor(
            inpt.as_subclass(torch.Tensor), old_color_space=inpt.color_space, new_color_space=color_space
        )
        return datapoints.Image.wrap_like(inpt, output, color_space=color_space)
    elif isinstance(inpt, datapoints.Video):
        output = convert_color_space_video(
            inpt.as_subclass(torch.Tensor), old_color_space=inpt.color_space, new_color_space=color_space
        )
        return datapoints.Video.wrap_like(inpt, output, color_space=color_space)
    elif isinstance(inpt, PIL.Image.Image):
        return convert_color_space_image_pil(inpt, color_space=color_space)
    else:
        raise TypeError(
            f"Input can either be a plain tensor, an `Image` or `Video` datapoint, or a PIL image, "
            f"but got {type(inpt)} instead."
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
    inpt: Union[datapoints.ImageTypeJIT, datapoints.VideoTypeJIT], dtype: torch.dtype = torch.float
) -> torch.Tensor:
    if not torch.jit.is_scripting():
        _log_api_usage_once(convert_dtype)

    if isinstance(inpt, torch.Tensor) and (
        torch.jit.is_scripting() or not isinstance(inpt, (datapoints.Image, datapoints.Video))
    ):
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
