from typing import cast, List, Optional, Tuple

import PIL.Image
import torch
from torchvision.prototype import features
from torchvision.prototype.features import BoundingBoxFormat, ColorSpace
from torchvision.transforms import functional_pil as _FP, functional_tensor as _FT


get_dimensions_image_tensor = _FT.get_dimensions
get_dimensions_image_pil = _FP.get_dimensions


def get_dimensions(image: features.ImageOrVideoTypeJIT) -> List[int]:
    if isinstance(image, torch.Tensor) and (
        torch.jit.is_scripting() or not isinstance(image, (features.Image, features.Video))
    ):
        return get_dimensions_image_tensor(image)
    elif isinstance(image, (features.Image, features.Video)):
        channels = image.num_channels
        height, width = image.spatial_size
        return [channels, height, width]
    else:
        return get_dimensions_image_pil(image)


get_num_channels_image_tensor = _FT.get_image_num_channels
get_num_channels_image_pil = _FP.get_image_num_channels


def get_num_channels_video(video: torch.Tensor) -> int:
    return get_num_channels_image_tensor(video)


def get_num_channels(image: features.ImageOrVideoTypeJIT) -> int:
    if isinstance(image, torch.Tensor) and (
        torch.jit.is_scripting() or not isinstance(image, (features.Image, features.Video))
    ):
        return _FT.get_image_num_channels(image)
    elif isinstance(image, (features.Image, features.Video)):
        return image.num_channels
    else:
        return _FP.get_image_num_channels(image)


# We changed the names to ensure it can be used not only for images but also videos. Thus, we just alias it without
# deprecating the old names.
get_image_num_channels = get_num_channels


def get_spatial_size_image_tensor(image: torch.Tensor) -> List[int]:
    width, height = _FT.get_image_size(image)
    return [height, width]


@torch.jit.unused
def get_spatial_size_image_pil(image: PIL.Image.Image) -> List[int]:
    width, height = _FP.get_image_size(image)
    return [height, width]


def get_spatial_size_video(video: torch.Tensor) -> List[int]:
    return get_spatial_size_image_tensor(video)


def get_spatial_size_mask(mask: torch.Tensor) -> List[int]:
    return get_spatial_size_image_tensor(mask)


@torch.jit.unused
def get_spatial_size_bounding_box(bounding_box: features.BoundingBox) -> List[int]:
    return list(bounding_box.spatial_size)


def get_spatial_size(inpt: features.InputTypeJIT) -> List[int]:
    if isinstance(inpt, torch.Tensor) and (torch.jit.is_scripting() or not isinstance(inpt, features._Feature)):
        return get_spatial_size_image_tensor(inpt)
    elif isinstance(inpt, (features.Image, features.Video, features.BoundingBox, features.Mask)):
        return list(inpt.spatial_size)
    else:
        return get_spatial_size_image_pil(inpt)  # type: ignore[no-any-return]


def get_num_frames_video(video: torch.Tensor) -> int:
    return video.shape[-4]


def get_num_frames(inpt: features.VideoTypeJIT) -> int:
    if isinstance(inpt, torch.Tensor) and (torch.jit.is_scripting() or not isinstance(inpt, features.Video)):
        return get_num_frames_video(inpt)
    elif isinstance(inpt, features.Video):
        return inpt.num_frames
    else:
        raise TypeError(f"The video should be a Tensor. Got {type(inpt)}")


def _xywh_to_xyxy(xywh: torch.Tensor) -> torch.Tensor:
    xyxy = xywh.clone()
    xyxy[..., 2:] += xyxy[..., :2]
    return xyxy


def _xyxy_to_xywh(xyxy: torch.Tensor) -> torch.Tensor:
    xywh = xyxy.clone()
    xywh[..., 2:] -= xywh[..., :2]
    return xywh


def _cxcywh_to_xyxy(cxcywh: torch.Tensor) -> torch.Tensor:
    cx, cy, w, h = torch.unbind(cxcywh, dim=-1)
    x1 = cx - 0.5 * w
    y1 = cy - 0.5 * h
    x2 = cx + 0.5 * w
    y2 = cy + 0.5 * h
    return torch.stack((x1, y1, x2, y2), dim=-1).to(cxcywh.dtype)


def _xyxy_to_cxcywh(xyxy: torch.Tensor) -> torch.Tensor:
    x1, y1, x2, y2 = torch.unbind(xyxy, dim=-1)
    cx = (x1 + x2) / 2
    cy = (y1 + y2) / 2
    w = x2 - x1
    h = y2 - y1
    return torch.stack((cx, cy, w, h), dim=-1).to(xyxy.dtype)


def convert_format_bounding_box(
    bounding_box: torch.Tensor, old_format: BoundingBoxFormat, new_format: BoundingBoxFormat, copy: bool = True
) -> torch.Tensor:
    if new_format == old_format:
        if copy:
            return bounding_box.clone()
        else:
            return bounding_box

    if old_format == BoundingBoxFormat.XYWH:
        bounding_box = _xywh_to_xyxy(bounding_box)
    elif old_format == BoundingBoxFormat.CXCYWH:
        bounding_box = _cxcywh_to_xyxy(bounding_box)

    if new_format == BoundingBoxFormat.XYWH:
        bounding_box = _xyxy_to_xywh(bounding_box)
    elif new_format == BoundingBoxFormat.CXCYWH:
        bounding_box = _xyxy_to_cxcywh(bounding_box)

    return bounding_box


def clamp_bounding_box(
    bounding_box: torch.Tensor, format: BoundingBoxFormat, spatial_size: Tuple[int, int]
) -> torch.Tensor:
    # TODO: (PERF) Possible speed up clamping if we have different implementations for each bbox format.
    # Not sure if they yield equivalent results.
    xyxy_boxes = convert_format_bounding_box(bounding_box, format, BoundingBoxFormat.XYXY)
    xyxy_boxes[..., 0::2].clamp_(min=0, max=spatial_size[1])
    xyxy_boxes[..., 1::2].clamp_(min=0, max=spatial_size[0])
    return convert_format_bounding_box(xyxy_boxes, BoundingBoxFormat.XYXY, format, copy=False)


def _split_alpha(image: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    return image[..., :-1, :, :], image[..., -1:, :, :]


def _strip_alpha(image: torch.Tensor) -> torch.Tensor:
    image, alpha = _split_alpha(image)
    if not torch.all(alpha == _FT._max_value(alpha.dtype)):
        raise RuntimeError(
            "Stripping the alpha channel if it contains values other than the max value is not supported."
        )
    return image


def _add_alpha(image: torch.Tensor, alpha: Optional[torch.Tensor] = None) -> torch.Tensor:
    if alpha is None:
        shape = list(image.shape)
        shape[-3] = 1
        alpha = torch.full(shape, _FT._max_value(image.dtype), dtype=image.dtype, device=image.device)
    return torch.cat((image, alpha), dim=-3)


def _gray_to_rgb(grayscale: torch.Tensor) -> torch.Tensor:
    repeats = [1] * grayscale.ndim
    repeats[-3] = 3
    return grayscale.repeat(repeats)


_rgb_to_gray = _FT.rgb_to_grayscale


def convert_color_space_image_tensor(
    image: torch.Tensor, old_color_space: ColorSpace, new_color_space: ColorSpace, copy: bool = True
) -> torch.Tensor:
    if new_color_space == old_color_space:
        if copy:
            return image.clone()
        else:
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
        image, alpha = _split_alpha(image)
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
        image, alpha = _split_alpha(image)
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
def convert_color_space_image_pil(
    image: PIL.Image.Image, color_space: ColorSpace, copy: bool = True
) -> PIL.Image.Image:
    old_mode = image.mode
    try:
        new_mode = _COLOR_SPACE_TO_PIL_MODE[color_space]
    except KeyError:
        raise ValueError(f"Conversion from {ColorSpace.from_pil_mode(old_mode)} to {color_space} is not supported.")

    if not copy and image.mode == new_mode:
        return image

    return image.convert(new_mode)


def convert_color_space_video(
    video: torch.Tensor, old_color_space: ColorSpace, new_color_space: ColorSpace, copy: bool = True
) -> torch.Tensor:
    return convert_color_space_image_tensor(
        video, old_color_space=old_color_space, new_color_space=new_color_space, copy=copy
    )


def convert_color_space(
    inpt: features.ImageOrVideoTypeJIT,
    color_space: ColorSpace,
    old_color_space: Optional[ColorSpace] = None,
    copy: bool = True,
) -> features.ImageOrVideoTypeJIT:
    if isinstance(inpt, torch.Tensor) and (
        torch.jit.is_scripting() or not isinstance(inpt, (features.Image, features.Video))
    ):
        if old_color_space is None:
            raise RuntimeError(
                "In order to convert the color space of simple tensors, "
                "the `old_color_space=...` parameter needs to be passed."
            )
        return convert_color_space_image_tensor(
            inpt, old_color_space=old_color_space, new_color_space=color_space, copy=copy
        )
    elif isinstance(inpt, (features.Image, features.Video)):
        return inpt.to_color_space(color_space, copy=copy)
    else:
        return cast(features.ImageOrVideoTypeJIT, convert_color_space_image_pil(inpt, color_space, copy=copy))
