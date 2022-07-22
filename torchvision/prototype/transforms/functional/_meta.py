from typing import Optional, Tuple

import PIL.Image
import torch
from torchvision.prototype.features import BoundingBoxFormat, ColorSpace
from torchvision.transforms import functional_pil as _FP, functional_tensor as _FT

get_dimensions_image_tensor = _FT.get_dimensions
get_dimensions_image_pil = _FP.get_dimensions


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
    return torch.stack((x1, y1, x2, y2), dim=-1)


def _xyxy_to_cxcywh(xyxy: torch.Tensor) -> torch.Tensor:
    x1, y1, x2, y2 = torch.unbind(xyxy, dim=-1)
    cx = (x1 + x2) / 2
    cy = (y1 + y2) / 2
    w = x2 - x1
    h = y2 - y1
    return torch.stack((cx, cy, w, h), dim=-1)


def convert_bounding_box_format(
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


def convert_image_color_space_tensor(
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


def convert_image_color_space_pil(
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
