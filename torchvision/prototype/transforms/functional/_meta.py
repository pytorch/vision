import PIL.Image
import torch
from torchvision.prototype.features import BoundingBoxFormat, ColorSpace
from torchvision.transforms import functional_tensor as _FT, functional_pil as _FP


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
    bounding_box: torch.Tensor, *, old_format: BoundingBoxFormat, new_format: BoundingBoxFormat
) -> torch.Tensor:
    if new_format == old_format:
        return bounding_box.clone()

    if old_format == BoundingBoxFormat.XYWH:
        bounding_box = _xywh_to_xyxy(bounding_box)
    elif old_format == BoundingBoxFormat.CXCYWH:
        bounding_box = _cxcywh_to_xyxy(bounding_box)

    if new_format == BoundingBoxFormat.XYWH:
        bounding_box = _xyxy_to_xywh(bounding_box)
    elif new_format == BoundingBoxFormat.CXCYWH:
        bounding_box = _xyxy_to_cxcywh(bounding_box)

    return bounding_box


def _strip_alpha(image: torch.Tensor) -> torch.Tensor:
    alpha = image[..., -1, :, :]
    if not torch.all(alpha == _FT._max_value(image.dtype)):
        raise RuntimeError(
            "Stripping the alpha channel if it contains values other than the max value is not supported."
        )
    return image[..., :-1, :, :]


def _add_alpha(image: torch.Tensor) -> torch.Tensor:
    shape = list(image.shape)
    shape[-3] = 1
    alpha = torch.full(shape, _FT._max_value(image.dtype), dtype=image.dtype, device=image.device)
    return torch.cat((image, alpha), dim=-3)


def _grayscale_alpha_to_grayscale_tensor(grayscale_alpha: torch.Tensor) -> torch.Tensor:
    return _strip_alpha(grayscale_alpha)


def _grayscale_to_grayscale_alpha_tensor(grayscale: torch.Tensor) -> torch.Tensor:
    return _add_alpha(grayscale)


def _grayscale_to_rgb_tensor(grayscale: torch.Tensor) -> torch.Tensor:
    repeats = [1] * grayscale.ndim
    repeats[-3] = 3
    return grayscale.repeat(repeats)


_rgb_to_grayscale_tensor = _FT.rgb_to_grayscale


def _rgba_to_rgb_tensor(rgba: torch.Tensor) -> torch.Tensor:
    return _strip_alpha(rgba)


def _rgb_to_rgba_tensor(rgb: torch.Tensor) -> torch.Tensor:
    return _add_alpha(rgb)


def convert_image_color_space_tensor(
    image: torch.Tensor, old_color_space: ColorSpace, new_color_space: ColorSpace
) -> torch.Tensor:
    if new_color_space == old_color_space:
        return image.clone()

    if old_color_space == ColorSpace.OTHER or new_color_space == ColorSpace.OTHER:
        raise RuntimeError(f"Conversion to or from {ColorSpace.OTHER} is not supported.")

    # This function uses RGB as the intermediate representation for all regular conversions. This means by default
    # every image gets converted into RGB first and subsequently into `new_color_space`. Since the conversion
    # `rgb_to_grayscale(grayscale_to_rgb(image))` is lossy, we need to special case grayscale with alpha channel here.
    if old_color_space == ColorSpace.GRAYSCALE_ALPHA:
        image = _grayscale_alpha_to_grayscale_tensor(image)
        if new_color_space == ColorSpace.GRAYSCALE:
            return image

        old_color_space = ColorSpace.GRAYSCALE

    if old_color_space == ColorSpace.GRAYSCALE:
        image = _grayscale_to_rgb_tensor(image)
    elif old_color_space == ColorSpace.RGBA:
        image = _rgba_to_rgb_tensor(image)

    if new_color_space == ColorSpace.GRAYSCALE:
        image = _rgb_to_grayscale_tensor(image)
    if new_color_space == ColorSpace.GRAYSCALE_ALPHA:
        image = _grayscale_to_grayscale_alpha_tensor(_rgb_to_grayscale_tensor(image))
    elif new_color_space == ColorSpace.RGBA:
        image = _rgb_to_rgba_tensor(image)

    return image


_COLOR_SPACE_TO_PIL_MODE = {
    ColorSpace.GRAYSCALE: "L",
    ColorSpace.GRAYSCALE_ALPHA: "LA",
    ColorSpace.RGB: "RGB",
    ColorSpace.RGBA: "RGBA",
}


def convert_image_color_space_pil(image: PIL.Image.Image, color_space: ColorSpace) -> PIL.Image.Image:
    old_mode = image.mode
    try:
        new_mode = _COLOR_SPACE_TO_PIL_MODE[color_space]
    except KeyError:
        raise ValueError(f"Conversion of {ColorSpace.from_pil_mode(old_mode)} to {color_space} is not supported.")

    return image.convert(new_mode)
