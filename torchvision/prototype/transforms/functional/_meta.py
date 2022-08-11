from typing import Optional, Tuple

import PIL.Image
import torch
from torchvision.prototype.constants import ColorSpace
from torchvision.transforms import functional_pil as _FP, functional_tensor as _FT

get_dimensions_image_tensor = _FT.get_dimensions
get_dimensions_image_pil = _FP.get_dimensions


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
