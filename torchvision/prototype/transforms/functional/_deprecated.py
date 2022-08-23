import warnings
from typing import Any

import PIL.Image
import torch

from torchvision.prototype import features

from ._meta import convert_color_space, convert_color_space_image_pil


# TODO: this was copied from `torchvision.prototype.transforms._utils`. Given that this is not related to pytree / the
#  Transform object, we should probably move it to `functional._utils`.
def is_simple_tensor(inpt: Any) -> bool:
    return isinstance(inpt, torch.Tensor) and not isinstance(inpt, features._Feature)


def to_grayscale(inpt: PIL.Image.Image, num_output_channels: int = 1) -> PIL.Image.Image:
    if not isinstance(inpt, PIL.Image.Image):
        raise TypeError("Input should be PIL Image")

    if num_output_channels not in (1, 3):
        raise ValueError("num_output_channels should be either 1 or 3")

    call = ", num_output_channels=3" if num_output_channels == 3 else ""
    replacement = "convert_color_space(..., color_space=features.ColorSpace.GRAY)"
    if num_output_channels == 3:
        replacement = f"convert_color_space({replacement}, color_space=features.ColorSpace.RGB)"
    warnings.warn(
        f"The function `to_grayscale(...{call})` is deprecated in will be removed in a future release. "
        f"Instead, please use `{replacement}`.",
    )

    # We can't use `convert_color_space_image_pil` since it only supports a subset of the input color spaces that PIL
    # supports, namely `features.ColorSpace`.
    # TODO: Can we even deprecate in favor of `convert_color_space` in such a case? Should we tell the users to call
    #  `image.convert("RGB")` on their own?
    output = inpt.convert("RGB")

    if num_output_channels == 3:
        output = convert_color_space_image_pil(inpt, color_space=features.ColorSpace.RGB)
    return output


def rgb_to_grayscale(inpt: Any, num_output_channels: int = 1) -> Any:
    if num_output_channels not in (1, 3):
        raise ValueError("num_output_channels should be either 1 or 3")

    old_color_space = features.Image.guess_color_space(inpt) if is_simple_tensor(inpt) else None

    call = ", num_output_channels=3" if num_output_channels == 3 else ""
    replacement = (
        f"convert_color_space(..., color_space=features.ColorSpace.GRAY"
        f"{f', old_color_space=features.ColorSpace.{old_color_space}' if old_color_space is not None else ''})"
    )
    if num_output_channels == 3:
        replacement = (
            f"convert_color_space({replacement}, color_space=features.ColorSpace.RGB"
            f"{f', old_color_space=features.ColorSpace.GRAY' if old_color_space is not None else ''})"
        )
    warnings.warn(
        f"The function `rgb_to_grayscale(...{call})` is deprecated in will be removed in a future release. "
        f"Instead, please use `{replacement}`.",
    )

    output = convert_color_space(inpt, color_space=features.ColorSpace.GRAY, old_color_space=old_color_space)
    if num_output_channels == 3:
        output = convert_color_space(
            inpt, color_space=features.ColorSpace.RGB, old_color_space=features.ColorSpace.GRAY
        )
    return output
