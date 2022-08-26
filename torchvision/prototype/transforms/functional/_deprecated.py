import warnings
from typing import Any

import PIL.Image
import torch

from torchvision.prototype import features
from torchvision.transforms import functional as _F


def to_grayscale(inpt: PIL.Image.Image, num_output_channels: int = 1) -> PIL.Image.Image:
    call = ", num_output_channels=3" if num_output_channels == 3 else ""
    replacement = "convert_color_space(..., color_space=features.ColorSpace.GRAY)"
    if num_output_channels == 3:
        replacement = f"convert_color_space({replacement}, color_space=features.ColorSpace.RGB)"
    warnings.warn(
        f"The function `to_grayscale(...{call})` is deprecated in will be removed in a future release. "
        f"Instead, please use `{replacement}`.",
    )

    return _F.to_grayscale(inpt, num_output_channels=num_output_channels)


def rgb_to_grayscale(inpt: Any, num_output_channels: int = 1) -> Any:
    old_color_space = features.Image.guess_color_space(inpt) if features.is_simple_tensor(inpt) else None

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

    return _F.rgb_to_grayscale(inpt, num_output_channels=num_output_channels)


def to_tensor(inpt: Any) -> torch.Tensor:
    warnings.warn(
        "The function `to_tensor(...)` is deprecated and will be removed in a future release. "
        "Instead, please use `to_image_tensor(...)` followed by `convert_image_dtype(...)`."
    )
    return _F.to_tensor(inpt)
