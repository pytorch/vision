import warnings
from typing import Any, List, Union

import PIL.Image
import torch

from torchvision.prototype import features
from torchvision.transforms import functional as _F


@torch.jit.unused
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


def rgb_to_grayscale(
    inpt: Union[features.LegacyImageTypeJIT, features.LegacyVideoTypeJIT], num_output_channels: int = 1
) -> Union[features.LegacyImageTypeJIT, features.LegacyVideoTypeJIT]:
    old_color_space = (
        features._image._from_tensor_shape(inpt.shape)  # type: ignore[arg-type]
        if isinstance(inpt, torch.Tensor)
        and (torch.jit.is_scripting() or not isinstance(inpt, (features.Image, features.Video)))
        else None
    )

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


@torch.jit.unused
def to_tensor(inpt: Any) -> torch.Tensor:
    warnings.warn(
        "The function `to_tensor(...)` is deprecated and will be removed in a future release. "
        "Instead, please use `to_image_tensor(...)` followed by `convert_image_dtype(...)`."
    )
    return _F.to_tensor(inpt)


def get_image_size(inpt: features.ImageOrVideoTypeJIT) -> List[int]:
    warnings.warn(
        "The function `get_image_size(...)` is deprecated and will be removed in a future release. "
        "Instead, please use `get_spatial_size(...)` which returns `[h, w]` instead of `[w, h]`."
    )
    return _F.get_image_size(inpt)
