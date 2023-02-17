import warnings
from typing import Any, List, Union

import PIL.Image
import torch

from torchvision import datapoints
from torchvision.transforms import functional as _F


@torch.jit.unused
def to_grayscale(inpt: PIL.Image.Image, num_output_channels: int = 1) -> PIL.Image.Image:
    call = ", num_output_channels=3" if num_output_channels == 3 else ""
    replacement = "convert_color_space(..., color_space=datapoints.ColorSpace.GRAY)"
    if num_output_channels == 3:
        replacement = f"convert_color_space({replacement}, color_space=datapoints.ColorSpace.RGB)"
    warnings.warn(
        f"The function `to_grayscale(...{call})` is deprecated in will be removed in a future release. "
        f"Instead, please use `{replacement}`.",
    )

    return _F.to_grayscale(inpt, num_output_channels=num_output_channels)


@torch.jit.unused
def to_tensor(inpt: Any) -> torch.Tensor:
    warnings.warn(
        "The function `to_tensor(...)` is deprecated and will be removed in a future release. "
        "Instead, please use `to_image_tensor(...)` followed by `convert_image_dtype(...)`."
    )
    return _F.to_tensor(inpt)


def get_image_size(inpt: Union[datapoints._ImageTypeJIT, datapoints._VideoTypeJIT]) -> List[int]:
    warnings.warn(
        "The function `get_image_size(...)` is deprecated and will be removed in a future release. "
        "Instead, please use `get_spatial_size(...)` which returns `[h, w]` instead of `[w, h]`."
    )
    return _F.get_image_size(inpt)
