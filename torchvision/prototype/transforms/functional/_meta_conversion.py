from typing import TypeVar, Any, cast

import PIL.Image
import torch
from torchvision.ops import box_convert
from torchvision.prototype import features
from torchvision.prototype.transforms import kernels as K
from torchvision.transforms import functional as _F

from ._utils import dispatch

T = TypeVar("T", bound=features._Feature)


@dispatch(
    {
        torch.Tensor: None,
        features.BoundingBox: None,
    }
)
def convert_format(input: T, *args: Any, **kwargs: Any) -> T:
    format = kwargs["format"]
    if type(input) is torch.Tensor:
        old_format = kwargs.get("old_format")
        if old_format is None:
            raise TypeError("For vanilla tensors the `old_format` needs to be provided.")
        return box_convert(input, in_fmt=kwargs["old_format"].name.lower(), out_fmt=format.name.lower())  # type: ignore[return-value]
    elif isinstance(input, features.BoundingBox):
        output = K.convert_bounding_box_format(input, old_format=input.format, new_format=kwargs["format"])
        return cast(T, features.BoundingBox.new_like(input, output, format=format))

    raise RuntimeError


@dispatch(
    {
        torch.Tensor: None,
        PIL.Image.Image: None,
        features.Image: None,
    }
)
def convert_color_space(input: T, *args: Any, **kwargs: Any) -> T:
    color_space = kwargs["color_space"]
    if type(input) is torch.Tensor or isinstance(input, PIL.Image.Image):
        if color_space != features.ColorSpace.GRAYSCALE:
            raise ValueError("For vanilla tensors and PIL images only RGB to grayscale is supported")
        return _F.rgb_to_grayscale(input)  # type: ignore[return-value]
    elif isinstance(input, features.Image):
        output = K.convert_color_space(input, old_color_space=input.color_space, new_color_space=color_space)
        return cast(T, features.Image.new_like(input, output, color_space=color_space))

    raise RuntimeError
