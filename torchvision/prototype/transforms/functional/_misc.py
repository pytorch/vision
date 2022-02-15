from typing import TypeVar, Any, cast, List

import PIL.Image
import torch
from torchvision.prototype import features
from torchvision.prototype.transforms import kernels as K
from torchvision.transforms import functional as _F
from torchvision.transforms.functional_pil import (
    get_image_size as _get_image_size_pil,
    get_image_num_channels as _get_image_num_channels_pil,
)
from torchvision.transforms.functional_tensor import (
    get_image_size as _get_image_size_tensor,
    get_image_num_channels as _get_image_num_channels_tensor,
)

from ._utils import dispatch

T = TypeVar("T", bound=features._Feature)


@dispatch(
    {
        torch.Tensor: _F.normalize,
        features.Image: K.normalize_image,
    }
)
def normalize(input: T, *args: Any, **kwargs: Any) -> T:
    """TODO: add docstring"""
    ...


@dispatch(
    {
        torch.Tensor: _F.gaussian_blur,
        PIL.Image.Image: _F.gaussian_blur,
        features.Image: K.gaussian_blur_image,
    }
)
def gaussian_blur(input: T, *args: Any, **kwargs: Any) -> T:
    """TODO: add docstring"""
    ...


@dispatch(
    {
        torch.Tensor: _get_image_size_tensor,
        PIL.Image.Image: _get_image_size_pil,
        features.Image: None,
        features.BoundingBox: None,
    }
)
def get_image_size(input: T, *args: Any, **kwargs: Any) -> T:
    if isinstance(input, (features.Image, features.BoundingBox)):
        return cast(List[int], list(input.image_size))  # type: ignore[return-value]

    raise RuntimeError


@dispatch(
    {
        torch.Tensor: _get_image_num_channels_tensor,
        PIL.Image.Image: _get_image_num_channels_pil,
        features.Image: None,
    }
)
def get_image_num_channels(input: T, *args: Any, **kwargs: Any) -> T:
    if isinstance(input, features.Image):
        return input.num_channels  # type: ignore[return-value]

    raise RuntimeError
