from typing import Any

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


@dispatch(
    {
        torch.Tensor: _F.normalize,
        features.Image: K.normalize_image,
    }
)
def normalize(input: Any, *args: Any, **kwargs: Any) -> Any:
    """TODO: add docstring"""
    ...


@dispatch(
    {
        torch.Tensor: _F.gaussian_blur,
        PIL.Image.Image: _F.gaussian_blur,
        features.Image: K.gaussian_blur_image,
    }
)
def gaussian_blur(input: Any, *args: Any, **kwargs: Any) -> Any:
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
def get_image_size(input: Any, *args: Any, **kwargs: Any) -> Any:
    if isinstance(input, (features.Image, features.BoundingBox)):
        return list(input.image_size)

    raise RuntimeError


@dispatch(
    {
        torch.Tensor: _get_image_num_channels_tensor,
        PIL.Image.Image: _get_image_num_channels_pil,
        features.Image: None,
    }
)
def get_image_num_channels(input: Any, *args: Any, **kwargs: Any) -> Any:
    if isinstance(input, features.Image):
        return input.num_channels

    raise RuntimeError
