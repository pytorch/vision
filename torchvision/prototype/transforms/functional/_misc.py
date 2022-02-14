from typing import TypeVar, Any

import PIL.Image
import torch
from torchvision.prototype import features
from torchvision.prototype.transforms import kernels as K
from torchvision.transforms import functional as _F

from ._utils import dispatch

T = TypeVar("T", bound=features._Feature)


@dispatch(
    {
        torch.Tensor: _F.normalize,
        features.Image: K.normalize_image,
    }
)
def normalize(input: T, *args: Any, **kwargs: Any) -> T:
    """ADDME"""
    ...


@dispatch(
    {
        torch.Tensor: _F.gaussian_blur,
        PIL.Image.Image: _F.gaussian_blur,
        features.Image: K.gaussian_blur_image,
    }
)
def ten_gaussian_blur(input: T, *args: Any, **kwargs: Any) -> T:
    """ADDME"""
    ...
