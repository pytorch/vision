from typing import TypeVar, Any

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
