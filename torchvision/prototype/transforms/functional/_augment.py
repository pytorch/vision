from typing import Tuple
from typing import TypeVar

import torch
from torchvision.prototype import features
from torchvision.prototype.transforms import kernels as K

from ._utils import dispatch, FEATURE_SPECIFIC_PARAM

T = TypeVar("T", bound=features.Feature)


@dispatch(
    {
        features.Image: K.erase_image,
    },
)
def erase(input: T, *, i: int, j: int, h: int, w: int, v: torch.Tensor, inplace: bool = False) -> T:
    """ADDME"""
    pass


@dispatch(
    {
        features.Image: K.mixup_image,
        features.OneHotLabel: K.mixup_one_hot_label,
    },
)
def mixup(input: T, *, lam: float, inplace: bool = False) -> T:
    """ADDME"""
    pass


@dispatch(
    {
        features.Image: K.cutmix_image,
        features.OneHotLabel: K.cutmix_one_hot_label,
    },
)
def cutmix(
    input: T,
    *,
    box: Tuple[int, int, int, int] = FEATURE_SPECIFIC_PARAM,  # type: ignore[assignment]
    lam_adjusted: float = FEATURE_SPECIFIC_PARAM,  # type: ignore[assignment]
    inplace: bool = False,
) -> T:
    """ADDME"""
    pass
