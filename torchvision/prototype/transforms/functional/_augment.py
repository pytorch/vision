from typing import TypeVar, Any

import torch
from torchvision.prototype import features
from torchvision.prototype.transforms import kernels as K
from torchvision.transforms import functional as _F

from ._utils import dispatch

T = TypeVar("T", bound=features.Feature)


@dispatch(
    {
        torch.Tensor: _F.erase,
        features.Image: K.erase_image,
    }
)
def erase(input: T, *args: Any, **kwargs: Any) -> T:
    """ADDME"""
    ...


@dispatch(
    {
        features.Image: K.mixup_image,
        features.OneHotLabel: K.mixup_one_hot_label,
    }
)
def mixup(input: T, *args: Any, **kwargs: Any) -> T:
    """ADDME"""
    ...


@dispatch(
    {
        features.Image: K.cutmix_image,
        features.OneHotLabel: K.cutmix_one_hot_label,
    }
)
def cutmix(input: T, *args: Any, **kwargs: Any) -> T:
    """Perform the CutMix operation as introduced in the paper
    `"CutMix: Regularization Strategy to Train Strong Classifiers with Localizable Features" <https://arxiv.org/abs/1905.04899>`_.

    Dispatch to the corresponding kernels happens according to this table:

    .. table::
       :widths: 30 70

       ====================================================  ================================================================
       :class:`~torchvision.prototype.features.Image`        :func:`~torch.prototype.transforms.kernels.cutmix_image`
       :class:`~torchvision.prototype.features.OneHotLabel`  :func:`~torch.prototype.transforms.kernels.cutmix_one_hot_label`
       ====================================================  ================================================================

    Please refer to the kernel documentations for a detailed explanation of the functionality and parameters.
    """
    ...
