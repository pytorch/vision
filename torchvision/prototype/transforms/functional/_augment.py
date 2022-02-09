from typing import Tuple
from typing import TypeVar

import torch
from torchvision.prototype import features
from torchvision.prototype.transforms import kernels as K

from ._utils import dispatch

T = TypeVar("T", bound=features.Feature)


@dispatch(
    {
        features.Image: K.erase_image,
    },
)
def erase(input: T, *, i: int, j: int, h: int, w: int, v: torch.Tensor, inplace: bool) -> T:
    """ADDME"""
    pass


@dispatch(
    {
        features.Image: K.mixup_image,
        features.OneHotLabel: K.mixup_one_hot_label,
    },
)
def mixup(input: T, *, lam: float, inplace: bool) -> T:
    """ADDME"""
    pass


@dispatch(
    {
        features.Image: K.cutmix_image,
        features.OneHotLabel: K.cutmix_one_hot_label,
    },
)
def cutmix(input: T, *, box: Tuple[int, int, int, int], lam_adjusted: float, inplace: bool) -> T:
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

    .. note::

        The ``box`` parameter is only required for inputs of type

        - :class:`~torchvision.prototype.features.Image`

    .. note::

        The ``lam_adjusted`` parameter is only required for inputs of type

        - :class:`~torchvision.prototype.features.OneHotLabel`
    """
    pass
