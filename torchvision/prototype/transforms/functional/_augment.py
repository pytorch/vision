from typing import Any

import torch
from torchvision.prototype import features
from torchvision.prototype.transforms import kernels as K
from torchvision.transforms import functional as _F

from ._utils import dispatch


@dispatch(
    {
        torch.Tensor: _F.erase,
        features.Image: K.erase_image,
    }
)
def erase(input: Any, *args: Any, **kwargs: Any) -> Any:
    """TODO: add docstring"""
    ...


@dispatch(
    {
        features.Image: K.mixup_image,
        features.OneHotLabel: K.mixup_one_hot_label,
    }
)
def mixup(input: Any, *args: Any, **kwargs: Any) -> Any:
    """TODO: add docstring"""
    ...


@dispatch(
    {
        features.Image: None,
        features.OneHotLabel: None,
    }
)
def cutmix(input: Any, *args: Any, **kwargs: Any) -> Any:
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
    if isinstance(input, features.Image):
        kwargs.pop("lam_adjusted", None)
        output = K.cutmix_image(input, **kwargs)
        return features.Image.new_like(input, output)
    elif isinstance(input, features.OneHotLabel):
        kwargs.pop("box", None)
        output = K.cutmix_one_hot_label(input, **kwargs)
        return features.OneHotLabel.new_like(input, output)

    raise RuntimeError
