import torch

_TORCHFUNCTION_SUBCLASS = False


def set_return_type(type="Tensor"):
    global _TORCHFUNCTION_SUBCLASS
    _TORCHFUNCTION_SUBCLASS = {"tensor": False, "datapoint": True}[type.lower()]


def _must_return_subclass():
    return _TORCHFUNCTION_SUBCLASS


# For those ops we always want to preserve the original subclass instead of returning a pure Tensor
_FORCE_TORCHFUNCTION_SUBCLASS = {torch.Tensor.clone, torch.Tensor.to, torch.Tensor.detach, torch.Tensor.requires_grad_}
