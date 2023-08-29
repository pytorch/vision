import torch

_TORCHFUNCTION_SUBCLASS = False


class _ReturnTypeCM:
    def __init__(self, to_restore):
        self.to_restore = to_restore

    def __enter__(self):
        return self

    def __exit__(self, *args):
        global _TORCHFUNCTION_SUBCLASS
        _TORCHFUNCTION_SUBCLASS = self.to_restore


def set_return_type(return_type: str):
    """[BETA] Set the return type of torch operations on vision_tensors.

    This only affects the behaviour of torch operations. It has no effect on
    ``torchvision`` transforms or functionals, which will always return as
    output the same type that was passed as input.

    .. warning::

        We recommend using :class:`~torchvision.transforms.v2.ToPureTensor` at
        the end of your transform pipelines if you use
        ``set_return_type("dataptoint")``. This will avoid the
        ``__torch_function__`` overhead in the models ``forward()``.

    Can be used as a global flag for the entire program:

    .. code:: python

        img = vision_tensors.Image(torch.rand(3, 5, 5))
        img + 2  # This is a pure Tensor (default behaviour)

        set_return_type("vision_tensors")
        img + 2  # This is an Image

    or as a context manager to restrict the scope:

    .. code:: python

        img = vision_tensors.Image(torch.rand(3, 5, 5))
        img + 2  # This is a pure Tensor
        with set_return_type("vision_tensors"):
            img + 2  # This is an Image
        img + 2  # This is a pure Tensor

    Args:
        return_type (str): Can be "vision_tensor" or "tensor". Default is "tensor".
    """
    global _TORCHFUNCTION_SUBCLASS
    to_restore = _TORCHFUNCTION_SUBCLASS
    _TORCHFUNCTION_SUBCLASS = {"tensor": False, "vision_tensor": True}[return_type.lower()]

    return _ReturnTypeCM(to_restore)


def _must_return_subclass():
    return _TORCHFUNCTION_SUBCLASS


# For those ops we always want to preserve the original subclass instead of returning a pure Tensor
_FORCE_TORCHFUNCTION_SUBCLASS = {torch.Tensor.clone, torch.Tensor.to, torch.Tensor.detach, torch.Tensor.requires_grad_}
