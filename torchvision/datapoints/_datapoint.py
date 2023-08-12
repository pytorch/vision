from __future__ import annotations

from typing import Any, Callable, Dict, Mapping, Optional, Sequence, Tuple, Type, TypeVar, Union

import torch
from torch._C import DisableTorchFunctionSubclass

from torchvision.datapoints._torch_function_helpers import (
    _FORCE_TORCHFUNCTION_SUBCLASS,
    _must_return_subclass,
    _TORCHFUNCTION_SUBCLASS,
)


D = TypeVar("D", bound="Datapoint")


class Datapoint(torch.Tensor):
    """[Beta] Base class for all datapoints.

    You probably don't want to use this class unless you're defining your own
    custom Datapoints. See
    :ref:`sphx_glr_auto_examples_plot_custom_datapoints.py` for details.
    """

    @staticmethod
    def _to_tensor(
        data: Any,
        dtype: Optional[torch.dtype] = None,
        device: Optional[Union[torch.device, str, int]] = None,
        requires_grad: Optional[bool] = None,
    ) -> torch.Tensor:
        if requires_grad is None:
            requires_grad = data.requires_grad if isinstance(data, torch.Tensor) else False
        return torch.as_tensor(data, dtype=dtype, device=device).requires_grad_(requires_grad)

    @classmethod
    def wrap_like(cls: Type[D], other: D, tensor: torch.Tensor) -> D:
        return tensor.as_subclass(cls)

    @classmethod
    def _wrap_output(
        cls,
        output: torch.tensor,
        args: Sequence[Any] = (),
        kwargs: Optional[Mapping[str, Any]] = None,
    ) -> D:
        return torch._tensor._convert(output, cls)

    @classmethod
    def __torch_function__(
        cls,
        func: Callable[..., torch.Tensor],
        types: Tuple[Type[torch.Tensor], ...],
        args: Sequence[Any] = (),
        kwargs: Optional[Mapping[str, Any]] = None,
    ) -> torch.Tensor:
        """For general information about how the __torch_function__ protocol works,
        see https://pytorch.org/docs/stable/notes/extending.html#extending-torch

        TL;DR: Every time a PyTorch operator is called, it goes through the inputs and looks for the
        ``__torch_function__`` method. If one is found, it is invoked with the operator as ``func`` as well as the
        ``args`` and ``kwargs`` of the original call.

        The default behavior of :class:`~torch.Tensor`'s is to retain a custom tensor type. For the :class:`Datapoint`
        use case, this has two downsides:

        1. Since some :class:`Datapoint`'s require metadata to be constructed, the default wrapping, i.e.
           ``return cls(func(*args, **kwargs))``, will fail for them.
        2. For most operations, there is no way of knowing if the input type is still valid for the output.

        For these reasons, the automatic output wrapping is turned off for most operators. The only exceptions are
        listed in _FORCE_TORCHFUNCTION_SUBCLASS
        """
        # Since super().__torch_function__ has no hook to prevent the coercing of the output into the input type, we
        # need to reimplement the functionality.

        if not all(issubclass(cls, t) for t in types):
            return NotImplemented

        with DisableTorchFunctionSubclass():
            output = func(*args, **kwargs or dict())

        if _must_return_subclass() or (func in _FORCE_TORCHFUNCTION_SUBCLASS and isinstance(args[0], cls)):
            # We also require the primary operand, i.e. `args[0]`, to be
            # an instance of the class that `__torch_function__` was invoked on. The __torch_function__ protocol will
            # invoke this method on *all* types involved in the computation by walking the MRO upwards. For example,
            # `torch.Tensor(...).to(datapoints.Image(...))` will invoke `datapoints.Image.__torch_function__` with
            # `args = (torch.Tensor(), datapoints.Image())` first. Without this guard, the original `torch.Tensor` would
            # be wrapped into a `datapoints.Image`.
            return cls._wrap_output(output, args, kwargs)

        if not _TORCHFUNCTION_SUBCLASS and isinstance(output, cls):
            # DisableTorchFunctionSubclass is ignored by inplace ops like `.add_(...)`,
            # so for those, the output is still a Datapoint. Thus, we need to manually unwrap.
            return output.as_subclass(torch.Tensor)

        return output

    def _make_repr(self, **kwargs: Any) -> str:
        # This is a poor man's implementation of the proposal in https://github.com/pytorch/pytorch/issues/76532.
        # If that ever gets implemented, remove this in favor of the solution on the `torch.Tensor` class.
        extra_repr = ", ".join(f"{key}={value}" for key, value in kwargs.items())
        return f"{super().__repr__()[:-1]}, {extra_repr})"

    # Add properties for common attributes like shape, dtype, device, ndim etc
    # this way we return the result without passing into __torch_function__
    @property
    def shape(self) -> _size:  # type: ignore[override]
        with DisableTorchFunctionSubclass():
            return super().shape

    @property
    def ndim(self) -> int:  # type: ignore[override]
        with DisableTorchFunctionSubclass():
            return super().ndim

    @property
    def device(self, *args: Any, **kwargs: Any) -> _device:  # type: ignore[override]
        with DisableTorchFunctionSubclass():
            return super().device

    @property
    def dtype(self) -> _dtype:  # type: ignore[override]
        with DisableTorchFunctionSubclass():
            return super().dtype

    def __deepcopy__(self: D, memo: Dict[int, Any]) -> D:
        # We need to detach first, since a plain `Tensor.clone` will be part of the computation graph, which does
        # *not* happen for `deepcopy(Tensor)`. A side-effect from detaching is that the `Tensor.requires_grad`
        # attribute is cleared, so we need to refill it before we return.
        # Note: We don't explicitly handle deep-copying of the metadata here. The only metadata we currently have is
        # `BoundingBoxes.format` and `BoundingBoxes.canvas_size`, which are immutable and thus implicitly deep-copied by
        # `BoundingBoxes.clone()`.
        return self.detach().clone().requires_grad_(self.requires_grad)  # type: ignore[return-value]
