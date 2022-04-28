from typing import Any, cast, TypeVar, Union, Optional, Type, Callable, Tuple, Sequence, Mapping

import torch
from torch._C import _TensorBase, DisableTorchFunction


F = TypeVar("F", bound="_Feature")


class _Feature(torch.Tensor):
    def __new__(
        cls: Type[F],
        data: Any,
        *,
        dtype: Optional[torch.dtype] = None,
        device: Optional[Union[torch.device, str, int]] = None,
        requires_grad: bool = False,
    ) -> F:
        return cast(
            F,
            torch.Tensor._make_subclass(
                cast(_TensorBase, cls),
                torch.as_tensor(data, dtype=dtype, device=device),  # type: ignore[arg-type]
                requires_grad,
            ),
        )

    @classmethod
    def new_like(
        cls: Type[F],
        other: F,
        data: Any,
        *,
        dtype: Optional[torch.dtype] = None,
        device: Optional[Union[torch.device, str, int]] = None,
        requires_grad: Optional[bool] = None,
        **kwargs: Any,
    ) -> F:
        return cls(
            data,
            dtype=dtype if dtype is not None else other.dtype,
            device=device if device is not None else other.device,
            requires_grad=requires_grad if requires_grad is not None else other.requires_grad,
            **kwargs,
        )

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

        The default behavior of :class:`~torch.Tensor`'s is to retain a custom tensor type. For the :class:`Feature`
        use case, this has two downsides:

        1. Since some :class:`Feature`'s require metadata to be constructed, the default wrapping, i.e.
           ``return cls(func(*args, **kwargs))``, will fail for them.
        2. For most operations, there is no way of knowing if the input type is still valid for the output.

        For these reasons, the automatic output wrapping is turned off for most operators.

        Exceptions to this are:

        - :func:`torch.clone`
        - :meth:`torch.Tensor.to`
        """
        kwargs = kwargs or dict()
        with DisableTorchFunction():
            output = func(*args, **kwargs)

        if func is torch.Tensor.clone:
            return cls.new_like(args[0], output)
        elif func is torch.Tensor.to:
            return cls.new_like(args[0], output, dtype=output.dtype, device=output.device)
        else:
            return output
