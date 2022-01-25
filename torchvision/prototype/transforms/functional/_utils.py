import inspect
from typing import Any, Optional, Callable, TypeVar, Mapping, Type

import torch
import torch.overrides
from torchvision.prototype import features

F = TypeVar("F", bound=features._Feature)


class Dispatcher:
    """Wrap a function to automatically dispatch to registered kernels based on the call arguments.

    The wrapped function should have this signature

    .. code:: python

        @dispatch(
            ...
        )
        def dispatch_fn(input, *args, **kwargs):
            ...

    where ``input`` is used to determine which kernel to dispatch to.

    Args:
        kernels: Dictionary with types as keys that maps to a kernel to call. The resolution order is checking for
            exact type matches first and if none is found falls back to checking for subclasses. If a value is
            ``None``, the decorated function is called.

    Raises:
        TypeError: If any value in ``kernels`` is not callable with ``kernel(input, *args, **kwargs)``.
        TypeError: If the decorated function is called with an input that cannot be dispatched.
    """

    def __init__(self, fn: Callable, kernels: Mapping[Type, Optional[Callable]]):
        self._fn = fn

        for feature_type, kernel in kernels.items():
            if not self._check_kernel(kernel):
                raise TypeError(
                    f"Kernel for feature type {feature_type.__name__} is not callable with "
                    f"kernel(input, *args, **kwargs)."
                )

        self._kernels = kernels

    def _check_kernel(self, kernel: Optional[Callable]) -> bool:
        if kernel is None:
            return True

        if not callable(kernel):
            return False

        params = list(inspect.signature(kernel).parameters.values())
        if not params:
            return False

        return params[0].kind != inspect.Parameter.KEYWORD_ONLY

    def _resolve(self, feature_type: Type) -> Optional[Callable]:
        try:
            return self._kernels[feature_type]
        except KeyError:
            try:
                return next(
                    kernel
                    for registered_feature_type, kernel in self._kernels.items()
                    if issubclass(feature_type, registered_feature_type)
                )
            except StopIteration:
                raise TypeError(f"No support for feature type {type(input).__name__}") from None

    def __contains__(self, obj: Any) -> bool:
        try:
            self._resolve(type(obj))
            return True
        except TypeError:
            return False

    def __call__(self, input: Any, *args: Any, **kwargs: Any) -> Any:
        kernel = self._resolve(type(input))

        if kernel is None:
            output = self._fn(input, *args, **kwargs)
            if output is None:
                raise RuntimeError(
                    f"{self._fn.__name__}() did not handle inputs of type {type(input).__name__} "
                    f"although it was configured to do so."
                )
        else:
            output = kernel(input, *args, **kwargs)

        if isinstance(input, features._Feature) and type(output) is torch.Tensor:
            output = type(input).new_like(input, output)

        return output


def dispatch(kernels: Mapping[Type, Optional[Callable]]) -> Callable[[Callable], Dispatcher]:
    """Decorates a function and turns it into a :class:`Dispatcher`."""
    return lambda fn: Dispatcher(fn, kernels)
