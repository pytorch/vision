import functools
import inspect
from typing import Any, Optional, Callable, TypeVar, Dict

import torch
import torch.overrides
from torchvision.prototype import features

F = TypeVar("F", bound=features._Feature)


def dispatch(kernels: Dict[Any, Optional[Callable]]) -> Callable[[Callable[..., F]], Callable[..., F]]:
    """Decorates a function to automatically dispatch to registered kernels based on the call arguments.

    The dispatch function should have this signature

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

    def check_kernel(kernel: Any) -> bool:
        if kernel is None:
            return True

        if not callable(kernel):
            return False

        params = list(inspect.signature(kernel).parameters.values())
        if not params:
            return False

        return params[0].kind != inspect.Parameter.KEYWORD_ONLY

    for feature_type, kernel in kernels.items():
        if not check_kernel(kernel):
            raise TypeError(
                f"Kernel for feature type {feature_type.__name__} is not callable with kernel(input, *args, **kwargs)."
            )

    def outer_wrapper(dispatch_fn: Callable[..., F]) -> Callable[..., F]:
        @functools.wraps(dispatch_fn)
        def inner_wrapper(input: F, *args: Any, **kwargs: Any) -> F:
            feature_type = type(input)
            try:
                kernel = kernels[feature_type]
            except KeyError:
                try:
                    feature_type, kernel = next(
                        (feature_type, kernel)
                        for feature_type, kernel in kernels.items()
                        if isinstance(input, feature_type)
                    )
                except StopIteration:
                    raise TypeError(f"No support for {type(input).__name__}") from None

            if kernel is None:
                output = dispatch_fn(input, *args, **kwargs)
                if output is None:
                    raise RuntimeError(
                        f"{dispatch_fn.__name__}() did not handle inputs of type {type(input).__name__} "
                        f"although it was configured to do so."
                    )
            else:
                output = kernel(input, *args, **kwargs)

            if issubclass(feature_type, features._Feature) and type(output) is torch.Tensor:
                output = feature_type.new_like(input, output)

            return output

        return inner_wrapper

    return outer_wrapper
