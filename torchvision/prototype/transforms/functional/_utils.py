import functools
import inspect
from typing import Any, Optional, Callable, TypeVar, Dict

import torch
import torch.overrides
from torchvision.prototype import features

F = TypeVar("F", bound=features.Feature)


def dispatch(kernels: Dict[Any, Optional[Callable]]) -> Callable[[Callable[..., F]], Callable[..., F]]:
    """Decorates a function to automatically dispatch to ``kernels`` based on the call arguments.

    The function body of the dispatcher can be empty as it is never called. The signature and the docstring however are
    used in the documentation and thus should be accurate.

    The dispatch function should have this signature

    .. code:: python

        from typing import Any, TypeVar

        from torchvision.protoype import features

        T = TypeVar("T", bound=features.Feature)

        @dispatch
        def dispatch_fn(input: T, *args: Any, **kwargs: Any) -> T:
            ...

    where ``input`` is a strict subclass of :class:`~torchvision.prototype.features.Feature` and is used to determine
    which kernel to dispatch to.

    .. note::

        For backward compatibility, ``input`` can also be a ``PIL`` image in which case the call will be dispatched to
        ``pil_kernel`` if available. Furthermore, ``input`` can also be a vanilla :class:`~torch.Tensor` in which case
        it will be converted into a :class:`~torchvision.prototype.features.Image`.

    Args:
        kernels: Dictionary of subclasses of :class:`~torchvision.prototype.features.Feature` that maps to a kernel
            to call for this feature type.
        pil_kernel: Optional kernel for ``PIL`` images.

    Raises:
        TypeError: If any key in ``kernels`` is not a strict subclass of
            :class:`~torchvision.prototype.features.Feature`.
        TypeError: If any value in ``kernels`` is not callable with ``kernel(input, *args, **kwargs)``.
        TypeError: If ``pil_kernel`` is specified, but no kernel for :class:`~torchvision.prototype.features.Image` is
            available.
        TypeError: If the decorated function is called with neither a ``PIL`` image nor a :class:`~torch.Tensor`.
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
                        f"dispatch_fn() did not handle inputs of type {type(input).__name__} "
                        f"although it was configured to do so."
                    )
            else:
                output = kernel(input, *args, **kwargs)

            if issubclass(feature_type, features.Feature) and type(output) is torch.Tensor:
                output = feature_type.new_like(input, output)

            return output

        return inner_wrapper

    return outer_wrapper
