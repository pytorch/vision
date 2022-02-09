import functools
import inspect
from typing import Any, Optional, Callable, TypeVar, Dict, Union

import PIL.Image
import torch
import torch.overrides
from torchvision.prototype import features
from torchvision.prototype.utils._internal import sequence_to_str

F = TypeVar("F", bound=features.Feature)

# Sentinel to use as default value of a dispatcher parameter if it is only required for a subset of the kernels. If the
# decorated function is called without the parameter for a kernel that requires it, an expressive :class:`TypeError` is
# raised.
FEATURE_SPECIFIC_PARAM = object()

# Sentinel to use as default value of a dispatcher parameter if the kernels use different default values for it.
FEATURE_SPECIFIC_DEFAULT = object()


def dispatch(
    kernels: Dict[Any, Callable[..., Union[torch.Tensor, F]]],
    *,
    pil_kernel: Optional[Callable] = None,
) -> Callable[[Callable[..., F]], Callable[..., F]]:
    """Decorates a function to automatically dispatch to ``kernels`` based on the call arguments.

    The function body of the dispatcher can be empty as it is never called. The signature and the docstring however are
    used in the documentation and thus should be accurate.

    The dispatch function should have this signature

    .. code:: python

        @dispatch
        def dispatch_fn(input, *args, **kwargs):
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
        TypeError: If ``pil_kernel`` is specified, but no kernel for :class:`~torchvision.prototype.features.Image` is
            available.
        TypeError: If the decorated function is called with neither a ``PIL`` image nor a :class:`~torch.Tensor`.
        TypeError: If the decorated function is called with an input that cannot be dispatched.
    """
    for feature_type in kernels:
        if not (issubclass(feature_type, features.Feature) and feature_type is not features.Feature):
            raise TypeError("XXX")
    if pil_kernel and features.Image not in kernels:
        raise TypeError("PIL kernel can only be registered for images")

    params = {feature_type: inspect.signature(kernel).parameters for feature_type, kernel in kernels.items()}

    def outer_wrapper(dispatch_fn: Callable[..., F]) -> Callable[..., F]:
        feature_specific_params = {
            feature_type: [
                name
                for name, param in inspect.signature(dispatch_fn).parameters.items()
                if param.default is FEATURE_SPECIFIC_PARAM
                and name in params_
                and params_[name].default is inspect.Parameter.empty
            ]
            for feature_type, params_ in params.items()
        }

        @functools.wraps(dispatch_fn)
        def inner_wrapper(input: F, *args: Any, **kwargs: Any) -> F:
            feature_type = type(input)

            if issubclass(feature_type, PIL.Image.Image):
                if pil_kernel is None:
                    raise TypeError("No PIL kernel")

                # TODO: maybe warn or fail here if we have decided on the scope of BC and deprecations
                return pil_kernel(input, *args, **kwargs)  # type: ignore[no-any-return]

            if not issubclass(feature_type, torch.Tensor):
                raise TypeError("No tensor")

            if not issubclass(feature_type, features.Feature):
                # TODO: maybe warn or fail here if we have decided on the scope of BC and deprecations
                input = features.Image(input)

            try:
                kernel = kernels[feature_type]
            except KeyError:
                raise TypeError(f"No support for {feature_type.__name__}") from None

            missing_args = [
                param
                for param in feature_specific_params[feature_type]
                if kwargs.get(param, FEATURE_SPECIFIC_PARAM) is FEATURE_SPECIFIC_PARAM
            ]
            if missing_args:
                raise TypeError(
                    f"{dispatch_fn.__name__}() missing {len(missing_args)} required keyword-only arguments "
                    f"for feature type {feature_type.__name__}: {sequence_to_str(missing_args, separate_last='and ')}"
                )

            output = kernel(input, *args, **kwargs)

            if not isinstance(output, feature_type):
                output = feature_type.new_like(input, output)

            return output

        return inner_wrapper

    return outer_wrapper
