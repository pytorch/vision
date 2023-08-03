import functools
import warnings
from typing import Any, Callable, Dict, Type

import torch
from torchvision import datapoints


def is_simple_tensor(inpt: Any) -> bool:
    return isinstance(inpt, torch.Tensor) and not isinstance(inpt, datapoints.Datapoint)


# {dispatcher: {input_type: type_specific_kernel}}
_KERNEL_REGISTRY: Dict[Callable, Dict[Type, Callable]] = {}


def _kernel_datapoint_wrapper(kernel):
    @functools.wraps(kernel)
    def wrapper(inpt, *args, **kwargs):
        output = kernel(inpt.as_subclass(torch.Tensor), *args, **kwargs)
        return type(inpt).wrap_like(inpt, output)

    return wrapper


def _register_kernel_internal(dispatcher, input_type, *, datapoint_wrapper=True):
    registry = _KERNEL_REGISTRY.setdefault(dispatcher, {})
    if input_type in registry:
        raise TypeError(f"Dispatcher {dispatcher} already has a kernel registered for type {input_type}.")

    def decorator(kernel):
        registry[input_type] = (
            _kernel_datapoint_wrapper(kernel)
            if issubclass(input_type, datapoints.Datapoint) and datapoint_wrapper
            else kernel
        )
        return kernel

    return decorator


def register_kernel(dispatcher, datapoint_cls):
    if not (
        callable(dispatcher)
        and getattr(dispatcher, "__module__", "").startswith("torchvision.transforms.v2.functional")
    ):
        raise ValueError(
            f"Kernels can only be registered on dispatchers from the torchvision.transforms.v2.functional namespace, "
            f"but got {dispatcher}."
        )
    elif not (
        isinstance(datapoint_cls, type)
        and issubclass(datapoint_cls, datapoints.Datapoint)
        and datapoint_cls is not datapoints.Datapoint
    ):
        raise ValueError(
            f"Kernels can only be registered for subclasses of torchvision.datapoints.Datapoint, "
            f"but got {datapoint_cls}."
        )
    return _register_kernel_internal(dispatcher, datapoint_cls, datapoint_wrapper=False)


def _get_kernel(dispatcher, input_type):
    registry = _KERNEL_REGISTRY.get(dispatcher)
    if not registry:
        raise ValueError(f"No kernel registered for dispatcher {dispatcher.__name__}.")

    # in case we have an exact type match, we take a shortcut
    if input_type in registry:
        return registry[input_type]

    # in case of datapoints, we check if we have a kernel for a superclass registered
    if issubclass(input_type, datapoints.Datapoint):
        for cls in input_type.__mro__:
            if cls is datapoints.Datapoint:
                break
            elif cls in registry:
                return registry[cls]

        # Note that in the future we are not going to return a noop here, but rather raise the
        # error below
        return _noop

    raise TypeError(
        f"Dispatcher {dispatcher} supports inputs of type torch.Tensor, PIL.Image.Image, "
        f"and subclasses of torchvision.datapoints.Datapoint, "
        f"but got {input_type} instead."
    )


# Everything below this block is stuff that we need right now, since it looks like we need to release in an intermediate
# stage. See https://github.com/pytorch/vision/pull/7747#issuecomment-1661698450 for details.


# In the future, the default behavior will be to error on unsupported types in dispatchers. The noop behavior that we
# need for transforms will be handled by _get_kernel rather than actually registering no-ops on the dispatcher.
# Finally, the use case of preventing users from registering kernels for our builtin types will be handled inside
# register_kernel.
def _register_explicit_noop(*datapoints_classes, warn_passthrough=False):
    """
    Although this looks redundant with the no-op behavior of _get_kernel, this explicit registration prevents users
    from registering kernels for builtin datapoints on builtin dispatchers that rely on the no-op behavior.

    For example, without explicit no-op registration the following would be valid user code:

    .. code::
        from torchvision.transforms.v2 import functional as F

        @F.register_kernel(F.adjust_brightness, datapoints.BoundingBox)
        def lol(...):
            ...
    """

    def decorator(dispatcher):
        for cls in datapoints_classes:
            msg = (
                f"F.{dispatcher.__name__} is currently passing through inputs of type datapoints.{cls.__name__}. "
                f"This will likely change in the future."
            )
            _register_kernel_internal(dispatcher, cls, datapoint_wrapper=False)(
                functools.partial(_noop, __msg__=msg if warn_passthrough else None)
            )
        return dispatcher

    return decorator


def _noop(inpt, *args, __msg__=None, **kwargs):
    if __msg__:
        warnings.warn(__msg__, UserWarning, stacklevel=2)
    return inpt


# TODO: we only need this, since our default behavior in case no kernel is found is passthrough. When we change that
# to error later, this decorator can be removed, since the error will be raised by _get_kernel
def _register_unsupported_type(*input_types):
    def kernel(inpt, *args, __dispatcher_name__, **kwargs):
        raise TypeError(f"F.{__dispatcher_name__} does not support inputs of type {type(inpt)}.")

    def decorator(dispatcher):
        for input_type in input_types:
            _register_kernel_internal(dispatcher, input_type, datapoint_wrapper=False)(
                functools.partial(kernel, __dispatcher_name__=dispatcher.__name__)
            )
        return dispatcher

    return decorator


# This basically replicates _register_kernel_internal, but with a specialized wrapper for five_crop / ten_crop
# We could get rid of this by letting _register_kernel_internal take arbitrary dispatchers rather than wrap_kernel: bool
def _register_five_ten_crop_kernel(dispatcher, input_type):
    registry = _KERNEL_REGISTRY.setdefault(dispatcher, {})
    if input_type in registry:
        raise TypeError(f"Dispatcher '{dispatcher}' already has a kernel registered for type '{input_type}'.")

    def wrap(kernel):
        @functools.wraps(kernel)
        def wrapper(inpt, *args, **kwargs):
            output = kernel(inpt, *args, **kwargs)
            container_type = type(output)
            return container_type(type(inpt).wrap_like(inpt, o) for o in output)

        return wrapper

    def decorator(kernel):
        registry[input_type] = wrap(kernel) if issubclass(input_type, datapoints.Datapoint) else kernel
        return kernel

    return decorator
