import functools
import inspect
import warnings
from typing import Any

import torch
from torchvision import datapoints
from torchvision.datapoints._datapoint import Datapoint


def is_simple_tensor(inpt: Any) -> bool:
    return isinstance(inpt, torch.Tensor) and not isinstance(inpt, Datapoint)


_KERNEL_REGISTRY = {}


def _kernel_wrapper_internal(dispatcher, kernel):
    dispatcher_params = list(inspect.signature(dispatcher).parameters)[1:]
    kernel_params = list(inspect.signature(kernel).parameters)[1:]

    needs_args_kwargs_handling = kernel_params != dispatcher_params

    kernel_params = set(kernel_params)
    explicit_metadata = {
        input_type: available_metadata & kernel_params
        for input_type, available_metadata in [(datapoints.BoundingBoxes, {"format", "canvas_size"})]
    }

    @functools.wraps(kernel)
    def wrapper(inpt, *args, **kwargs):
        input_type = type(inpt)

        if needs_args_kwargs_handling:
            # Convert args to kwargs to simplify further processing
            kwargs.update(dict(zip(dispatcher_params, args)))
            args = ()

            # drop parameters that are not relevant for the kernel, but have a default value
            # in the dispatcher
            for kwarg in kwargs.keys() - kernel_params:
                del kwargs[kwarg]

            # add parameters that are passed implicitly to the dispatcher as metadata,
            # but have to be explicit for the kernel
            for kwarg in explicit_metadata.get(input_type, set()):
                kwargs[kwarg] = getattr(inpt, kwarg)

        output = kernel(inpt.as_subclass(torch.Tensor), *args, **kwargs)

        if isinstance(inpt, datapoints.BoundingBoxes) and isinstance(output, tuple):
            output, canvas_size = output
            metadata = dict(canvas_size=canvas_size)
        else:
            metadata = dict()

        return input_type.wrap_like(inpt, output, **metadata)

    return wrapper


def _register_kernel_internal(dispatcher, datapoint_cls, *, wrap_kernel=True):
    registry = _KERNEL_REGISTRY.setdefault(dispatcher, {})
    if datapoint_cls in registry:
        raise TypeError(
            f"Dispatcher '{dispatcher.__name__}' already has a kernel registered for type '{datapoint_cls.__name__}'."
        )

    def decorator(kernel):
        registry[datapoint_cls] = _kernel_wrapper_internal(dispatcher, kernel) if wrap_kernel else kernel
        return kernel

    return decorator


def register_kernel(dispatcher, datapoint_cls):
    return _register_kernel_internal(dispatcher, datapoint_cls, wrap_kernel=False)


def _noop(inpt, *args, __future_warning__=None, **kwargs):
    if __future_warning__:
        warnings.warn(__future_warning__, FutureWarning, stacklevel=2)
    return inpt


def _register_explicit_noop(*datapoints_classes, future_warning=False):
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
            register_kernel(dispatcher, cls)(
                functools.partial(_noop, __future_warning__=msg if future_warning else None)
            )
        return dispatcher

    return decorator


# TODO: we only need this, since our default behavior in case no kernel is found is passthrough. When we change that
# to error later, this decorator can be removed, since the error will be raised by _get_kernel
def _register_unsupported_type(*datapoints_classes):
    def kernel(inpt, *args, __dispatcher_name__, **kwargs):
        raise TypeError(f"F.{__dispatcher_name__} does not support inputs of type {type(inpt)}.")

    def decorator(dispatcher):
        for cls in datapoints_classes:
            register_kernel(dispatcher, cls)(functools.partial(kernel, __dispatcher_name__=dispatcher.__name__))
        return dispatcher

    return decorator


def _get_kernel(dispatcher, datapoint_cls):
    registry = _KERNEL_REGISTRY.get(dispatcher)
    if not registry:
        raise ValueError(f"No kernel registered for dispatcher '{dispatcher.__name__}'.")

    if datapoint_cls in registry:
        return registry[datapoint_cls]

    for registered_cls, kernel in registry.items():
        if issubclass(datapoint_cls, registered_cls):
            return kernel

    return _noop


# This basically replicates _register_kernel_internal, but with a specialized wrapper for five_crop / ten_crop
# We could get rid of this by letting _register_kernel_internal take arbitrary dispatchers rather than wrap_kernel: bool
# TODO: decide if we want that
def _register_five_ten_crop_kernel(dispatcher, datapoint_cls):
    registry = _KERNEL_REGISTRY.setdefault(dispatcher, {})
    if datapoint_cls in registry:
        raise TypeError(
            f"Dispatcher '{dispatcher.__name__}' already has a kernel registered for type '{datapoint_cls.__name__}'."
        )

    def wrap(kernel):
        @functools.wraps(kernel)
        def wrapper(inpt, *args, **kwargs):
            output = kernel(inpt, *args, **kwargs)
            container_type = type(output)
            return container_type(type(inpt).wrap_like(inpt, o) for o in output)

        return wrapper

    def decorator(kernel):
        registry[datapoint_cls] = wrap(kernel)
        return kernel

    return decorator
