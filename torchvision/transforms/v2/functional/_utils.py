import functools
import inspect
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

    # this avoids converting list -> set at runtime below
    kernel_params = set(kernel_params)

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
            for kwarg in input_type.__annotations__.keys() & kernel_params:
                kwargs[kwarg] = getattr(inpt, kwarg)

        output = kernel(inpt.as_subclass(torch.Tensor), *args, **kwargs)

        if isinstance(inpt, datapoints.BoundingBox) and isinstance(output, tuple):
            output, spatial_size = output
            metadata = dict(spatial_size=spatial_size)
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


def _noop(inpt, *args, **kwargs):
    return inpt


def _register_explicit_noop(dispatcher, *datapoints_classes):
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
    for cls in datapoints_classes:
        register_kernel(dispatcher, cls)(_noop)


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
