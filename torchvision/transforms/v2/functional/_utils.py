import functools
from typing import Any

import torch
from torchvision.datapoints._datapoint import Datapoint


def is_simple_tensor(inpt: Any) -> bool:
    return isinstance(inpt, torch.Tensor) and not isinstance(inpt, Datapoint)


_KERNEL_REGISTRY = {}


def register_kernel(dispatcher, datapoint_cls, *, datapoint_wrapping=True):
    def datapoint_wrapper(kernel):
        @functools.wraps(kernel)
        def wrapper(inpt, *args, **kwargs):
            return type(inpt).wrap_like(inpt, kernel(inpt.as_subclass(torch.Tensor), *args, **kwargs))

        return wrapper

    registry = _KERNEL_REGISTRY.setdefault(dispatcher, {})
    if datapoint_cls in registry:
        raise TypeError(
            f"Dispatcher '{dispatcher.__name__}' already has a kernel registered for type '{datapoint_cls.__name__}'."
        )

    def decorator(kernel):
        registry[datapoint_cls] = datapoint_wrapper(kernel) if datapoint_wrapping else kernel
        return kernel

    return decorator


def _noop(inpt, *args, **kwargs):
    return inpt


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
