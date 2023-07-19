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

    def decorator(kernel):
        _KERNEL_REGISTRY[(dispatcher, datapoint_cls)] = datapoint_wrapper(kernel) if datapoint_wrapping else kernel
        return kernel

    return decorator


def _get_kernel(dispatcher, datapoint_cls):
    return _KERNEL_REGISTRY[(dispatcher, datapoint_cls)]
