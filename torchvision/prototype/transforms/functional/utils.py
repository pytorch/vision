import types
from typing import Any

import torch
import torch.overrides
from torchvision.prototype import features


class Dispatcher:
    def __init__(self, dispatch_fn):
        self._dispatch_fn = dispatch_fn
        self._support = set()

    def supports(self, obj: Any) -> bool:
        return (obj if isinstance(obj, type) else type(obj)) in self._support

    def implements(self, feature_type):
        def wrapper(implement_fn):
            feature_type._KERNELS[self._dispatch_fn] = implement_fn
            self._support.add(feature_type)
            return implement_fn

        return wrapper

    def __call__(self, input, *args, **kwargs):
        if not isinstance(input, torch.Tensor):
            raise ValueError("No tensor")

        if not (issubclass(type(input), features.Feature)):
            input = features.Image(input)

        if not self.supports(input):
            raise ValueError(f"No support for {type(input).__name__}")

        return torch.overrides.handle_torch_function(self._dispatch_fn, (input,), input, *args, **kwargs)


def _from_legacy_kernel(legacy_kernel, new_name=None):
    kernel = types.FunctionType(
        code=legacy_kernel.__code__,
        globals=legacy_kernel.__globals__,
        name=new_name or f"{legacy_kernel.__name__}_image",
        argdefs=legacy_kernel.__defaults__,
        closure=legacy_kernel.__closure__,
    )
    kernel.__annotations__ = legacy_kernel.__annotations__
    return kernel
