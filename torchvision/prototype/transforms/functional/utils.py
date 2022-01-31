import functools
from typing import Any, Type, Optional, Callable

import PIL.Image
import torch
import torch.overrides
from torchvision.prototype import features
from torchvision.prototype.utils._internal import sequence_to_str


def is_supported(obj: Any, *types: Type) -> bool:
    return (obj if isinstance(obj, type) else type(obj)) in types


class Dispatcher:
    FEATURE_SPECIFIC_PARAM = object()

    def __init__(self, dispatch_fn):
        self._dispatch_fn = dispatch_fn
        self._support = set()
        self._pil_kernel: Optional[Callable] = None

    def supports(self, obj: Any) -> bool:
        return is_supported(obj, *self._support)

    def implements(self, feature_type, *, feature_specific_params=(), pil_kernel=None):
        if pil_kernel is not None:
            if not issubclass(feature_type, features.Image):
                raise TypeError("PIL kernel can only be registered for images")

            self._pil_kernel = pil_kernel

        def outer_wrapper(implement_fn):
            @functools.wraps(implement_fn)
            def inner_wrapper(*args, **kwargs) -> Any:
                missing = [
                    param
                    for param in feature_specific_params
                    if kwargs.get(param, self.FEATURE_SPECIFIC_PARAM) is self.FEATURE_SPECIFIC_PARAM
                ]
                if missing:
                    raise TypeError(
                        f"{self._dispatch_fn.__name__}() missing {len(missing)} required keyword-only arguments "
                        f"for feature type {feature_type.__name__}: {sequence_to_str(missing, separate_last='and ')}"
                    )

                return implement_fn(*args, **kwargs)

            feature_type._KERNELS[self._dispatch_fn] = inner_wrapper
            self._support.add(feature_type)

            return inner_wrapper

        return outer_wrapper

    def __call__(self, input, *args, **kwargs):
        feature_type = type(input)
        if issubclass(feature_type, PIL.Image.Image):
            if self._pil_kernel is None:
                raise TypeError("No PIL kernel")

            return self._pil_kernel(input, *args, **kwargs)
        elif not isinstance(input, torch.Tensor):
            raise TypeError("No tensor")

        if not (issubclass(type(input), features.Feature)):
            input = features.Image(input)

        if not self.supports(input):
            raise ValueError(f"No support for {type(input).__name__}")

        return torch.overrides.handle_torch_function(self._dispatch_fn, (input,), input, *args, **kwargs)
