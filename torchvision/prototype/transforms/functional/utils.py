import functools
import inspect
from typing import Any, Type, Optional, Callable, TypeVar, Dict, Union

import PIL.Image
import torch
import torch.overrides
from torchvision.prototype import features
from torchvision.prototype.utils._internal import sequence_to_str

F = TypeVar("F", bound=features.Feature)


def is_supported(obj: Any, *types: Type) -> bool:
    return (obj if isinstance(obj, type) else type(obj)) in types


class dispatch:
    FEATURE_SPECIFIC_PARAM = object()
    FEATURE_SPECIFIC_DEFAULT = object()

    def __init__(self, dispatch_fn: Callable[..., F]):
        self._dispatch_fn = dispatch_fn
        self.__doc__ = dispatch_fn.__doc__
        self.__signature__ = inspect.signature(dispatch_fn)

        self._fns: Dict[Type[F], Callable[..., F]] = {}
        self._pil_fn: Optional[Callable] = None

    def supports(self, obj: Any) -> bool:
        return is_supported(obj, *self._fns.keys())

    def register(
        self,
        fn: Callable[..., Union[torch.Tensor, F]],
        feature_type: Type[F],
        *,
        wrap_output: bool = True,
        pil_kernel: Optional[Callable[..., PIL.Image.Image]] = None,
    ) -> None:
        if not (issubclass(feature_type, features.Feature) and feature_type is not features.Feature):
            raise TypeError("Can only register kernels for subclasses of `torchvision.prototype.features.Feature`.")

        if pil_kernel is not None:
            if not issubclass(feature_type, features.Image):
                raise TypeError("PIL kernel can only be registered for images")

            self._pil_fn = pil_kernel

        params = inspect.signature(fn).parameters
        feature_specific_params = [
            name
            for name, param in self.__signature__.parameters.items()
            if param.default is self.FEATURE_SPECIFIC_PARAM
            and name in params
            and params[name].default is inspect.Parameter.empty
        ]

        @functools.wraps(fn)
        def wrapper(input: F, *args: Any, **kwargs: Any) -> Any:
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

            output = fn(input, *args, **kwargs)

            if wrap_output:
                output = feature_type.new_like(input, output)

            return output

        self._fns[feature_type] = wrapper

    def __call__(self, input: F, *args: Any, **kwargs: Any) -> Union[F, PIL.Image.Image]:
        feature_type = type(input)

        if issubclass(feature_type, PIL.Image.Image):
            if self._pil_fn is None:
                raise TypeError("No PIL kernel")

            return self._pil_fn(input, *args, **kwargs)
        elif not issubclass(feature_type, torch.Tensor):
            raise TypeError("No tensor")

        if not issubclass(feature_type, features.Feature):
            input = features.Image(input)

        if not self.supports(feature_type):
            raise ValueError(f"No support for {feature_type.__name__}")

        return self._fns[feature_type](input, *args, **kwargs)
