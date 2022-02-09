import functools
import inspect
from typing import Any, Optional, Callable, TypeVar, Dict, Union

import PIL.Image
import torch
import torch.overrides
from torchvision.prototype import features
from torchvision.prototype.utils._internal import sequence_to_str

F = TypeVar("F", bound=features.Feature)


class dispatch:
    FEATURE_SPECIFIC_PARAM = object()
    FEATURE_SPECIFIC_DEFAULT = object()

    def __init__(
        self,
        kernels: Dict[Any, Callable[..., Union[torch.Tensor, F]]],
        *,
        pil_kernel: Optional[Callable] = None,
    ) -> None:
        self._kernels = kernels
        if pil_kernel and features.Image not in kernels:
            raise TypeError("PIL kernel can only be registered for images")
        self._pil_kernel = pil_kernel

    def __call__(self, dispatch_fn: Callable[..., F]) -> Callable[..., F]:
        params = {feature_type: inspect.signature(kernel).parameters for feature_type, kernel in self._kernels.items()}
        feature_specific_params = {
            feature_type: [
                name
                for name, param in inspect.signature(dispatch_fn).parameters.items()
                if param.default is self.FEATURE_SPECIFIC_PARAM
                and name in params_
                and params_[name].default is inspect.Parameter.empty
            ]
            for feature_type, params_ in params.items()
        }

        @functools.wraps(dispatch_fn)
        def wrapper(input: F, *args: Any, **kwargs: Any) -> F:
            feature_type = type(input)

            if issubclass(feature_type, PIL.Image.Image):
                if self._pil_kernel is None:
                    raise TypeError("No PIL kernel")

                return self._pil_kernel(input, *args, **kwargs)  # type: ignore[no-any-return]

            if not issubclass(feature_type, torch.Tensor):
                raise TypeError("No tensor")

            if not issubclass(feature_type, features.Feature):
                input = features.Image(input)

            try:
                kernel = self._kernels[feature_type]
            except KeyError:
                raise ValueError(f"No support for {feature_type.__name__}") from None

            missing_args = [
                param
                for param in feature_specific_params[feature_type]
                if kwargs.get(param, self.FEATURE_SPECIFIC_PARAM) is self.FEATURE_SPECIFIC_PARAM
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

        return wrapper
