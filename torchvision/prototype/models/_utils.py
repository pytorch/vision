import functools
import warnings
from typing import Any, Dict, Optional, TypeVar, Callable, Tuple, Union

from torch import nn
from torchvision.prototype.utils._internal import kwonly_to_pos_or_kw

from ._api import WeightsEnum

W = TypeVar("W", bound=WeightsEnum)
M = TypeVar("M", bound=nn.Module)
V = TypeVar("V")


def handle_legacy_interface(
    **pretrained_weights: Tuple[str, Union[Optional[W], Callable[[Dict[str, Any]], Optional[W]]]]
):
    def outer_wrapper(builder: Callable[..., M]) -> Callable[..., M]:
        @functools.wraps(builder)
        def inner_wrapper(**kwargs: Any) -> M:
            for weights_param, (pretrained_param, default) in pretrained_weights.items():  # type: ignore[union-attr]
                weights_arg = kwargs.get(weights_param)
                if weights_param in kwargs and not isinstance(weights_arg, bool):
                    continue

                pretrained_positional = pretrained_param not in kwargs
                pretrained_arg = weights_arg if pretrained_positional else kwargs.pop(pretrained_param)
                if pretrained_arg:
                    default_arg = default(kwargs) if callable(default) else default
                    if default_arg is None:
                        raise ValueError(f"No checkpoint is available for model {builder.__name__}")
                else:
                    default_arg = None

                msg = f"The current behavior is equivalent to passing `{weights_param}={default_arg}`."
                if not pretrained_positional:
                    msg = (
                        f"The parameter '{pretrained_param}' is deprecated, please use '{weights_param}' instead. {msg}"
                    )
                if default_arg is not None:
                    msg = f"{msg} You can also use `{weights_param}='default'` to get the most up-to-date weights."
                warnings.warn(msg)
                kwargs[weights_param] = default_arg

            return builder(**kwargs)

        return kwonly_to_pos_or_kw(inner_wrapper)

    return outer_wrapper


# TODO: remove this in favor of handle_legacy_interface
def _deprecated_param(
    kwargs: Dict[str, Any], deprecated_param: str, new_param: str, default_value: Optional[W]
) -> Optional[W]:
    warnings.warn(f"The parameter '{deprecated_param}' is deprecated, please use '{new_param}' instead.")
    if kwargs.pop(deprecated_param):
        if default_value is not None:
            return default_value
        else:
            raise ValueError("No checkpoint is available for model.")
    else:
        return None


# TODO: remove this in favor of handle_legacy_interface
def _deprecated_positional(kwargs: Dict[str, Any], deprecated_param: str, new_param: str, default_value: V) -> None:
    warnings.warn(
        f"The positional parameter '{deprecated_param}' is deprecated, please use keyword parameter '{new_param}'"
        + " instead."
    )
    kwargs[deprecated_param] = default_value


def _ovewrite_named_param(kwargs: Dict[str, Any], param: str, new_value: V) -> None:
    if param in kwargs:
        if kwargs[param] != new_value:
            raise ValueError(f"The parameter '{param}' expected value {new_value} but got {kwargs[param]} instead.")
    else:
        kwargs[param] = new_value


def _ovewrite_value_param(param: Optional[V], new_value: V) -> V:
    if param is not None:
        if param != new_value:
            raise ValueError(f"The parameter '{param}' expected value {new_value} but got {param} instead.")
    return new_value
