import functools
import warnings
from typing import Any, Dict, Optional, TypeVar, Callable, Tuple, Union

from torch import nn
from torchvision.prototype.utils._internal import kwonly_to_pos_or_kw

from ._api import Weights

W = TypeVar("W", bound=Weights)
M = TypeVar("M", bound=nn.Module)
V = TypeVar("V")


def handle_legacy_interface(
    default_weights: Union[
        Optional[W],
        Callable[[Dict[str, Any]], Optional[W]],
        Dict[str, Tuple[str, Union[Optional[W], Callable[[Dict[str, Any]], Optional[W]]]]],
    ],
):
    if not isinstance(default_weights, dict):
        default_weights = dict(pretrained=("weights", default_weights))

    def outer_wrapper(builder: Callable[..., M]) -> Callable[..., M]:
        @functools.wraps(builder)
        def inner_wrapper(**kwargs: Any) -> M:
            for deprecated_param, (new_param, default) in default_weights.items():  # type: ignore[union-attr]
                if not kwargs.pop(deprecated_param, False):
                    continue

                default_arg = default(kwargs) if callable(default) else default
                if default_arg is None:
                    raise ValueError(f"No checkpoint is available for model {builder.__name__}")

                warnings.warn(
                    f"The parameter '{deprecated_param}' is deprecated, please use '{new_param}' instead. "
                    f"The current behavior is equivalent to passing `weights={default_arg}`. "
                    f"You can also use `weights='default'` to get the most up-to-date weights."
                )
                kwargs[new_param] = default_arg

            return builder(**kwargs)

        return kwonly_to_pos_or_kw(
            param_map={
                new_param: deprecated_param
                for deprecated_param, (new_param, _) in default_weights.items()  # type: ignore[union-attr]
            },
            warn=True,
        )(inner_wrapper)

    return outer_wrapper


def handle_num_categories_mismatch(*, param: str = "num_classes") -> Callable[[Callable[..., M]], Callable[..., M]]:
    def outer_wrapper(builder: Callable[..., M]) -> Callable[..., M]:
        @functools.wraps(builder)
        def inner_wrapper(*args: Any, weights: Optional[W], **kwargs: Any) -> M:
            if param in kwargs and weights is not None and kwargs[param] != len(weights.meta["categories"]):
                raise ValueError(
                    f"The number of categories of the weights does not match the `{param}` argument: "
                    f"{len(weights.meta['categories'])} != {kwargs[param]}."
                )

            return builder(*args, weights=weights, **kwargs)

        return inner_wrapper

    return outer_wrapper


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
