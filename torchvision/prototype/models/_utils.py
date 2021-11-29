import functools
import inspect
import warnings
from typing import Any, Dict, Optional, TypeVar, Callable, Tuple, Union, cast
from warnings import warn

from torch import nn
from torchvision.prototype.utils._internal import sequence_to_str

from ._api import Weights

W = TypeVar("W", bound=Weights)
M = TypeVar("M", bound=nn.Module)
V = TypeVar("V")


def handle_positional_to_keyword_only(
    builder: Callable[..., M],
    args: Tuple[Any, ...],
    kwargs: Dict[str, Any],
    *,
    keyword_only_start_idx: int,
    parameter_map: Dict[str, str],
) -> Tuple[Tuple[Any, ...], Dict[str, Any]]:
    keyword_only_parameters = [
        parameter_map.get(parameter, parameter) for parameter in tuple(inspect.signature(builder).parameters)
    ]

    args, keyword_only_args = args[:keyword_only_start_idx], args[keyword_only_start_idx:]

    if keyword_only_args:
        keyword_only_kwargs = dict(zip(keyword_only_parameters, keyword_only_args))
        if warn:
            warnings.warn(
                f"Using {sequence_to_str(tuple(keyword_only_kwargs.keys()), separate_last='and ')} "
                "as positional parameter(s) is deprecated. Please use them as keyword parameter(s) instead."
            )
        kwargs.update(keyword_only_kwargs)

    return args, kwargs


def handle_pretrained_to_weights(
    builder: Callable[..., M],
    kwargs: Dict[str, Any],
    *,
    default_weights: Dict[str, Tuple[str, Union[Optional[W], Callable[[Dict[str, Any]], Optional[W]]]]],
) -> Dict[str, Any]:
    for deprecated_param, (new_param, default) in default_weights.items():
        if not kwargs.pop(deprecated_param, False):
            continue

        warnings.warn(f"The parameter '{deprecated_param}' is deprecated, please use '{new_param}' instead.")

        default_arg = default(kwargs) if callable(default) else default
        if default_arg is None:
            raise ValueError(f"No checkpoint is available for model {builder.__name__}")

        kwargs[new_param] = default_arg

    return kwargs


def handle_legacy_interface(
    default_weights: Union[
        Optional[W],
        Callable[[Dict[str, Any]], Optional[W]],
        Dict[str, Tuple[str, Union[Optional[W], Callable[[Dict[str, Any]], Optional[W]]]]],
    ],
    *,
    keyword_only_start_idx: int = 0,
):
    if not isinstance(default_weights, dict):
        default_weights = dict(pretrained=("weights", default_weights))

    def outer_wrapper(builder: Callable[..., M]) -> Callable[..., M]:
        @functools.wraps(builder)
        def inner_wrapper(*args: Any, **kwargs: Any) -> M:
            # mypy does not pick up on the type change in the our scope. Note that although we use nonlocal here,
            # we do not change the default values, since `cast` is a no-op at runtime.
            nonlocal default_weights
            default_weights = cast(
                Dict[str, Tuple[str, Union[Optional[W], Callable[[Dict[str, Any]], Optional[W]]]]], default_weights
            )

            args, kwargs = handle_positional_to_keyword_only(
                builder,
                args,
                kwargs,
                keyword_only_start_idx=keyword_only_start_idx,
                parameter_map={
                    new_param: deprecated_param for deprecated_param, (new_param, _) in default_weights.items()
                },
            )
            kwargs = handle_pretrained_to_weights(builder, kwargs, default_weights=default_weights)

            return builder(*args, **kwargs)

        return inner_wrapper

    return outer_wrapper


def handle_num_categories_mismatch(*, parameter: str = "num_classes") -> Callable[[Callable[..., M]], Callable[..., M]]:
    def outer_wrapper(builder: Callable[..., M]) -> Callable[..., M]:
        @functools.wraps(builder)
        def inner_wrapper(*args: Any, weights: Optional[W], **kwargs: Any) -> M:
            if weights is not None and len(weights.meta["categories"]) != kwargs[parameter]:
                raise ValueError(
                    f"The number of categories of the weights does not match the `{parameter}` argument: "
                    f"{len(weights.meta['categories'])} != {kwargs[parameter]}."
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
