import warnings
from typing import Any, Dict, Optional, TypeVar

from ._api import WeightsEnum


W = TypeVar("W", bound=WeightsEnum)
V = TypeVar("V")


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
