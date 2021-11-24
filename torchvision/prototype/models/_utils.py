import warnings
from typing import Any, Dict, Optional, TypeVar

from ._api import Weights


W = TypeVar("W", bound=Weights)


def _deprecated_param(
    deprecated_param: str, new_param: str, default_value: Optional[W], kwargs: Dict[str, Any]
) -> Optional[W]:
    warnings.warn(f"The parameter '{deprecated_param}' is deprecated, please use {new_param} instead.")
    if kwargs.pop(deprecated_param):
        if default_value is not None:
            return default_value
        else:
            raise ValueError("No checkpoint is available for model.")
    else:
        return None


def _ovewrite_named_param(kwargs: Dict[str, Any], param: str, new_value: Any) -> None:
    if param in kwargs:
        if kwargs[param] != new_value:
            raise ValueError(f"The parameter {param} expected value {new_value} but got {kwargs[param]} instead.")
    else:
        kwargs[param] = new_value


def _ovewrite_value_param(param: Any, new_value: Any) -> Any:
    if param is not None:
        if param != new_value:
            raise ValueError(f"The parameter {param} expected value {new_value} but got {param} instead.")
    return new_value
