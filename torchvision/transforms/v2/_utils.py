import functools
import numbers
from collections import defaultdict
from typing import Any, Dict, Literal, Sequence, Type, TypeVar, Union

from torchvision import datapoints
from torchvision.datapoints._datapoint import _FillType, _FillTypeJIT

from torchvision.transforms.transforms import _check_sequence_input, _setup_angle, _setup_size  # noqa: F401


def _setup_float_or_seq(arg: Union[float, Sequence[float]], name: str, req_size: int = 2) -> Sequence[float]:
    if not isinstance(arg, (float, Sequence)):
        raise TypeError(f"{name} should be float or a sequence of floats. Got {type(arg)}")
    if isinstance(arg, Sequence) and len(arg) != req_size:
        raise ValueError(f"If {name} is a sequence its length should be one of {req_size}. Got {len(arg)}")
    if isinstance(arg, Sequence):
        for element in arg:
            if not isinstance(element, float):
                raise ValueError(f"{name} should be a sequence of floats. Got {type(element)}")

    if isinstance(arg, float):
        arg = [float(arg), float(arg)]
    if isinstance(arg, (list, tuple)) and len(arg) == 1:
        arg = [arg[0], arg[0]]
    return arg


def _check_fill_arg(fill: Union[_FillType, Dict[Type, _FillType]]) -> None:
    if isinstance(fill, dict):
        for key, value in fill.items():
            # Check key for type
            _check_fill_arg(value)
        if isinstance(fill, defaultdict) and callable(fill.default_factory):
            default_value = fill.default_factory()
            _check_fill_arg(default_value)
    else:
        if fill is not None and not isinstance(fill, (numbers.Number, tuple, list)):
            raise TypeError("Got inappropriate fill arg, only Numbers, tuples, lists and dicts are allowed.")


T = TypeVar("T")


def _default_arg(value: T) -> T:
    return value


def _get_defaultdict(default: T) -> Dict[Any, T]:
    # This weird looking construct only exists, since `lambda`'s cannot be serialized by pickle.
    # If it were possible, we could replace this with `defaultdict(lambda: default)`
    return defaultdict(functools.partial(_default_arg, default))


def _convert_fill_arg(fill: datapoints._FillType) -> datapoints._FillTypeJIT:
    # Fill = 0 is not equivalent to None, https://github.com/pytorch/vision/issues/6517
    # So, we can't reassign fill to 0
    # if fill is None:
    #     fill = 0
    if fill is None:
        return fill

    if not isinstance(fill, (int, float)):
        fill = [float(v) for v in list(fill)]
    return fill  # type: ignore[return-value]


def _setup_fill_arg(fill: Union[_FillType, Dict[Type, _FillType]]) -> Dict[Type, _FillTypeJIT]:
    _check_fill_arg(fill)

    if isinstance(fill, dict):
        for k, v in fill.items():
            fill[k] = _convert_fill_arg(v)
        if isinstance(fill, defaultdict) and callable(fill.default_factory):
            default_value = fill.default_factory()
            sanitized_default = _convert_fill_arg(default_value)
            fill.default_factory = functools.partial(_default_arg, sanitized_default)
        return fill  # type: ignore[return-value]

    return _get_defaultdict(_convert_fill_arg(fill))


def _check_padding_arg(padding: Union[int, Sequence[int]]) -> None:
    if not isinstance(padding, (numbers.Number, tuple, list)):
        raise TypeError("Got inappropriate padding arg")

    if isinstance(padding, (tuple, list)) and len(padding) not in [1, 2, 4]:
        raise ValueError(f"Padding must be an int or a 1, 2, or 4 element tuple, not a {len(padding)} element tuple")


# TODO: let's use torchvision._utils.StrEnum to have the best of both worlds (strings and enums)
# https://github.com/pytorch/vision/issues/6250
def _check_padding_mode_arg(padding_mode: Literal["constant", "edge", "reflect", "symmetric"]) -> None:
    if padding_mode not in ["constant", "edge", "reflect", "symmetric"]:
        raise ValueError("Padding mode should be either constant, edge, reflect or symmetric")
