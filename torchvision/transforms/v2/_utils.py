import collections.abc
import functools
import numbers
from collections import defaultdict
from contextlib import suppress
from typing import Any, Callable, Dict, Literal, Optional, Sequence, Type, TypeVar, Union

import torch

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


def _find_labels_default_heuristic(inputs: Any) -> torch.Tensor:
    """
    This heuristic covers three cases:

    1. The input is two-tuple whose second item is a labels tensor. This happens for already batched classification
       inputs for Mixup and Cutmix.
    2. The input is a two-tuple whose second item is a dictionary that contains the labels tensor under a label-like
       (see below) key. This happens for the inputs of detection models.
    3. The input is a dictionary that is structured as the one from 2.

    What is "label-like" key? We first search for an case-insensitive match of 'labels' inside the keys of the
    dictionary. This is the name our detection models expect. If we can't find that, we look for a case-insensitive
    match of the term 'label' anywhere inside the key, i.e. 'FooLaBeLBar'. If we can't find that either, the dictionary
    contains no "label-like" key.

    """
    if isinstance(inputs, tuple):
        inputs = inputs[1]

    # Mixup, Cutmix
    if isinstance(inputs, torch.Tensor):
        return inputs

    if not isinstance(inputs, collections.abc.Mapping):
        raise ValueError(
            f"When using the default labels_getter, the input passed to forward must be a dicstionary or a two-tuple "
            f"whose second item is a dictionary or a tensor, but got {inputs} instead."
        )

    # Tries to find a "labels" key, otherwise tries for the first key that contains "label" - case insensitive
    candidate_key = None
    with suppress(StopIteration):
        candidate_key = next(key for key in inputs.keys() if key.lower() == "labels")
    if candidate_key is None:
        with suppress(StopIteration):
            candidate_key = next(key for key in inputs.keys() if "label" in key.lower())
    if candidate_key is None:
        raise ValueError(
            "Could not infer where the labels are in the sample. Try passing a callable as the labels_getter parameter?"
            "If there are no labels in the sample by design, pass labels_getter=None."
        )

    return inputs[candidate_key]


def _parse_labels_getter(
    labels_getter: Union[str, Callable[[Any], Optional[torch.Tensor]], None]
) -> Callable[[Any], Optional[torch.Tensor]]:
    if labels_getter == "default":
        return _find_labels_default_heuristic
    elif callable(labels_getter):
        return labels_getter
    elif labels_getter is None:
        return lambda _: None
    else:
        raise ValueError(f"labels_getter should either be 'default', a callable, or None, but got {labels_getter}.")
