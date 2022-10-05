import numbers
from collections import defaultdict

from typing import Any, Callable, Dict, Sequence, Tuple, Type, Union

import PIL.Image

from torch.utils._pytree import tree_flatten
from torchvision._utils import sequence_to_str
from torchvision.prototype import features
from torchvision.prototype.features._feature import FillType

from torchvision.prototype.transforms.functional._meta import get_chw
from torchvision.transforms.transforms import _check_sequence_input, _setup_angle, _setup_size  # noqa: F401

from typing_extensions import Literal


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


def _check_fill_arg(fill: Union[FillType, Dict[Type, FillType]]) -> None:
    if isinstance(fill, dict):
        for key, value in fill.items():
            # Check key for type
            _check_fill_arg(value)
    else:
        if fill is not None and not isinstance(fill, (numbers.Number, tuple, list)):
            raise TypeError("Got inappropriate fill arg")


def _setup_fill_arg(fill: Union[FillType, Dict[Type, FillType]]) -> Dict[Type, FillType]:
    _check_fill_arg(fill)

    if isinstance(fill, dict):
        return fill

    return defaultdict(lambda: fill)  # type: ignore[return-value, arg-type]


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


def query_bounding_box(sample: Any) -> features.BoundingBox:
    flat_sample, _ = tree_flatten(sample)
    bounding_boxes = {item for item in flat_sample if isinstance(item, features.BoundingBox)}
    if not bounding_boxes:
        raise TypeError("No bounding box was found in the sample")
    elif len(bounding_boxes) > 1:
        raise ValueError("Found multiple bounding boxes in the sample")
    return bounding_boxes.pop()


def query_chw(sample: Any) -> Tuple[int, int, int]:
    flat_sample, _ = tree_flatten(sample)
    chws = {
        get_chw(item)
        for item in flat_sample
        if isinstance(item, (features.Image, PIL.Image.Image)) or features.is_simple_tensor(item)
    }
    if not chws:
        raise TypeError("No image was found in the sample")
    elif len(chws) > 1:
        raise ValueError(f"Found multiple CxHxW dimensions in the sample: {sequence_to_str(sorted(chws))}")
    return chws.pop()


def _isinstance(obj: Any, types_or_checks: Tuple[Union[Type, Callable[[Any], bool]], ...]) -> bool:
    for type_or_check in types_or_checks:
        if isinstance(obj, type_or_check) if isinstance(type_or_check, type) else type_or_check(obj):
            return True
    return False


def has_any(sample: Any, *types_or_checks: Union[Type, Callable[[Any], bool]]) -> bool:
    flat_sample, _ = tree_flatten(sample)
    for obj in flat_sample:
        if _isinstance(obj, types_or_checks):
            return True
    return False


def has_all(sample: Any, *types_or_checks: Union[Type, Callable[[Any], bool]]) -> bool:
    flat_sample, _ = tree_flatten(sample)
    for type_or_check in types_or_checks:
        for obj in flat_sample:
            if isinstance(obj, type_or_check) if isinstance(type_or_check, type) else type_or_check(obj):
                break
        else:
            return False
    return True
