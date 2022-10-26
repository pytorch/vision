import functools
import numbers
from collections import defaultdict
from typing import Any, Callable, Dict, List, Sequence, Tuple, Type, TypeVar, Union

import PIL.Image

from torchvision._utils import sequence_to_str
from torchvision.prototype import features
from torchvision.prototype.features._feature import FillType, FillTypeJIT

from torchvision.prototype.transforms.functional._meta import get_dimensions, get_spatial_size
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


def _convert_fill_arg(fill: features.FillType) -> features.FillTypeJIT:
    # Fill = 0 is not equivalent to None, https://github.com/pytorch/vision/issues/6517
    # So, we can't reassign fill to 0
    # if fill is None:
    #     fill = 0
    if fill is None:
        return fill

    # This cast does Sequence -> List[float] to please mypy and torch.jit.script
    if not isinstance(fill, (int, float)):
        fill = [float(v) for v in list(fill)]
    return fill


def _setup_fill_arg(fill: Union[FillType, Dict[Type, FillType]]) -> Dict[Type, FillTypeJIT]:
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


def query_bounding_box(flat_inputs: List[Any]) -> features.BoundingBox:
    bounding_boxes = [inpt for inpt in flat_inputs if isinstance(inpt, features.BoundingBox)]
    if not bounding_boxes:
        raise TypeError("No bounding box was found in the sample")
    elif len(bounding_boxes) > 1:
        raise ValueError("Found multiple bounding boxes in the sample")
    return bounding_boxes.pop()


def query_chw(flat_inputs: List[Any]) -> Tuple[int, int, int]:
    chws = {
        tuple(get_dimensions(inpt))
        for inpt in flat_inputs
        if isinstance(inpt, (features.Image, PIL.Image.Image, features.Video)) or features.is_simple_tensor(inpt)
    }
    if not chws:
        raise TypeError("No image or video was found in the sample")
    elif len(chws) > 1:
        raise ValueError(f"Found multiple CxHxW dimensions in the sample: {sequence_to_str(sorted(chws))}")
    c, h, w = chws.pop()
    return c, h, w


def query_spatial_size(flat_inputs: List[Any]) -> Tuple[int, int]:
    sizes = {
        tuple(get_spatial_size(inpt))
        for inpt in flat_inputs
        if isinstance(inpt, (features.Image, PIL.Image.Image, features.Video, features.Mask, features.BoundingBox))
        or features.is_simple_tensor(inpt)
    }
    if not sizes:
        raise TypeError("No image, video, mask or bounding box was found in the sample")
    elif len(sizes) > 1:
        raise ValueError(f"Found multiple HxW dimensions in the sample: {sequence_to_str(sorted(sizes))}")
    h, w = sizes.pop()
    return h, w


def _isinstance(obj: Any, types_or_checks: Tuple[Union[Type, Callable[[Any], bool]], ...]) -> bool:
    for type_or_check in types_or_checks:
        if isinstance(obj, type_or_check) if isinstance(type_or_check, type) else type_or_check(obj):
            return True
    return False


def has_any(flat_inputs: List[Any], *types_or_checks: Union[Type, Callable[[Any], bool]]) -> bool:
    for inpt in flat_inputs:
        if _isinstance(inpt, types_or_checks):
            return True
    return False


def has_all(flat_inputs: List[Any], *types_or_checks: Union[Type, Callable[[Any], bool]]) -> bool:
    for type_or_check in types_or_checks:
        for inpt in flat_inputs:
            if isinstance(inpt, type_or_check) if isinstance(type_or_check, type) else type_or_check(inpt):
                break
        else:
            return False
    return True
