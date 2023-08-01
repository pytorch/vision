from __future__ import annotations

from typing import Any, Callable, List, Tuple, Type, Union

import PIL.Image
from torchvision import datapoints

from torchvision._utils import sequence_to_str
from torchvision.transforms.v2.functional import get_dimensions, get_size, is_simple_tensor


def get_bounding_boxes(flat_inputs: List[Any]) -> datapoints.BoundingBoxes:
    # This assumes there is only one bbox per sample as per the general convention
    try:
        return next(inpt for inpt in flat_inputs if isinstance(inpt, datapoints.BoundingBoxes))
    except StopIteration:
        raise ValueError("No bounding boxes were found in the sample")


def query_chw(flat_inputs: List[Any]) -> Tuple[int, int, int]:
    chws = {
        tuple(get_dimensions(inpt))
        for inpt in flat_inputs
        if check_type(inpt, (is_simple_tensor, datapoints.Image, PIL.Image.Image, datapoints.Video))
    }
    if not chws:
        raise TypeError("No image or video was found in the sample")
    elif len(chws) > 1:
        raise ValueError(f"Found multiple CxHxW dimensions in the sample: {sequence_to_str(sorted(chws))}")
    c, h, w = chws.pop()
    return c, h, w


def query_size(flat_inputs: List[Any]) -> Tuple[int, int]:
    sizes = {
        tuple(get_size(inpt))
        for inpt in flat_inputs
        if check_type(
            inpt,
            (
                is_simple_tensor,
                datapoints.Image,
                PIL.Image.Image,
                datapoints.Video,
                datapoints.Mask,
                datapoints.BoundingBoxes,
            ),
        )
    }
    if not sizes:
        raise TypeError("No image, video, mask or bounding box was found in the sample")
    elif len(sizes) > 1:
        raise ValueError(f"Found multiple HxW dimensions in the sample: {sequence_to_str(sorted(sizes))}")
    h, w = sizes.pop()
    return h, w


def check_type(obj: Any, types_or_checks: Tuple[Union[Type, Callable[[Any], bool]], ...]) -> bool:
    for type_or_check in types_or_checks:
        if isinstance(obj, type_or_check) if isinstance(type_or_check, type) else type_or_check(obj):
            return True
    return False


def has_any(flat_inputs: List[Any], *types_or_checks: Union[Type, Callable[[Any], bool]]) -> bool:
    for inpt in flat_inputs:
        if check_type(inpt, types_or_checks):
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
