from __future__ import annotations

from typing import Any, Callable, List, Tuple, Type, Union

import PIL.Image
from torchvision import datapoints

from torchvision._utils import sequence_to_str
from torchvision.transforms.v2.functional import get_dimensions, get_spatial_size, is_simple_tensor


def query_bounding_box(flat_inputs: List[Any]) -> datapoints.BoundingBox:
    bounding_boxes = [inpt for inpt in flat_inputs if isinstance(inpt, datapoints.BoundingBox)]
    if not bounding_boxes:
        raise TypeError("No bounding box was found in the sample")
    elif len(bounding_boxes) > 1:
        raise ValueError("Found multiple bounding boxes in the sample")
    return bounding_boxes.pop()


def query_chw(flat_inputs: List[Any]) -> Tuple[int, int, int]:
    chws = {
        tuple(get_dimensions(inpt))
        for inpt in flat_inputs
        if isinstance(inpt, (datapoints.Image, PIL.Image.Image, datapoints.Video)) or is_simple_tensor(inpt)
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
        if isinstance(
            inpt, (datapoints.Image, PIL.Image.Image, datapoints.Video, datapoints.Mask, datapoints.BoundingBox)
        )
        or is_simple_tensor(inpt)
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
