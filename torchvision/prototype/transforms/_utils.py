from typing import Any, Callable, Tuple, Type, Union

import PIL.Image
import torch
from torch.utils._pytree import tree_flatten
from torchvision._utils import sequence_to_str
from torchvision.prototype import features

from .functional._meta import get_dimensions_image_pil, get_dimensions_image_tensor


def query_bounding_box(sample: Any) -> features.BoundingBox:
    flat_sample, _ = tree_flatten(sample)
    for i in flat_sample:
        if isinstance(i, features.BoundingBox):
            return i

    raise TypeError("No bounding box was found in the sample")


def get_chw(image: Union[PIL.Image.Image, torch.Tensor, features.Image]) -> Tuple[int, int, int]:
    if isinstance(image, features.Image):
        channels = image.num_channels
        height, width = image.image_size
    elif isinstance(image, torch.Tensor):
        channels, height, width = get_dimensions_image_tensor(image)
    elif isinstance(image, PIL.Image.Image):
        channels, height, width = get_dimensions_image_pil(image)
    else:
        raise TypeError(f"unable to get image dimensions from object of type {type(image).__name__}")
    return channels, height, width


def query_chw(sample: Any) -> Tuple[int, int, int]:
    flat_sample, _ = tree_flatten(sample)
    chws = {
        get_chw(item)
        for item in flat_sample
        if isinstance(item, (features.Image, PIL.Image.Image)) or is_simple_tensor(item)
    }
    if not chws:
        raise TypeError("No image was found in the sample")
    elif len(chws) > 2:
        raise TypeError(f"Found multiple CxHxW dimensions in the sample: {sequence_to_str(sorted(chws))}")
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


# TODO: Given that this is not related to pytree / the Transform object, we should probably move it to somewhere else.
#  One possibility is `functional._utils` so both the functionals and the transforms have proper access to it. We could
#  also move it `features` since it literally checks for the _Feature type.
def is_simple_tensor(inpt: Any) -> bool:
    return isinstance(inpt, torch.Tensor) and not isinstance(inpt, features._Feature)
