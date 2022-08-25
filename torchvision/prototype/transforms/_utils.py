from typing import Any, Callable, Tuple, Type, Union

import PIL.Image
import torch
from torch.utils._pytree import tree_flatten
from torchvision.prototype import features

from .functional._meta import get_dimensions_image_pil, get_dimensions_image_tensor


def query_image(sample: Any) -> Union[PIL.Image.Image, torch.Tensor, features.Image]:
    flat_sample, _ = tree_flatten(sample)
    for i in flat_sample:
        if type(i) == torch.Tensor or isinstance(i, (PIL.Image.Image, features.Image)):
            return i

    raise TypeError("No image was found in the sample")


def query_bounding_box(sample: Any) -> features.BoundingBox:
    flat_sample, _ = tree_flatten(sample)
    for i in flat_sample:
        if isinstance(i, features.BoundingBox):
            return i

    raise TypeError("No bounding box was found in the sample")


def get_image_dimensions(image: Union[PIL.Image.Image, torch.Tensor, features.Image]) -> Tuple[int, int, int]:
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


def has_any(sample: Any, *types_or_checks: Union[Type, Callable[[Any], bool]]) -> bool:
    flat_sample, _ = tree_flatten(sample)
    for type_or_check in types_or_checks:
        for obj in flat_sample:
            if isinstance(obj, type_or_check) if isinstance(type_or_check, type) else type_or_check(obj):
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


def is_simple_tensor(inpt: Any) -> bool:
    return isinstance(inpt, torch.Tensor) and not isinstance(inpt, features._Feature)
