from typing import Any, Iterator, Optional, Tuple, Type, Union

import PIL.Image
import torch
from torchvision.prototype import features
from torchvision.prototype.utils._internal import query_recursively

from .functional._meta import get_dimensions_image_pil, get_dimensions_image_tensor


def query_image(sample: Any) -> Union[PIL.Image.Image, torch.Tensor, features.Image]:
    def fn(
        id: Tuple[Any, ...], input: Any
    ) -> Optional[Tuple[Tuple[Any, ...], Union[PIL.Image.Image, torch.Tensor, features.Image]]]:
        if type(input) in {torch.Tensor, features.Image} or isinstance(input, PIL.Image.Image):
            return id, input

        return None

    try:
        return next(query_recursively(fn, sample))[1]
    except StopIteration:
        raise TypeError("No image was found in the sample")


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


def _extract_types(sample: Any) -> Iterator[Type]:
    return query_recursively(lambda id, input: type(input), sample)


def has_any(sample: Any, *types: Type) -> bool:
    return any(issubclass(type, types) for type in _extract_types(sample))


def has_all(sample: Any, *types: Type) -> bool:
    return not bool(set(types) - set(_extract_types(sample)))


def is_simple_tensor(input: Any) -> bool:
    return isinstance(input, torch.Tensor) and not isinstance(input, features._Feature)
