from typing import Any, Optional, Tuple, Union, Iterator

import PIL.Image
import torch
from torchvision.prototype import features
from torchvision.prototype.utils._internal import query_recursively

from .functional._meta import get_dimensions_image_tensor, get_dimensions_image_pil


def _extract_image(input: Any) -> Optional[Union[PIL.Image.Image, torch.Tensor, features.Image]]:
    if type(input) in {torch.Tensor, features.Image} or isinstance(input, PIL.Image.Image):
        return input

    return None


def query_images(sample: Any) -> Iterator[Union[PIL.Image.Image, torch.Tensor, features.Image]]:
    return query_recursively(_extract_image, sample)


def query_image(sample: Any) -> Union[PIL.Image.Image, torch.Tensor, features.Image]:
    try:
        return next(query_images(sample))
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
