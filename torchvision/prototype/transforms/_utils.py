from typing import Any, Optional, Tuple, Union

import PIL.Image
import torch
from torchvision.prototype import features
from torchvision.prototype.utils._internal import query_recursively
from torchvision.transforms import functional_tensor as _FT, functional_pil as _FP


def query_image(sample: Any) -> Union[PIL.Image.Image, torch.Tensor, features.Image]:
    def fn(input: Any) -> Optional[Union[PIL.Image.Image, torch.Tensor, features.Image]]:
        if type(input) in {torch.Tensor, features.Image} or isinstance(input, PIL.Image.Image):
            return input

        return None

    try:
        return next(query_recursively(fn, sample))
    except StopIteration:
        raise TypeError("No image was found in the sample")


def get_image_dimensions(image: Union[PIL.Image.Image, torch.Tensor, features.Image]) -> Tuple[int, int, int]:
    if isinstance(image, features.Image):
        channels = image.num_channels
        height, width = image.image_size
    elif isinstance(image, torch.Tensor):
        channels, height, width = _FT.get_dimensions(image)
    elif isinstance(image, PIL.Image.Image):
        channels, height, width = _FP.get_dimensions(image)
    else:
        raise TypeError(f"unable to get image dimensions from object of type {type(image).__name__}")
    return channels, height, width
