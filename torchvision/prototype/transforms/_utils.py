from typing import Any, Optional, Tuple, cast, Union

import PIL.Image
import torch
from torchvision.prototype import features
from torchvision.prototype.utils._internal import query_recursively


def query_image(sample: Any) -> Union[PIL.Image.Image, torch.Tensor, features.Image]:
    def fn(input: Any) -> Optional[Union[PIL.Image.Image, torch.Tensor, features.Image]]:
        if type(input) in {torch.Tensor, features.Image} or isinstance(input, PIL.Image.Image):
            return input

        return None

    try:
        return next(query_recursively(fn, sample))
    except StopIteration:
        raise TypeError("No image was found in the sample")


def get_image_size(image: Union[PIL.Image.Image, torch.Tensor, features.Image]) -> Tuple[int, int]:
    if type(image) is torch.Tensor:
        return cast(Tuple[int, int], image.shape[-2:])
    elif isinstance(image, PIL.Image.Image):
        return image.height, image.width
    elif type(image) is features.Image:
        return image.image_size
    else:
        raise TypeError(f"unable to get image size from object of type {type(image).__name__}")


def get_image_num_channels(image: Union[PIL.Image.Image, torch.Tensor, features.Image]) -> int:
    if type(image) is torch.Tensor:
        return image.shape[-3]
    elif isinstance(image, PIL.Image.Image):
        return len(image.getbands())
    elif type(image) is features.Image:
        return image.num_channels
    else:
        raise TypeError(f"unable to get image size from object of type {type(image).__name__}")
