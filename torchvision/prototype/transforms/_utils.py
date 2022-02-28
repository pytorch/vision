from typing import Any, Optional, Union, Iterator

import PIL.Image
import torch
from torchvision.prototype import features
from torchvision.prototype.utils._internal import query_recursively


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
