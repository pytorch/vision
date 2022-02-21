from typing import Any, Optional, Union

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
