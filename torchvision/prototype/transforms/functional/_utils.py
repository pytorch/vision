from typing import Tuple, Union, cast

import PIL.Image
import torch
from torchvision.prototype import features
from torchvision.transforms import functional_tensor as _FT, functional_pil as _FP


def get_image_size(image: Union[PIL.Image.Image, torch.Tensor, features.Image]) -> Tuple[int, int]:
    if isinstance(image, features.Image):
        height, width = image.image_size
        return width, height
    elif isinstance(image, torch.Tensor):
        return cast(Tuple[int, int], tuple(_FT.get_image_size(image)))
    if isinstance(image, PIL.Image.Image):
        return cast(Tuple[int, int], tuple(_FP.get_image_size(image)))
    else:
        raise TypeError(f"unable to get image size from object of type {type(image).__name__}")


def get_image_num_channels(image: Union[PIL.Image.Image, torch.Tensor, features.Image]) -> int:
    if isinstance(image, features.Image):
        return image.num_channels
    elif isinstance(image, torch.Tensor):
        return _FT.get_image_num_channels(image)
    if isinstance(image, PIL.Image.Image):
        return cast(int, _FP.get_image_num_channels(image))
    else:
        raise TypeError(f"unable to get num channels from object of type {type(image).__name__}")
