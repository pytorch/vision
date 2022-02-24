from typing import Tuple, Union

import PIL.Image
import torch
from torchvision.prototype import features
from torchvision.transforms import functional as _F

# FIXME


def get_image_size(image: Union[PIL.Image.Image, torch.Tensor, features.Image]) -> Tuple[int, int]:
    if type(image) is torch.Tensor or isinstance(image, PIL.Image.Image):
        width, height = _F.get_image_size(image)
        return height, width
    elif type(image) is features.Image:
        return image.image_size
    else:
        raise TypeError(f"unable to get image size from object of type {type(image).__name__}")


def get_image_num_channels(image: Union[PIL.Image.Image, torch.Tensor, features.Image]) -> int:
    if type(image) is torch.Tensor or isinstance(image, PIL.Image.Image):
        return _F.get_image_num_channels(image)
    elif type(image) is features.Image:
        return image.num_channels
    else:
        raise TypeError(f"unable to get num channels from object of type {type(image).__name__}")
