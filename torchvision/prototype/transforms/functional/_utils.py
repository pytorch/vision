from typing import Tuple, cast, Union

import PIL.Image
import torch
from torchvision.prototype import features


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
        raise TypeError(f"unable to get num channels from object of type {type(image).__name__}")
