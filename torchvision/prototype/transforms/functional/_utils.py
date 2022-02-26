from typing import Tuple, Union

import PIL.Image
import torch
from torchvision.prototype import features


def get_image_dims(image: Union[PIL.Image.Image, torch.Tensor, features.Image]) -> Tuple[int, int, int]:
    if isinstance(image, features.Image):
        channels = image.num_channels
        height, width = image.image_size
    elif isinstance(image, torch.Tensor):
        channels = 1 if image.ndim == 2 else image.shape[-3]
        height, width = image.shape[-2:]
    elif isinstance(image, PIL.Image.Image):
        channels = len(image.getbands())
        width, height = image.size
    else:
        raise TypeError(f"unable to get image dimensions from object of type {type(image).__name__}")
    return channels, height, width
