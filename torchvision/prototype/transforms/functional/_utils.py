from typing import List, Tuple, Union

import PIL.Image
import torch
from torchvision.prototype import features
from torchvision.transforms import functional_pil as _FP, functional_tensor as _FT

get_dimensions_image_tensor = _FT.get_dimensions
get_dimensions_image_pil = _FP.get_dimensions


def get_chw(image: Union[PIL.Image.Image, torch.Tensor, features.Image]) -> Tuple[int, int, int]:
    if isinstance(image, features.Image):
        channels = image.num_channels
        height, width = image.image_size
    elif features.is_simple_tensor(image):
        channels, height, width = get_dimensions_image_tensor(image)
    elif isinstance(image, PIL.Image.Image):
        channels, height, width = get_dimensions_image_pil(image)
    else:
        raise TypeError(f"unable to get image dimensions from object of type {type(image).__name__}")
    return channels, height, width


# The three functions below are here for BC. Whether we want to have two different kernels and how they and the
# compound version should be named is still under discussion: https://github.com/pytorch/vision/issues/6491
# Given that these kernels should also support boxes, masks, and videos, it is unlikely that there name will stay.
# They will either be deprecated or simply aliased to the new kernels if we have reached consensus about the issue
# detailed above.


def get_dimensions(image: Union[PIL.Image.Image, torch.Tensor, features.Image]) -> List[int]:
    return list(get_chw(image))


def get_image_num_channels(image: Union[PIL.Image.Image, torch.Tensor, features.Image]) -> int:
    num_channels, *_ = get_chw(image)
    return num_channels


def get_image_size(image: Union[PIL.Image.Image, torch.Tensor, features.Image]) -> List[int]:
    _, *image_size = get_chw(image)
    return image_size
