from typing import Union

import PIL.Image

import torch
from torchvision import datapoints
from torchvision.transforms.functional import pil_to_tensor, to_pil_image
from torchvision.utils import _log_api_usage_once

from ._utils import is_simple_tensor


def erase_image_tensor(
    image: torch.Tensor, i: int, j: int, h: int, w: int, v: torch.Tensor, inplace: bool = False
) -> torch.Tensor:
    if not inplace:
        image = image.clone()

    image[..., i : i + h, j : j + w] = v
    return image


@torch.jit.unused
def erase_image_pil(
    image: PIL.Image.Image, i: int, j: int, h: int, w: int, v: torch.Tensor, inplace: bool = False
) -> PIL.Image.Image:
    t_img = pil_to_tensor(image)
    output = erase_image_tensor(t_img, i=i, j=j, h=h, w=w, v=v, inplace=inplace)
    return to_pil_image(output, mode=image.mode)


def erase_video(
    video: torch.Tensor, i: int, j: int, h: int, w: int, v: torch.Tensor, inplace: bool = False
) -> torch.Tensor:
    return erase_image_tensor(video, i=i, j=j, h=h, w=w, v=v, inplace=inplace)


def erase(
    inpt: Union[datapoints._ImageTypeJIT, datapoints._VideoTypeJIT],
    i: int,
    j: int,
    h: int,
    w: int,
    v: torch.Tensor,
    inplace: bool = False,
) -> Union[datapoints._ImageTypeJIT, datapoints._VideoTypeJIT]:
    if not torch.jit.is_scripting():
        _log_api_usage_once(erase)

    if torch.jit.is_scripting() or is_simple_tensor(inpt):
        return erase_image_tensor(inpt, i=i, j=j, h=h, w=w, v=v, inplace=inplace)
    elif isinstance(inpt, datapoints.Image):
        output = erase_image_tensor(inpt.as_subclass(torch.Tensor), i=i, j=j, h=h, w=w, v=v, inplace=inplace)
        return datapoints.Image.wrap_like(inpt, output)
    elif isinstance(inpt, datapoints.Video):
        output = erase_video(inpt.as_subclass(torch.Tensor), i=i, j=j, h=h, w=w, v=v, inplace=inplace)
        return datapoints.Video.wrap_like(inpt, output)
    elif isinstance(inpt, PIL.Image.Image):
        return erase_image_pil(inpt, i=i, j=j, h=h, w=w, v=v, inplace=inplace)
    else:
        raise TypeError(
            f"Input can either be a plain tensor, an `Image` or `Video` datapoint, or a PIL image, "
            f"but got {type(inpt)} instead."
        )
