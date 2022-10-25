from typing import Union

import PIL.Image

import torch
from torchvision.prototype import features
from torchvision.transforms import functional_tensor as _FT
from torchvision.transforms.functional import pil_to_tensor, to_pil_image

erase_image_tensor = _FT.erase


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
    inpt: Union[features.ImageTypeJIT, features.VideoTypeJIT],
    i: int,
    j: int,
    h: int,
    w: int,
    v: torch.Tensor,
    inplace: bool = False,
) -> Union[features.ImageTypeJIT, features.VideoTypeJIT]:
    if isinstance(inpt, torch.Tensor) and (
        torch.jit.is_scripting() or not isinstance(inpt, (features.Image, features.Video))
    ):
        return erase_image_tensor(inpt, i=i, j=j, h=h, w=w, v=v, inplace=inplace)
    elif isinstance(inpt, features.Image):
        output = erase_image_tensor(inpt.as_subclass(torch.Tensor), i=i, j=j, h=h, w=w, v=v, inplace=inplace)
        return features.Image.wrap_like(inpt, output)
    elif isinstance(inpt, features.Video):
        output = erase_video(inpt.as_subclass(torch.Tensor), i=i, j=j, h=h, w=w, v=v, inplace=inplace)
        return features.Video.wrap_like(inpt, output)
    else:  # isinstance(inpt, PIL.Image.Image):
        return erase_image_pil(inpt, i=i, j=j, h=h, w=w, v=v, inplace=inplace)
