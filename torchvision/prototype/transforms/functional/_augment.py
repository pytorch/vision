import PIL.Image

import torch
from torchvision.prototype import features
from torchvision.transforms import functional_tensor as _FT
from torchvision.transforms.functional import pil_to_tensor, to_pil_image

from ...features._image import ImageTypeJIT

erase_image_tensor = _FT.erase


@torch.jit.unused
def erase_image_pil(
    img: PIL.Image.Image, i: int, j: int, h: int, w: int, v: torch.Tensor, inplace: bool = False
) -> PIL.Image.Image:
    t_img = pil_to_tensor(img)
    output = erase_image_tensor(t_img, i=i, j=j, h=h, w=w, v=v, inplace=inplace)
    return to_pil_image(output, mode=img.mode)


def erase(
    inpt: ImageTypeJIT,
    i: int,
    j: int,
    h: int,
    w: int,
    v: torch.Tensor,
    inplace: bool = False,
) -> ImageTypeJIT:
    if isinstance(inpt, torch.Tensor):
        output = erase_image_tensor(inpt, i=i, j=j, h=h, w=w, v=v, inplace=inplace)
        if not torch.jit.is_scripting() and isinstance(inpt, features.Image):
            output = features.Image.new_like(inpt, output)
        return output
    else:  # isinstance(inpt, PIL.Image.Image):
        return erase_image_pil(inpt, i=i, j=j, h=h, w=w, v=v, inplace=inplace)
