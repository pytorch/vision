from typing import Union

import numpy as np
import PIL.Image
import torch
from torchvision import tv_tensors
from torchvision.io import read_image
from torchvision.transforms import functional as _F


@torch.jit.unused
def to_image(inpt: Union[torch.Tensor, PIL.Image.Image, np.ndarray, str]) -> tv_tensors.Image:
    """See :class:`~torchvision.transforms.v2.ToImage` for details."""
    if isinstance(inpt, np.ndarray):
        output = torch.from_numpy(np.atleast_3d(inpt)).permute((2, 0, 1)).contiguous()
    elif isinstance(inpt, PIL.Image.Image):
        output = pil_to_tensor(inpt)
    elif isinstance(inpt, torch.Tensor):
        output = inpt
    elif isinstance(inpt, str):
        output = read_image(inpt)
    else:
        raise TypeError(
            f"Input can either be a pure Tensor, a numpy array, a PIL image, "
            f"or a string representing image path, but got {type(inpt)} instead."
        )
    return tv_tensors.Image(output)


to_pil_image = _F.to_pil_image
pil_to_tensor = _F.pil_to_tensor
