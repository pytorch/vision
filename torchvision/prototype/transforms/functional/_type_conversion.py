from typing import Union

import numpy as np
import PIL.Image
import torch
from torchvision.prototype import features
from torchvision.transforms import functional as _F


@torch.jit.unused
def to_image_tensor(image: Union[torch.Tensor, PIL.Image.Image, np.ndarray]) -> features.Image:
    if isinstance(image, np.ndarray):
        output = torch.from_numpy(image).permute((2, 0, 1)).contiguous()
    elif isinstance(image, PIL.Image.Image):
        output = pil_to_tensor(image)
    else:  # isinstance(inpt, torch.Tensor):
        output = image
    return features.Image(output)


to_image_pil = _F.to_pil_image
pil_to_tensor = _F.pil_to_tensor

# We changed the names to align them with the new naming scheme. Still, `to_pil_image` is
# prevalent and well understood. Thus, we just alias it without deprecating the old name.
to_pil_image = to_image_pil
