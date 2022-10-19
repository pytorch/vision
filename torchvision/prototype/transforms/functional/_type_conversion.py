import unittest.mock
from typing import Any, Dict, Tuple, Union

import numpy as np
import PIL.Image
import torch
from torchvision.io.video import read_video
from torchvision.prototype import features
from torchvision.prototype.utils._internal import ReadOnlyTensorBuffer
from torchvision.transforms import functional as _F


@torch.jit.unused
def decode_image_with_pil(encoded_image: torch.Tensor) -> features.Image:
    image = torch.as_tensor(np.array(PIL.Image.open(ReadOnlyTensorBuffer(encoded_image)), copy=True))
    if image.ndim == 2:
        image = image.unsqueeze(2)
    return features.Image(image.permute(2, 0, 1))


@torch.jit.unused
def decode_video_with_av(encoded_video: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, Any]]:
    with unittest.mock.patch("torchvision.io.video.os.path.exists", return_value=True):
        return read_video(ReadOnlyTensorBuffer(encoded_video))  # type: ignore[arg-type]


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

convert_image_dtype = _F.convert_image_dtype
