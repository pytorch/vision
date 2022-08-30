import unittest.mock
from typing import Any, Dict, Tuple, Union

import numpy as np
import PIL.Image
import torch
from torchvision.io.video import read_video
from torchvision.prototype import features
from torchvision.prototype.utils._internal import ReadOnlyTensorBuffer
from torchvision.transforms import functional as _F


def decode_image_with_pil(encoded_image: torch.Tensor) -> features.Image:
    image = torch.as_tensor(np.array(PIL.Image.open(ReadOnlyTensorBuffer(encoded_image)), copy=True))
    if image.ndim == 2:
        image = image.unsqueeze(2)
    return features.Image(image.permute(2, 0, 1))


def decode_video_with_av(encoded_video: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, Any]]:
    with unittest.mock.patch("torchvision.io.video.os.path.exists", return_value=True):
        return read_video(ReadOnlyTensorBuffer(encoded_video))  # type: ignore[arg-type]


def to_image_tensor(image: Union[torch.Tensor, PIL.Image.Image, np.ndarray]) -> features.Image:
    if isinstance(image, np.ndarray):
        output = torch.from_numpy(image)
    else:
        output = _F.pil_to_tensor(image)
    return features.Image(output)


to_image_pil = _F.to_pil_image

# We changed the names to align them with the new naming scheme. Still, `to_pil_image` and `pil_to_tensor` are
# prevalent and well understood. Thus, we just alias them without deprecating the old names.
to_pil_image = to_image_pil
pil_to_tensor = to_image_tensor

convert_image_dtype = _F.convert_image_dtype
