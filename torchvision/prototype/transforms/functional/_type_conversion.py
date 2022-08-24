import unittest.mock
from typing import Any, Dict, Optional, Tuple, Union

import numpy as np
import PIL.Image
import torch
from torchvision.io.video import read_video
from torchvision.prototype.utils._internal import ReadOnlyTensorBuffer
from torchvision.transforms import functional as _F


def decode_image_with_pil(encoded_image: torch.Tensor) -> torch.Tensor:
    image = torch.as_tensor(np.array(PIL.Image.open(ReadOnlyTensorBuffer(encoded_image)), copy=True))
    if image.ndim == 2:
        image = image.unsqueeze(2)
    return image.permute(2, 0, 1)


def decode_video_with_av(encoded_video: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, Any]]:
    with unittest.mock.patch("torchvision.io.video.os.path.exists", return_value=True):
        return read_video(ReadOnlyTensorBuffer(encoded_video))  # type: ignore[arg-type]


def to_image_tensor(image: Union[torch.Tensor, PIL.Image.Image, np.ndarray], copy: bool = False) -> torch.Tensor:
    if isinstance(image, np.ndarray):
        image = torch.from_numpy(image)

    if isinstance(image, torch.Tensor):
        if copy:
            return image.clone()
        else:
            return image

    return _F.pil_to_tensor(image)


def to_image_pil(
    image: Union[torch.Tensor, PIL.Image.Image, np.ndarray], mode: Optional[str] = None
) -> PIL.Image.Image:
    if isinstance(image, PIL.Image.Image):
        if mode != image.mode:
            return image.convert(mode)
        else:
            return image

    return _F.to_pil_image(image, mode=mode)
