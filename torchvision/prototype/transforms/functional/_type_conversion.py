import unittest.mock
from typing import Any, Dict, Tuple, Union

import numpy as np
import PIL.Image
import torch
from torch.nn.functional import one_hot
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


def label_to_one_hot(label: torch.Tensor, *, num_categories: int) -> torch.Tensor:
    return one_hot(label, num_classes=num_categories)  # type: ignore[no-any-return]


def to_image_tensor(image: Union[torch.Tensor, PIL.Image.Image, np.ndarray], copy: bool = False) -> torch.Tensor:
    if isinstance(image, torch.Tensor):
        if copy:
            return image.clone()
        else:
            return image

    return _F.to_tensor(image)


def to_image_pil(image: Union[torch.Tensor, PIL.Image.Image, np.ndarray], copy: bool = False) -> PIL.Image.Image:
    if isinstance(image, PIL.Image.Image):
        if copy:
            return image.copy()
        else:
            return image

    return _F.to_pil_image(to_image_tensor(image, copy=False))
