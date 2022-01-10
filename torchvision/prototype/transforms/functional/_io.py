import numpy as np
import PIL.Image
import torch
from torchvision.prototype.utils._internal import ReadOnlyTensorBuffer


def decode_image_with_pil(encoded_image: torch.Tensor) -> torch.Tensor:
    image = torch.as_tensor(np.array(PIL.Image.open(ReadOnlyTensorBuffer(encoded_image)), copy=True))
    if image.ndim == 2:
        image = image.unsqueeze(2)
    return image.permute(2, 0, 1)
