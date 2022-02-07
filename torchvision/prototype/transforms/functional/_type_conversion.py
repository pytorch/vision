import numpy as np
import PIL.Image
import torch
from torch.nn.functional import one_hot
from torchvision.prototype.utils._internal import ReadOnlyTensorBuffer


def decode_image_with_pil(encoded_image: torch.Tensor) -> torch.Tensor:
    image = torch.as_tensor(np.array(PIL.Image.open(ReadOnlyTensorBuffer(encoded_image)), copy=True))
    if image.ndim == 2:
        image = image.unsqueeze(2)
    return image.permute(2, 0, 1)


def label_to_one_hot(label: torch.Tensor, *, num_categories: int) -> torch.Tensor:
    return one_hot(label, num_classes=num_categories)
