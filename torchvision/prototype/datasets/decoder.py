import io

import PIL.Image
import torch
from torchvision.prototype import features
from torchvision.transforms.functional import pil_to_tensor

__all__ = ["raw", "pil"]


def raw(buffer: io.IOBase) -> torch.Tensor:
    raise RuntimeError("This is just a sentinel and should never be called.")


def pil(buffer: io.IOBase) -> features.Image:
    return features.Image(pil_to_tensor(PIL.Image.open(buffer)))
