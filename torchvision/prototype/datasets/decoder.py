import io

import PIL.Image
import torch

from torchvision.transforms.functional import pil_to_tensor

__all__ = ["pil"]


def pil(file: io.IOBase, mode="RGB") -> torch.Tensor:
    return pil_to_tensor(PIL.Image.open(file).convert(mode.upper()))
