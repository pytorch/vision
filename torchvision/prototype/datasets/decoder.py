import io

import numpy as np
import PIL.Image
import torch

__all__ = ["pil"]


def pil(path: str, file: io.BufferedIOBase, mode="RGB") -> torch.Tensor:
    image = PIL.Image.open(file).convert(mode.upper())
    return torch.from_numpy(np.array(image, copy=True)).permute((2, 0, 1))
