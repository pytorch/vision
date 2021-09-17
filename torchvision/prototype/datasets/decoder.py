import io
from typing import Callable
from typing import Dict, Any

import numpy as np
import PIL.Image
import torch

__all__ = ["decode_sample", "pil"]


def decode_sample(
    decoder: Callable[[io.IOBase], torch.Tensor], *features: str
) -> Callable[[Dict[str, Any]], Dict[str, Any]]:
    def map(sample: Dict[str, Any]) -> Dict[str, Any]:
        return {
            feature: decoder(value) if feature in features else value
            for feature, value in sample.items()
        }

    return map


def pil(file: io.IOBase, mode="RGB") -> torch.Tensor:
    image = PIL.Image.open(file).convert(mode.upper())
    return torch.from_numpy(np.array(image, copy=True)).permute((2, 0, 1))
