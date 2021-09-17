import collections.abc
import io
from typing import Callable

import numpy as np
import PIL.Image
import torch
from torch.utils.data import IterDataPipe, functional_datapipe

__all__ = ["pil"]


@functional_datapipe("decode_sample")
class SampleDecoder(IterDataPipe):
    def __init__(
        self,
        datapipe: IterDataPipe,
        fn: Callable[[io.IOBase], torch.Tensor],
        *features: str,
    ) -> None:
        self.datapipe = datapipe
        self.fn = fn
        self.features = features

    def __iter__(self):
        for sample in self.datapipe:
            yield {
                feature: self.fn(value) if feature in self.features else value
                for feature, value in sample.items()
            }

    def __len__(self) -> int:
        if isinstance(self.datapipe, collections.abc.Sized):
            return len(self.datapipe)
        raise TypeError(f"{type(self).__name__} instance doesn't have valid length")


def pil(file: io.IOBase, mode="RGB") -> torch.Tensor:
    image = PIL.Image.open(file).convert(mode.upper())
    return torch.from_numpy(np.array(image, copy=True)).permute((2, 0, 1))
