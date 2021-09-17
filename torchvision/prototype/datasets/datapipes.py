"""This is a temporary module for datapipes that are needed now, but should be absorbed by torchdata in the future."""

import random
from typing import Any

from torch.utils.data import IterDataPipe


class RandomPicker(IterDataPipe):
    def __init__(self, *datapipes: IterDataPipe) -> None:
        self.datapipes = datapipes

    def __iter__(self) -> Any:
        non_exhausted = [iter(dp) for dp in self.datapipes]
        while non_exhausted:
            dp = random.choice(non_exhausted)
            try:
                yield next(dp)
            except StopIteration:
                non_exhausted.remove(dp)
