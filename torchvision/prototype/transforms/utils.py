from typing import Callable, Tuple, TypeVar
from typing import Optional, Any

import torch
from torchvision.prototype import features
from torchvision.prototype.features import BoundingBox, Image
from torchvision.prototype.utils._internal import query_recursively

T = TypeVar("T")


class Query:
    def __init__(self, sample: Any) -> None:
        self.sample = sample

    def query(self, fn: Callable[[Any], Optional[T]]) -> T:
        try:
            return next(query_recursively(fn, self.sample))
        except StopIteration:
            raise RuntimeError from None

    def image(self):
        def fn(sample: Any) -> Optional[Image]:
            if isinstance(sample, Image):
                return sample
            else:
                return None

        return self.query(fn)

    def image_size(self) -> Tuple[int, int]:
        def fn(sample: Any) -> Optional[Tuple[int, int]]:
            if isinstance(sample, (Image, BoundingBox)):
                return sample.image_size
            else:
                return None

        return self.query(fn)

    def image_for_size_extraction(self):
        def fn(sample: Any) -> Optional[torch.Tensor]:
            if isinstance(sample, Image):
                return sample
            elif isinstance(sample, features.BoundingBox):
                return torch.empty(sample.image_size)
            else:
                return None

        return self.query(fn)

    def image_for_size_and_channels_extraction(self):
        def fn(sample: Any) -> Optional[torch.Tensor]:
            if isinstance(sample, Image):
                return sample
            else:
                return None

        return self.query(fn)
