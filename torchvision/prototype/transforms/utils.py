import functools
from typing import Callable, Tuple, TypeVar, Optional, Any, cast, Dict

import PIL.Image
import torch
from torchvision.prototype import features, transforms
from torchvision.prototype.utils._internal import query_recursively

T = TypeVar("T")


def legacy_transform(kernel: Callable, *attrs: str) -> Callable[[Callable], Callable]:
    def outer_wrapper(fn: Callable) -> Any:
        @functools.wraps(fn)
        def inner_wrapper(self: transforms.Transform, input: Any, params: Dict[str, Any]) -> Any:
            if type(input) is torch.Tensor or isinstance(input, PIL.Image.Image):
                return kernel(input, **params, **{attr: getattr(self, attr) for attr in attrs})

            return fn(self, input, params)

        return inner_wrapper

    return outer_wrapper


class Query:
    def __init__(self, sample: Any) -> None:
        self.sample = sample

    def query(self, fn: Callable[[Any], Optional[T]]) -> T:
        try:
            return next(query_recursively(fn, self.sample))
        except StopIteration:
            raise RuntimeError from None

    def image(self) -> features.Image:
        def fn(sample: Any) -> Optional[features.Image]:
            if isinstance(sample, features.Image):
                return sample
            else:
                return None

        return self.query(fn)

    def image_size(self) -> Tuple[int, int]:
        def fn(sample: Any) -> Optional[Tuple[int, int]]:
            if isinstance(sample, (features.Image, features.BoundingBox)):
                return sample.image_size
            elif isinstance(sample, torch.Tensor):
                return cast(Tuple[int, int], sample.shape[-2:])
            elif isinstance(sample, PIL.Image.Image):
                return sample.height, sample.width
            else:
                return None

        return self.query(fn)

    def image_for_size_extraction(self) -> features.Image:
        def fn(sample: Any) -> Optional[features.Image]:
            if isinstance(sample, features.Image):
                return sample

            if isinstance(sample, features.BoundingBox):
                height, width = sample.image_size
            elif isinstance(sample, torch.Tensor):
                height, width = sample.shape[-2:]
            elif isinstance(sample, PIL.Image.Image):
                height, width = sample.height, sample.width
            else:
                return None

            return features.Image(torch.empty(0, height, width))

        return self.query(fn)

    def image_for_size_and_channels_extraction(self) -> features.Image:
        def fn(sample: Any) -> Optional[features.Image]:
            if isinstance(sample, features.Image):
                return sample

            if isinstance(sample, torch.Tensor):
                num_channels, height, width = sample.shape[-3:]
            elif isinstance(sample, PIL.Image.Image):
                height, width = sample.height, sample.width
                num_channels = len(sample.num_bands())
            else:
                return None

            return features.Image(torch.empty(0, num_channels, height, width))

        return self.query(fn)
