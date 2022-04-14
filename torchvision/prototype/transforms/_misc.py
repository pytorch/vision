import functools
from typing import Any, Callable, Dict, List, Type

import torch
from torchvision.prototype.transforms import functional as F, Transform


class Identity(Transform):
    def _transform(self, input: Any, params: Dict[str, Any]) -> Any:
        return input


class Lambda(Transform):
    def __init__(self, fn: Callable[[Any], Any], *types: Type):
        super().__init__()
        self.fn = fn
        self.types = types

    def _transform(self, input: Any, params: Dict[str, Any]) -> Any:
        if type(input) in self.types:
            return self.fn(input)
        else:
            return input

    def extra_repr(self) -> str:
        extras = []
        name = getattr(self.fn, "__name__", None)
        if name:
            extras.append(name)
        extras.append(f"types={[type.__name__ for type in self.types]}")
        return ", ".join(extras)


class Normalize(Transform):
    def __init__(self, mean: List[float], std: List[float]):
        super().__init__()
        self.mean = mean
        self.std = std

    def _transform(self, input: Any, params: Dict[str, Any]) -> Any:
        if isinstance(input, torch.Tensor):
            # We don't need to differentiate between vanilla tensors and features.Image's here, since the result of the
            # normalization transform is no longer a features.Image
            return F.normalize_image_tensor(input, mean=self.mean, std=self.std)
        else:
            return input


class ToDtype(Lambda):
    def __init__(self, dtype: torch.dtype, *types: Type) -> None:
        self.dtype = dtype
        super().__init__(functools.partial(torch.Tensor.to, dtype=dtype), *types)

    def extra_repr(self) -> str:
        return ", ".join([f"dtype={self.dtype}", f"types={[type.__name__ for type in self.types]}"])
