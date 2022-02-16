import functools
from typing import Any, List, Type, Callable, Dict

import torch
from torchvision.prototype.transforms import Transform, functional as F


class Identity(Transform):
    def _transform(self, input: Any, params: Dict[str, Any]) -> Any:
        return input


class Lambda(Transform):
    def __init__(self, fn: Callable[[Any], Any], *types: Type):
        super().__init__()
        self.fn = fn
        self.types = types

    def _transform(self, input: Any, params: Dict[str, Any]) -> Any:
        if not isinstance(input, self.types):
            return input

        return self.fn(input)

    def extra_repr(self) -> str:
        extras = []
        name = getattr(self.fn, "__name__", None)
        if name:
            extras.append(name)
        extras.append(f"types={[type.__name__ for type in self.types]}")
        return ", ".join(extras)


class Normalize(Transform):
    _DISPATCHER = F.normalize

    def __init__(self, mean: List[float], std: List[float]):
        super().__init__()
        self.mean = mean
        self.std = std

    def _get_params(self, sample: Any) -> Dict[str, Any]:
        return dict(mean=self.mean, std=self.std)


class ToDtype(Lambda):
    def __init__(self, dtype: torch.dtype, *types: Type) -> None:
        self.dtype = dtype
        super().__init__(functools.partial(torch.Tensor.to, dtype=dtype), *types)

    def extra_repr(self) -> str:
        return ", ".join([f"dtype={self.dtype}", f"types={[type.__name__ for type in self.types]}"])
