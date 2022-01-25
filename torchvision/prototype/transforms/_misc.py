import functools
from typing import Any, List, Type, Callable, Dict

import torch
from torchvision.prototype import features
from torchvision.prototype.transforms import Transform, ConstantParamTransform

from . import functional as F


class Identity(Transform):
    def _supports(self, obj: Any) -> bool:
        return True

    def _dispatch(self, input: Any, params: Any) -> Any:
        return input


class Lambda(Transform):
    def __init__(self, fn: Callable[[Any], Any], *feature_types: Type[features._Feature]):
        super().__init__()
        self.fn = fn
        self.feature_types = feature_types

    def _supports(self, obj: Any) -> bool:
        return type(obj) in self.feature_types

    def _dispatch(self, input: Any, params: Dict[str, Any]) -> Any:
        return self.fn(input)

    def _feature_types_extra_repr(self) -> str:
        return f"feature_types=[{','.join(feature_type.__name__ for feature_type in self.feature_types)}]"

    def extra_repr(self) -> str:
        name = getattr(self.fn, "__name__", None)
        return f"{name if name else ''}, {self._feature_types_extra_repr()}"


class Normalize(ConstantParamTransform):
    _DISPATCHER = F.normalize

    def __init__(self, mean: List[float], std: List[float]):
        super().__init__(mean=mean, std=std)


class ToDtype(Lambda):
    def __init__(self, dtype: torch.dtype, *feature_types: Type[features._Feature]) -> None:
        self.dtype = dtype
        super().__init__(functools.partial(torch.Tensor.to, dtype=dtype), *feature_types)

    def extra_repr(self) -> str:
        return f"dtype={self.dtype}, {self._feature_types_extra_repr()}"
