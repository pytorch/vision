import functools
from typing import Any, TypeVar, List, Type, Callable, Dict

import torch
from torchvision.prototype import features
from torchvision.prototype.transforms import Transform, ConstantParamTransform

from . import functional as F
from .functional.utils import is_supported

T = TypeVar("T", bound=features.Feature)


class Identity(Transform):
    def supports(self, obj: Any) -> bool:
        return issubclass(obj if isinstance(obj, type) else type(obj), features.Feature)

    def _dispatch(self, feature: T, params: Any) -> T:
        return feature


class Lambda(Transform):
    def __init__(self, fn: Callable[[T], Any], *feature_types: Type[features.Feature]):
        super().__init__()
        self.fn = fn
        self.feature_types = feature_types or self._NATIVE_FEATURE_TYPES

    def supports(self, obj: Any) -> bool:
        return is_supported(obj, *self.feature_types)

    def _dispatch(self, feature: T, params: Dict[str, Any]) -> Any:
        return self.fn(feature)

    def _feature_types_extra_repr(self) -> str:
        return f"feature_types=[{','.join(feature_type.__name__ for feature_type in self.feature_types)}]"

    def extra_repr(self) -> str:
        name = getattr(self.fn, "__name__", None)
        return f"{name if name else ''}, {self._feature_types_extra_repr()}"


class Normalize(ConstantParamTransform):
    _DISPATCH = F.normalize

    def __init__(self, mean: List[float], std: List[float], inplace: bool = False):
        super().__init__(mean=mean, std=std, inplace=inplace)


class ToDtype(Lambda):
    def __init__(self, dtype: torch.dtype, *feature_types: Type[features.Feature]) -> None:
        self.dtype = dtype
        super().__init__(functools.partial(torch.Tensor.to, dtype=dtype), *feature_types)

    def extra_repr(self):
        return f"dtype={self.dtype}, {self._feature_types_extra_repr()}"
