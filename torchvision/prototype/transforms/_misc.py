from typing import Any, TypeVar, List

from torchvision.prototype import features
from torchvision.prototype.transforms import Transform, ConstantParamTransform

from . import functional as F

T = TypeVar("T", bound=features.Feature)


class Identity(Transform):
    def supports(self, obj: Any) -> bool:
        obj_type = obj if isinstance(obj, type) else type(obj)
        return issubclass(obj_type, features.Feature) and obj_type is not features.Feature

    def _dispatch(self, feature: T, **params: Any) -> T:
        return feature


class Normalize(ConstantParamTransform):
    _DISPATCHER = F.normalize

    def __init__(self, mean: List[float], std: List[float], inplace: bool = False):
        super().__init__(mean=mean, std=std, inplace=inplace)
