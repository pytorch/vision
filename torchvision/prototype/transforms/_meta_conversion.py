from typing import Any, Dict, TypeVar, Union

import torch
from torchvision.prototype import features
from torchvision.prototype.transforms import Transform

from . import functional as F

T = TypeVar("T", bound=features.Feature)


class ConvertFormat(Transform):
    def __init__(self, new_format: Union[str, features.BoundingBoxFormat]) -> None:
        super().__init__()
        self.new_format = new_format

    def get_params(self, sample: Any) -> Dict[str, Any]:
        return dict(new_format=self.new_format)

    def _dispatch(self, feature: T, **params) -> T:
        return F.convert_format(feature, **params)


class ConvertDtype(Transform):
    def __init__(self, new_dtype: torch.dtype = torch.float32) -> None:
        super().__init__()
        self.new_dtype = new_dtype

    def get_params(self, sample: Any) -> Dict[str, Any]:
        return dict(new_dtype=self.new_dtype)

    def _dispatch(self, feature: T, **params: Any) -> T:
        return F.convert_dtype(feature, **params)
