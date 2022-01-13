from typing import TypeVar, Union

import torch
from torchvision.prototype import features
from torchvision.prototype.transforms import ConstantParamTransform

from . import functional as F

T = TypeVar("T", bound=features.Feature)


class ConvertFormat(ConstantParamTransform):
    _DISPATCHER = F.convert_format

    def __init__(self, new_format: Union[str, features.BoundingBoxFormat]) -> None:
        super().__init__(new_format=new_format)


class ConvertDtype(ConstantParamTransform):
    _DISPATCHER = F.convert_dtype

    def __init__(self, new_dtype: torch.dtype = torch.float32) -> None:
        super().__init__(new_dtype=new_dtype)
