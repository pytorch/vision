import warnings
from typing import Any, List, Union

import torch

from torchvision import datapoints
from torchvision.transforms import functional as _F


@torch.jit.unused
def to_tensor(inpt: Any) -> torch.Tensor:
    warnings.warn(
        "The function `to_tensor(...)` is deprecated and will be removed in a future release. "
        "Instead, please use `to_image_tensor(...)` followed by `to_dtype(..., dtype=torch.float32, scale=True)`."
    )
    return _F.to_tensor(inpt)


def get_image_size(inpt: Union[datapoints._ImageTypeJIT, datapoints._VideoTypeJIT]) -> List[int]:
    warnings.warn(
        "The function `get_image_size(...)` is deprecated and will be removed in a future release. "
        "Instead, please use `get_size(...)` which returns `[h, w]` instead of `[w, h]`."
    )
    return _F.get_image_size(inpt)
