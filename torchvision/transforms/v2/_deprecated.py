import warnings
from typing import Any, Dict, Union

import numpy as np
import PIL.Image
import torch
from torchvision.transforms import functional as _F

from torchvision.transforms.v2 import Transform


class ToTensor(Transform):
    _transformed_types = (PIL.Image.Image, np.ndarray)

    def __init__(self) -> None:
        warnings.warn(
            "The transform `ToTensor()` is deprecated and will be removed in a future release. "
            "Instead, please use `transforms.Compose([transforms.ToImageTensor(), transforms.ConvertImageDtype()])`."
        )
        super().__init__()

    def _transform(self, inpt: Union[PIL.Image.Image, np.ndarray], params: Dict[str, Any]) -> torch.Tensor:
        return _F.to_tensor(inpt)
