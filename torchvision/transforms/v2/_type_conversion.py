from typing import Any, Dict, Optional, Union

import numpy as np
import PIL.Image
import torch

from torchvision import datapoints
from torchvision.transforms.v2 import functional as F, Transform

from torchvision.transforms.v2.utils import is_simple_tensor


class PILToTensor(Transform):
    _transformed_types = (PIL.Image.Image,)

    def _transform(self, inpt: PIL.Image.Image, params: Dict[str, Any]) -> torch.Tensor:
        return F.pil_to_tensor(inpt)


class ToImageTensor(Transform):
    _transformed_types = (is_simple_tensor, PIL.Image.Image, np.ndarray)

    def _transform(
        self, inpt: Union[torch.Tensor, PIL.Image.Image, np.ndarray], params: Dict[str, Any]
    ) -> datapoints.Image:
        return F.to_image_tensor(inpt)


class ToImagePIL(Transform):
    _transformed_types = (is_simple_tensor, datapoints.Image, np.ndarray)

    def __init__(self, mode: Optional[str] = None) -> None:
        super().__init__()
        self.mode = mode

    def _transform(
        self, inpt: Union[torch.Tensor, PIL.Image.Image, np.ndarray], params: Dict[str, Any]
    ) -> PIL.Image.Image:
        return F.to_image_pil(inpt, mode=self.mode)


# We changed the name to align them with the new naming scheme. Still, `ToPILImage` is
# prevalent and well understood. Thus, we just alias it without deprecating the old name.
ToPILImage = ToImagePIL
