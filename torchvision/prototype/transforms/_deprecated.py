from typing import Any, Dict, Optional

import numpy as np
import PIL.Image
from torchvision.prototype import features
from torchvision.prototype.transforms import Transform
from torchvision.transforms import functional as _F

from ._utils import is_simple_tensor


# TODO: add deprecation warning
class ToTensor(Transform):
    def _transform(self, input: Any, params: Dict[str, Any]) -> Any:
        if isinstance(input, (PIL.Image.Image, np.ndarray)):
            return _F.to_tensor(input)
        else:
            return input


# TODO: add deprecation warning
class PILToTensor(Transform):
    def _transform(self, input: Any, params: Dict[str, Any]) -> Any:
        if isinstance(input, PIL.Image.Image):
            return _F.pil_to_tensor(input)
        else:
            return input


# TODO: add deprecation warning
class ToPILImage(Transform):
    def __init__(self, mode: Optional[str] = None) -> None:
        super().__init__()
        self.mode = mode

    def _transform(self, input: Any, params: Dict[str, Any]) -> Any:
        if is_simple_tensor(input) or isinstance(input, (features.Image, np.ndarray)):
            return _F.to_pil_image(input, mode=self.mode)
        else:
            return input
