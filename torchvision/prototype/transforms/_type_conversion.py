from typing import Any, Dict, Optional

import numpy as np
import PIL.Image
from torchvision.prototype import features
from torchvision.prototype.transforms import Transform, functional as F

from ._utils import is_simple_tensor


class DecodeImage(Transform):
    def _transform(self, input: Any, params: Dict[str, Any]) -> Any:
        if isinstance(input, features.EncodedImage):
            output = F.decode_image_with_pil(input)
            return features.Image(output)
        else:
            return input


class LabelToOneHot(Transform):
    def __init__(self, num_categories: int = -1):
        super().__init__()
        self.num_categories = num_categories

    def _transform(self, input: Any, params: Dict[str, Any]) -> Any:
        if isinstance(input, features.Label):
            num_categories = self.num_categories
            if num_categories == -1 and input.categories is not None:
                num_categories = len(input.categories)
            output = F.label_to_one_hot(input, num_categories=num_categories)
            return features.OneHotLabel(output, categories=input.categories)
        else:
            return input

    def extra_repr(self) -> str:
        if self.num_categories == -1:
            return ""

        return f"num_categories={self.num_categories}"


class ToTensor(Transform):
    def _transform(self, input: Any, params: Dict[str, Any]) -> Any:
        if isinstance(input, (PIL.Image.Image, np.ndarray)):
            return F.to_tensor(input)
        else:
            return input


class PILToTensor(Transform):
    def _transform(self, input: Any, params: Dict[str, Any]) -> Any:
        if isinstance(input, PIL.Image.Image):
            return F.pil_to_tensor(input)
        else:
            return input


class ImageToPIL(Transform):
    def __init__(self, mode: Optional[str] = None) -> None:
        super().__init__()
        self.mode = mode

    def _transform(self, input: Any, params: Dict[str, Any]) -> Any:
        if is_simple_tensor(input) or isinstance(input, (features.Image, np.ndarray)):
            return F.image_to_pil(input, mode=self.mode)
        else:
            return input
