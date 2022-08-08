from typing import Any, Dict

import numpy as np
import PIL.Image
from torchvision.prototype import features
from torchvision.prototype.transforms import functional as F, Transform

from ._utils import is_simple_tensor


class DecodeImage(Transform):
    def _transform(self, inpt: Any, params: Dict[str, Any]) -> Any:
        if isinstance(inpt, features.EncodedImage):
            output = F.decode_image_with_pil(inpt)
            return features.Image(output)
        else:
            return inpt


class LabelToOneHot(Transform):
    def __init__(self, num_categories: int = -1):
        super().__init__()
        self.num_categories = num_categories

    def _transform(self, inpt: Any, params: Dict[str, Any]) -> Any:
        if isinstance(inpt, features.Label):
            num_categories = self.num_categories
            if num_categories == -1 and inpt.categories is not None:
                num_categories = len(inpt.categories)
            output = F.label_to_one_hot(inpt, num_categories=num_categories)
            return features.OneHotLabel(output, categories=inpt.categories)
        else:
            return inpt

    def extra_repr(self) -> str:
        if self.num_categories == -1:
            return ""

        return f"num_categories={self.num_categories}"


class ToImageTensor(Transform):
    def __init__(self, *, copy: bool = False) -> None:
        super().__init__()
        self.copy = copy

    def _transform(self, inpt: Any, params: Dict[str, Any]) -> Any:
        if isinstance(inpt, (features.Image, PIL.Image.Image, np.ndarray)) or is_simple_tensor(inpt):
            output = F.to_image_tensor(inpt, copy=self.copy)
            return features.Image(output)
        else:
            return inpt


class ToImagePIL(Transform):
    def __init__(self, *, copy: bool = False) -> None:
        super().__init__()
        self.copy = copy

    def _transform(self, inpt: Any, params: Dict[str, Any]) -> Any:
        if isinstance(inpt, (features.Image, PIL.Image.Image, np.ndarray)) or is_simple_tensor(inpt):
            return F.to_image_pil(inpt, copy=self.copy)
        else:
            return inpt
