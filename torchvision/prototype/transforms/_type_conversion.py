from typing import Any, Dict, Optional

import numpy as np
import PIL.Image

from torch.nn.functional import one_hot
from torchvision.prototype import features
from torchvision.prototype.transforms import functional as F, Transform


class DecodeImage(Transform):
    _transformed_types = (features.EncodedImage,)

    def _transform(self, inpt: Any, params: Dict[str, Any]) -> features.Image:
        return F.decode_image_with_pil(inpt)


class LabelToOneHot(Transform):
    _transformed_types = (features.Label,)

    def __init__(self, num_categories: int = -1):
        super().__init__()
        self.num_categories = num_categories

    def _transform(self, inpt: features.Label, params: Dict[str, Any]) -> features.OneHotLabel:
        num_categories = self.num_categories
        if num_categories == -1 and inpt.categories is not None:
            num_categories = len(inpt.categories)
        output = one_hot(inpt, num_classes=num_categories)
        return features.OneHotLabel(output, categories=inpt.categories)

    def extra_repr(self) -> str:
        if self.num_categories == -1:
            return ""

        return f"num_categories={self.num_categories}"


class ToImageTensor(Transform):
    _transformed_types = (features.is_simple_tensor, PIL.Image.Image, np.ndarray)

    def _transform(self, inpt: Any, params: Dict[str, Any]) -> features.Image:
        return F.to_image_tensor(inpt)


class ToImagePIL(Transform):
    _transformed_types = (features.is_simple_tensor, features.Image, np.ndarray)

    def __init__(self, *, mode: Optional[str] = None) -> None:
        super().__init__()
        self.mode = mode

    def _transform(self, inpt: Any, params: Dict[str, Any]) -> PIL.Image.Image:
        return F.to_image_pil(inpt, mode=self.mode)


# We changed the names to align them with the new naming scheme. Still, `PILToTensor` and `ToPILImage` are
# prevalent and well understood. Thus, we just alias them without deprecating the old names.
PILToTensor = ToImageTensor
ToPILImage = ToImagePIL
