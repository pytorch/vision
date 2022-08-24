from typing import Any, Dict, Optional

import numpy as np
import PIL.Image

import torch
from torchvision.prototype import features
from torchvision.prototype.transforms import functional as F, Transform

from ._utils import is_simple_tensor


class DecodeImage(Transform):
    def _transform(self, inpt: Any, params: Dict[str, Any]) -> Any:
        # datumbox: We shouldn't have if/elses here. We could just set:
        # `_transformed_types = (features.EncodedImage,)`
        # and eliminate them.
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
        # datumbox: We shouldn't have if/elses here. We could just set:
        # `_transformed_types = (features.Label,)`
        # and eliminate them.
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

    # Updated transformed types for ToImageTensor
    _transformed_types = (torch.Tensor, features._Feature, PIL.Image.Image, np.ndarray)
    # datumbox: I don't think Tensor and features._Feature should be here. This will lead to bboxes being converted to Tensors

    def __init__(self, *, copy: bool = False) -> None:
        super().__init__()
        self.copy = copy

    def _transform(self, inpt: Any, params: Dict[str, Any]) -> Any:
        # datumbox: Similarly we should rely on `_transformed_types` and avoid if/else here
        if isinstance(inpt, (features.Image, PIL.Image.Image, np.ndarray)) or is_simple_tensor(inpt):
            output = F.to_image_tensor(inpt, copy=self.copy)
            return features.Image(output)
        else:
            return inpt


class ToImagePIL(Transform):

    # Updated transformed types for ToImagePIL
    _transformed_types = (torch.Tensor, features._Feature, PIL.Image.Image, np.ndarray)
    # datumbox: same as above

    def __init__(self, *, mode: Optional[str] = None) -> None:
        super().__init__()
        self.mode = mode

    def _transform(self, inpt: Any, params: Dict[str, Any]) -> Any:
        # datumbox: same as above
        if isinstance(inpt, (features.Image, PIL.Image.Image, np.ndarray)) or is_simple_tensor(inpt):
            return F.to_image_pil(inpt, mode=self.mode)
        else:
            return inpt
