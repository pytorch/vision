from typing import Any, Dict, Optional, Union

import numpy as np
import PIL.Image
import torch

from torch.nn.functional import one_hot

from torchvision.prototype import datapoints
from torchvision.prototype.transforms import functional as F, Transform

from torchvision.prototype.transforms.utils import is_simple_tensor


class LabelToOneHot(Transform):
    _transformed_types = (datapoints.Label,)

    def __init__(self, num_categories: int = -1):
        super().__init__()
        self.num_categories = num_categories

    def _transform(self, inpt: datapoints.Label, params: Dict[str, Any]) -> datapoints.OneHotLabel:
        num_categories = self.num_categories
        if num_categories == -1 and inpt.categories is not None:
            num_categories = len(inpt.categories)
        output = one_hot(inpt.as_subclass(torch.Tensor), num_classes=num_categories)
        return datapoints.OneHotLabel(output, categories=inpt.categories)

    def extra_repr(self) -> str:
        if self.num_categories == -1:
            return ""

        return f"num_categories={self.num_categories}"


class PILToTensor(Transform):
    _transformed_types = (PIL.Image.Image,)

    def _transform(self, inpt: Union[PIL.Image.Image], params: Dict[str, Any]) -> torch.Tensor:
        return F.pil_to_tensor(inpt)


class ToImageTensor(Transform):
    _transformed_types = (is_simple_tensor, PIL.Image.Image, np.ndarray)

    def _transform(
        self, inpt: Union[torch.Tensor, PIL.Image.Image, np.ndarray], params: Dict[str, Any]
    ) -> datapoints.Image:
        return F.to_image_tensor(inpt)  # type: ignore[no-any-return]


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
