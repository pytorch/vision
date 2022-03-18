from typing import Any, Dict

from torchvision.prototype import features
from torchvision.prototype.transforms import Transform, functional as F


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
