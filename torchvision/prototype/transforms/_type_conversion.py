from typing import Any, Dict

from torchvision.prototype import features
from torchvision.prototype.transforms import Transform

from . import kernels as K


class DecodeImage(Transform):
    def _transform(self, input: Any, params: Dict[str, Any]) -> Any:
        if not isinstance(input, features.EncodedImage):
            return input

        return features.Image(K.decode_image_with_pil(input))


class LabelToOneHot(Transform):
    def __init__(self, num_categories: int = -1):
        super().__init__()
        self.num_categories = num_categories

    def _transform(self, input: Any, params: Dict[str, Any]) -> Any:
        if not isinstance(input, features.Label):
            return input

        num_categories = self.num_categories
        if num_categories == -1 and input.categories is not None:
            num_categories = len(input.categories)
        return features.OneHotLabel(
            K.label_to_one_hot(input, num_categories=num_categories), categories=input.categories
        )

    def extra_repr(self) -> str:
        if self.num_categories == -1:
            return ""

        return f"num_categories={self.num_categories}"
