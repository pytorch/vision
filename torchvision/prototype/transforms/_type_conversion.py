from typing import Any, Dict

from torchvision.prototype import features
from torchvision.prototype.transforms import Transform

from . import kernels as K


class DecodeImage(Transform):
    def _supports(self, obj: Any) -> bool:
        return isinstance(obj, features.EncodedImage)

    def _dispatch(self, input: features.EncodedImage, params: Dict[str, Any]) -> features.Image:
        return features.Image(K.decode_image_with_pil(input))


class LabelToOneHot(Transform):
    def __init__(self, num_categories: int = -1):
        super().__init__()
        self.num_categories = num_categories

    def get_params(self, sample: Any) -> Dict[str, Any]:
        return dict(num_categories=self.num_categories)

    def _supports(self, obj: Any) -> bool:
        return isinstance(obj, features.Label)

    def _dispatch(self, input: features.Label, params: Dict[str, Any]) -> features.OneHotLabel:
        num_categories = params["num_categories"]
        if num_categories == -1 and input.categories is not None:
            num_categories = len(input.categories)
        return features.OneHotLabel(
            K.label_to_one_hot(input, num_categories=num_categories), categories=input.categories
        )

    def extra_repr(self) -> str:
        if self.num_categories == -1:
            return ""

        return f"num_categories={self.num_categories}"
