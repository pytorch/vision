from typing import Any, TypeVar, Dict

from torchvision.prototype import features
from torchvision.prototype.transforms import Transform

from . import functional as F

T = TypeVar("T", bound=features.Feature)


class DecodeImage(Transform):
    def supports(self, obj: Any) -> bool:
        return F.utils.is_supported(obj, features.EncodedImage)

    def _dispatch(  # type: ignore[override]
        self,
        input: features.EncodedImage,
        params: Dict[str, Any],
    ) -> features.Image:
        return features.Image(F.decode_image_with_pil(input))


class LabelToOneHot(Transform):
    def __init__(self, num_categories: int = -1):
        super().__init__()
        self.num_categories = num_categories

    def get_params(self, sample: Any) -> Dict[str, Any]:
        return dict(num_categories=self.num_categories)

    def supports(self, obj: Any) -> bool:
        return F.utils.is_supported(obj, features.Label)

    def _dispatch(  # type: ignore[override]
        self,
        input: features.Label,
        params: Dict[str, Any],
    ) -> features.OneHotLabel:
        num_categories = params["num_categories"]
        if num_categories == -1 and input.categories is not None:
            num_categories = len(input.categories)
        return features.OneHotLabel(
            F.label_to_one_hot(input, num_categories=num_categories), categories=input.categories
        )

    def extra_repr(self) -> str:
        if self.num_categories == -1:
            return ""

        return f"num_categories={self.num_categories}"
