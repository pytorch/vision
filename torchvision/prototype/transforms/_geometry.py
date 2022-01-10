from typing import Any, Dict, Tuple, TypeVar

from torchvision.prototype import features
from torchvision.prototype.transforms import FeatureSpecificArguments, Transform

from . import functional as F

T = TypeVar("T", bound=features.Feature)


class HorizontalFlip(Transform):
    def _dispatch(self, feature: T, **params: Any) -> T:
        return F.horizontal_flip(feature, **params)


class Resize(Transform):
    def __init__(
        self,
        size: Tuple[int, int],
        interpolation_mode=FeatureSpecificArguments(
            {features.SegmentationMask: "nearest"},
            default="bilinear",
        ),
    ) -> None:
        super().__init__()
        self.size = size
        self.interpolation_mode = interpolation_mode

    def get_params(self, sample: Any) -> Dict[str, Any]:
        return dict(size=self.size, interpolation_mode=self.interpolation_mode)

    def _dispatch(self, feature: T, **params: Any) -> T:
        return F.resize(feature, **params)
