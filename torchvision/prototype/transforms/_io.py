from typing import Any, TypeVar

from torchvision.prototype import features
from torchvision.prototype.transforms import Transform

from . import functional as F

T = TypeVar("T", bound=features.Feature)


class DecodeImages(Transform):
    def _dispatch(self, feature: T, **params: Any) -> T:
        return F.decode_image(feature, **params)
