from typing import Tuple, TypeVar, List

from torchvision.prototype import features
from torchvision.prototype.transforms import FeatureSpecificArguments, ConstantParamTransform, Transform

from . import functional as F

T = TypeVar("T", bound=features.Feature)


class HorizontalFlip(Transform):
    _DISPATCHER = F.horizontal_flip


class Resize(ConstantParamTransform):
    _DISPATCHER = F.resize

    def __init__(
        self,
        size: Tuple[int, int],
        interpolation=FeatureSpecificArguments(
            {features.SegmentationMask: F.InterpolationMode.NEAREST},
            default=F.InterpolationMode.BILINEAR,
        ),
    ) -> None:
        super().__init__(size=size, interpolation=interpolation)


class CenterCrop(ConstantParamTransform):
    _DISPATCHER = F.center_crop

    def __init__(self, output_size: List[int]):
        super().__init__(output_size=output_size)
