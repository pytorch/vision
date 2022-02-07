from typing import Any, Dict
from typing import TypeVar, List, Union, Sequence

from torchvision import transforms as _transforms
from torchvision.prototype import features
from torchvision.prototype.transforms import FeatureSpecificArguments, ConstantParamTransform, Transform
from torchvision.transforms.transforms import _setup_size

from . import functional as F
from .utils import Query

T = TypeVar("T", bound=features.Feature)


class HorizontalFlip(Transform):
    _DISPATCH = F.horizontal_flip


class Resize(ConstantParamTransform):
    _DISPATCH = F.resize

    def __init__(
        self,
        size: Union[int, Sequence[int]],
        interpolation=FeatureSpecificArguments(
            {features.SegmentationMask: F.InterpolationMode.NEAREST},
            default=F.InterpolationMode.BILINEAR,
        ),
    ) -> None:
        super().__init__(size=size, interpolation=interpolation)

    def get_params(self, sample: Any) -> Dict[str, Any]:
        params = super().get_params(sample)
        if not isinstance(params["interpolation"], str):
            params["interpolation"] = params["interpolation"].value
        return params


class CenterCrop(ConstantParamTransform):
    _DISPATCH = F.center_crop

    def __init__(self, output_size: List[int]):
        super().__init__(output_size=output_size)


class RandomResizedCrop(Transform):
    _DISPATCH = F.resized_crop

    def __init__(
        self, size, scale=(0.08, 1.0), ratio=(3.0 / 4.0, 4.0 / 3.0), interpolation=F.InterpolationMode.BILINEAR
    ):
        super().__init__()
        self.size = _setup_size(size, error_msg="Please provide only two dimensions (h, w) for size.")
        self.scale = scale
        self.ratio = ratio
        self.interpolation = interpolation

    def get_params(self, sample: Any) -> Dict[str, Any]:
        image = Query(sample).image_for_size_extraction()
        top, left, height, width = _transforms.RandomResizedCrop.get_params(image, scale=self.scale, ratio=self.ratio)
        return dict(top=top, left=left, height=height, width=width, size=self.size, interpolation=self.interpolation)
