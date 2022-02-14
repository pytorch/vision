from typing import Any, Dict, List, Union, Sequence, Tuple

from torchvision import transforms as _transforms
from torchvision.prototype.transforms import Transform, InterpolationMode

from . import functional as F
from .utils import Query


class HorizontalFlip(Transform):
    _DISPATCHER = F.horizontal_flip


class Resize(Transform):
    _DISPATCHER = F.resize

    def __init__(
        self,
        size: Union[int, Sequence[int]],
        interpolation: InterpolationMode = InterpolationMode.BILINEAR,
    ) -> None:
        super().__init__()
        self.size = size
        self.interpolation = interpolation

    def get_params(self, sample: Any) -> Dict[str, Any]:
        return dict(size=self.size, interpolation=self.interpolation)

    def extra_repr(self) -> str:
        return self._extra_repr_from_attrs("size", "interpolation")


class CenterCrop(Transform):
    _DISPATCHER = F.center_crop

    def __init__(self, output_size: List[int]):
        super().__init__()
        self.output_size = output_size

    def get_params(self, sample: Any) -> Dict[str, Any]:
        return dict(output_size=self.output_size)

    def extra_repr(self) -> str:
        return self._extra_repr_from_attrs("output_size")


class RandomResizedCrop(Transform):
    _LEGACY_CLS = _transforms.RandomResizedCrop
    _DISPATCHER = F.resized_crop

    def __init__(
        self,
        size: Union[int, Sequence[int]],
        scale: Tuple[float, float] = (0.08, 1.0),
        ratio: Tuple[float, float] = (3.0 / 4.0, 4.0 / 3.0),
        interpolation: InterpolationMode = InterpolationMode.BILINEAR,
    ) -> None:
        super().__init__()
        legacy_transform = self._LEGACY_CLS(size=size, scale=scale, ratio=ratio, interpolation=interpolation)
        self.size = legacy_transform.size
        self.scale = legacy_transform.scale
        self.ratio = legacy_transform.ratio
        self.interpolation = legacy_transform.interpolation

    def get_params(self, sample: Any) -> Dict[str, Any]:
        image = Query(sample).image_for_size_extraction()
        top, left, height, width = self._LEGACY_CLS.get_params(image, scale=list(self.scale), ratio=list(self.ratio))
        return dict(
            top=top,
            left=left,
            height=height,
            width=width,
            size=self.size,
            interpolation=self.interpolation,
        )
