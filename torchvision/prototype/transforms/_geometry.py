from typing import Any, Dict, List, Union, Sequence, Tuple

from torchvision import transforms as _transforms
from torchvision.prototype import features
from torchvision.prototype.transforms import Transform, InterpolationMode, kernels as K
from torchvision.transforms import functional as _F

from .utils import Query, legacy_transform


class HorizontalFlip(Transform):
    @legacy_transform(_F.hflip)
    def _transform(self, input: Any, params: Dict[str, Any]) -> Any:
        if type(input) is features.Image:
            output = K.horizontal_flip_image(input)
            return features.Image.new_like(input, output)
        elif type(input) is features.BoundingBox:
            output = K.horizontal_flip_bounding_box(input, format=input.format, image_size=input.image_size)
            return features.BoundingBox.new_like(input, output)
        else:
            return input


class Resize(Transform):
    def __init__(
        self,
        size: Union[int, Sequence[int]],
        interpolation: InterpolationMode = InterpolationMode.BILINEAR,
    ) -> None:
        super().__init__()
        self.size = [size, size] if isinstance(size, int) else list(size)
        self.interpolation = interpolation

    @legacy_transform(_F.resize, "size", "interpolation")
    def _dispatch(self, input: Any, params: Dict[str, Any]) -> Any:
        if type(input) is features.Image:
            output = K.resize_image(input, size=self.size, interpolation=self.interpolation)
            return features.Image.new_like(input, output)
        elif type(input) is features.SegmentationMask:
            return features.SegmentationMask.new_like(input, K.resize_segmentation_mask(input, size=self.size))
        elif type(input) is features.BoundingBox:
            output = K.resize_bounding_box(input, size=self.size, image_size=input.image_size)
            return features.BoundingBox.new_like(input, output, image_size=self.size)
        else:
            return input

    def extra_repr(self) -> str:
        return self._extra_repr_from_attrs("size", "interpolation")


class CenterCrop(Transform):
    def __init__(self, output_size: List[int]):
        super().__init__()
        self.output_size = output_size

    @legacy_transform(_F.center_crop, "output_size")
    def _dispatch(self, input: Any, params: Dict[str, Any]) -> Any:
        if type(input) is features.Image:
            output = K.center_crop_image(input, **params)
            return features.Image.new_like(input, output)
        else:
            return input

    def extra_repr(self) -> str:
        return self._extra_repr_from_attrs("output_size")


class RandomResizedCrop(Transform):
    _LEGACY_CLS = _transforms.RandomResizedCrop

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
        top, left, height, width = _transforms.RandomResizedCrop.get_params(
            image, scale=list(self.scale), ratio=list(self.ratio)
        )
        return dict(
            top=top,
            left=left,
            height=height,
            width=width,
        )

    def _transform(self, input: Any, params: Dict[str, Any]) -> Any:
        if type(input) is features.Image:
            output = K.resized_crop_image(input, size=self.size, interpolation=self.interpolation, **params)
            return features.Image.new_like(input, output)
        else:
            return input

    def extra_repr(self) -> str:
        return self._extra_repr_from_attrs("size", "scale", "ratio", "interpolation")
