import collections.abc
import math
import warnings
from typing import Any, Dict, List, Union, Sequence, Tuple, cast

import PIL.Image
import torch
from torchvision.prototype import features
from torchvision.prototype.transforms import Transform, InterpolationMode, functional as F
from torchvision.transforms.functional import pil_to_tensor
from torchvision.transforms.transforms import _setup_size, _interpolation_modes_from_int

from ._utils import query_image, get_image_dimensions, has_any, is_simple_tensor


class RandomHorizontalFlip(Transform):
    def __init__(self, p: float = 0.5) -> None:
        super().__init__()
        self.p = p

    def forward(self, *inputs: Any) -> Any:
        sample = inputs if len(inputs) > 1 else inputs[0]
        if torch.rand(1) >= self.p:
            return sample

        return super().forward(sample)

    def _transform(self, input: Any, params: Dict[str, Any]) -> Any:
        if isinstance(input, features.Image):
            output = F.horizontal_flip_image_tensor(input)
            return features.Image.new_like(input, output)
        elif isinstance(input, features.SegmentationMask):
            output = F.horizontal_flip_segmentation_mask(input)
            return features.SegmentationMask.new_like(input, output)
        elif isinstance(input, features.BoundingBox):
            output = F.horizontal_flip_bounding_box(input, format=input.format, image_size=input.image_size)
            return features.BoundingBox.new_like(input, output)
        elif isinstance(input, PIL.Image.Image):
            return F.horizontal_flip_image_pil(input)
        elif is_simple_tensor(input):
            return F.horizontal_flip_image_tensor(input)
        else:
            return input


class Resize(Transform):
    def __init__(
        self,
        size: Union[int, Sequence[int]],
        interpolation: InterpolationMode = InterpolationMode.BILINEAR,
    ) -> None:
        super().__init__()
        self.size = [size] if isinstance(size, int) else list(size)
        self.interpolation = interpolation

    def _transform(self, input: Any, params: Dict[str, Any]) -> Any:
        if isinstance(input, features.Image):
            output = F.resize_image_tensor(input, self.size, interpolation=self.interpolation)
            return features.Image.new_like(input, output)
        elif isinstance(input, features.SegmentationMask):
            output = F.resize_segmentation_mask(input, self.size)
            return features.SegmentationMask.new_like(input, output)
        elif isinstance(input, features.BoundingBox):
            output = F.resize_bounding_box(input, self.size, image_size=input.image_size)
            return features.BoundingBox.new_like(input, output, image_size=cast(Tuple[int, int], tuple(self.size)))
        elif isinstance(input, PIL.Image.Image):
            return F.resize_image_pil(input, self.size, interpolation=self.interpolation)
        elif is_simple_tensor(input):
            return F.resize_image_tensor(input, self.size, interpolation=self.interpolation)
        else:
            return input


class CenterCrop(Transform):
    def __init__(self, output_size: List[int]):
        super().__init__()
        self.output_size = output_size

    def _transform(self, input: Any, params: Dict[str, Any]) -> Any:
        if isinstance(input, features.Image):
            output = F.center_crop_image_tensor(input, self.output_size)
            return features.Image.new_like(input, output)
        elif is_simple_tensor(input):
            return F.center_crop_image_tensor(input, self.output_size)
        elif isinstance(input, PIL.Image.Image):
            return F.center_crop_image_pil(input, self.output_size)
        else:
            return input

    def forward(self, *inputs: Any) -> Any:
        sample = inputs if len(inputs) > 1 else inputs[0]
        if has_any(sample, features.BoundingBox, features.SegmentationMask):
            raise TypeError(f"BoundingBox'es and SegmentationMask's are not supported by {type(self).__name__}()")
        return super().forward(sample)


class RandomResizedCrop(Transform):
    def __init__(
        self,
        size: Union[int, Sequence[int]],
        scale: Tuple[float, float] = (0.08, 1.0),
        ratio: Tuple[float, float] = (3.0 / 4.0, 4.0 / 3.0),
        interpolation: InterpolationMode = InterpolationMode.BILINEAR,
    ) -> None:
        super().__init__()
        self.size = _setup_size(size, error_msg="Please provide only two dimensions (h, w) for size.")

        if not isinstance(scale, Sequence):
            raise TypeError("Scale should be a sequence")
        scale = cast(Tuple[float, float], scale)
        if not isinstance(ratio, Sequence):
            raise TypeError("Ratio should be a sequence")
        ratio = cast(Tuple[float, float], ratio)
        if (scale[0] > scale[1]) or (ratio[0] > ratio[1]):
            warnings.warn("Scale and ratio should be of kind (min, max)")

        # Backward compatibility with integer value
        if isinstance(interpolation, int):
            warnings.warn(
                "Argument interpolation should be of type InterpolationMode instead of int. "
                "Please, use InterpolationMode enum."
            )
            interpolation = _interpolation_modes_from_int(interpolation)

        self.size = size
        self.scale = scale
        self.ratio = ratio
        self.interpolation = interpolation

    def _get_params(self, sample: Any) -> Dict[str, Any]:
        image = query_image(sample)
        _, height, width = get_image_dimensions(image)
        area = height * width

        log_ratio = torch.log(torch.tensor(self.ratio))
        for _ in range(10):
            target_area = area * torch.empty(1).uniform_(self.scale[0], self.scale[1]).item()
            aspect_ratio = torch.exp(
                torch.empty(1).uniform_(
                    log_ratio[0],  # type: ignore[arg-type]
                    log_ratio[1],  # type: ignore[arg-type]
                )
            ).item()

            w = int(round(math.sqrt(target_area * aspect_ratio)))
            h = int(round(math.sqrt(target_area / aspect_ratio)))

            if 0 < w <= width and 0 < h <= height:
                i = torch.randint(0, height - h + 1, size=(1,)).item()
                j = torch.randint(0, width - w + 1, size=(1,)).item()
                break
        else:
            # Fallback to central crop
            in_ratio = float(width) / float(height)
            if in_ratio < min(self.ratio):
                w = width
                h = int(round(w / min(self.ratio)))
            elif in_ratio > max(self.ratio):
                h = height
                w = int(round(h * max(self.ratio)))
            else:  # whole image
                w = width
                h = height
            i = (height - h) // 2
            j = (width - w) // 2

        return dict(top=i, left=j, height=h, width=w)

    def _transform(self, input: Any, params: Dict[str, Any]) -> Any:
        if isinstance(input, features.Image):
            output = F.resized_crop_image_tensor(
                input, **params, size=list(self.size), interpolation=self.interpolation
            )
            return features.Image.new_like(input, output)
        elif is_simple_tensor(input):
            return F.resized_crop_image_tensor(input, **params, size=list(self.size), interpolation=self.interpolation)
        elif isinstance(input, PIL.Image.Image):
            return F.resized_crop_image_pil(input, **params, size=list(self.size), interpolation=self.interpolation)
        else:
            return input

    def forward(self, *inputs: Any) -> Any:
        sample = inputs if len(inputs) > 1 else inputs[0]
        if has_any(sample, features.BoundingBox, features.SegmentationMask):
            raise TypeError(f"BoundingBox'es and SegmentationMask's are not supported by {type(self).__name__}()")
        return super().forward(sample)


class MultiCropResult(list):
    """Helper class for :class:`~torchvision.prototype.transforms.BatchMultiCrop`.

    Outputs of multi crop transforms such as :class:`~torchvision.prototype.transforms.FiveCrop` and
    `:class:`~torchvision.prototype.transforms.TenCrop` should be wrapped in this in order to be batched correctly by
    :class:`~torchvision.prototype.transforms.BatchMultiCrop`.
    """

    pass


class FiveCrop(Transform):
    def __init__(self, size: Union[int, Sequence[int]]) -> None:
        super().__init__()
        self.size = _setup_size(size, error_msg="Please provide only two dimensions (h, w) for size.")

    def _transform(self, input: Any, params: Dict[str, Any]) -> Any:
        if isinstance(input, features.Image):
            output = F.five_crop_image_tensor(input, self.size)
            return MultiCropResult(features.Image.new_like(input, o) for o in output)
        elif is_simple_tensor(input):
            return MultiCropResult(F.five_crop_image_tensor(input, self.size))
        elif isinstance(input, PIL.Image.Image):
            return MultiCropResult(F.five_crop_image_pil(input, self.size))
        else:
            return input

    def forward(self, *inputs: Any) -> Any:
        sample = inputs if len(inputs) > 1 else inputs[0]
        if has_any(sample, features.BoundingBox, features.SegmentationMask):
            raise TypeError(f"BoundingBox'es and SegmentationMask's are not supported by {type(self).__name__}()")
        return super().forward(sample)


class TenCrop(Transform):
    def __init__(self, size: Union[int, Sequence[int]], vertical_flip: bool = False) -> None:
        super().__init__()
        self.size = _setup_size(size, error_msg="Please provide only two dimensions (h, w) for size.")
        self.vertical_flip = vertical_flip

    def _transform(self, input: Any, params: Dict[str, Any]) -> Any:
        if isinstance(input, features.Image):
            output = F.ten_crop_image_tensor(input, self.size, vertical_flip=self.vertical_flip)
            return MultiCropResult(features.Image.new_like(input, o) for o in output)
        elif is_simple_tensor(input):
            return MultiCropResult(F.ten_crop_image_tensor(input, self.size))
        elif isinstance(input, PIL.Image.Image):
            return MultiCropResult(F.ten_crop_image_pil(input, self.size))
        else:
            return input

    def forward(self, *inputs: Any) -> Any:
        sample = inputs if len(inputs) > 1 else inputs[0]
        if has_any(sample, features.BoundingBox, features.SegmentationMask):
            raise TypeError(f"BoundingBox'es and SegmentationMask's are not supported by {type(self).__name__}()")
        return super().forward(sample)


class BatchMultiCrop(Transform):
    def forward(self, *inputs: Any) -> Any:
        # This is basically the functionality of `torchvision.prototype.utils._internal.apply_recursively` with one
        # significant difference:
        # Since we need multiple images to batch them together, we need to explicitly exclude `MultiCropResult` from
        # the sequence case.
        def apply_recursively(obj: Any) -> Any:
            if isinstance(obj, MultiCropResult):
                crops = obj
                if isinstance(obj[0], PIL.Image.Image):
                    crops = [pil_to_tensor(crop) for crop in crops]  # type: ignore[assignment]

                batch = torch.stack(crops)

                if isinstance(obj[0], features.Image):
                    batch = features.Image.new_like(obj[0], batch)

                return batch
            elif isinstance(obj, collections.abc.Sequence) and not isinstance(obj, str):
                return [apply_recursively(item) for item in obj]
            elif isinstance(obj, collections.abc.Mapping):
                return {key: apply_recursively(item) for key, item in obj.items()}
            else:
                return obj

        return apply_recursively(inputs if len(inputs) > 1 else inputs[0])
