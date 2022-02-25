import math
import warnings
from typing import Any, Dict, List, Union, Sequence, Tuple, cast

import PIL.Image
import torch
from torchvision.prototype import features
from torchvision.prototype.transforms import Transform, InterpolationMode, functional as F
from torchvision.transforms.transforms import _setup_size, _interpolation_modes_from_int

from ._utils import query_image


class HorizontalFlip(Transform):
    def _transform(self, input: Any, params: Dict[str, Any]) -> Any:
        if isinstance(input, features.Image):
            output = F.horizontal_flip_image_tensor(input)
            return features.Image.new_like(input, output)
        elif isinstance(input, features.BoundingBox):
            output = F.horizontal_flip_bounding_box(input, format=input.format, image_size=input.image_size)
            return features.BoundingBox.new_like(input, output)
        elif isinstance(input, PIL.Image.Image):
            return F.horizontal_flip_image_pil(input)
        elif isinstance(input, torch.Tensor):
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
            return features.BoundingBox.new_like(input, output, image_size=self.size)
        elif isinstance(input, PIL.Image.Image):
            return F.resize_image_pil(input, self.size, interpolation=self.interpolation)
        elif isinstance(input, torch.Tensor):
            return F.resize_image_tensor(input, self.size, interpolation=self.interpolation)
        else:
            return input


class CenterCrop(Transform):
    def __init__(self, output_size: List[int]):
        super().__init__()
        self.output_size = output_size

    def _transform(self, input: Any, params: Dict[str, Any]) -> Any:
        if isinstance(input, (features.BoundingBox, features.SegmentationMask)):
            raise TypeError(f"{type(input).__name__}'s are not supported by {type(self).__name__}()")
        elif isinstance(input, features.Image):
            output = F.center_crop_image_tensor(input, self.output_size)
            return features.Image.new_like(input, output)
        elif isinstance(input, torch.Tensor):
            return F.center_crop_image_tensor(input, self.output_size)
        elif isinstance(input, PIL.Image.Image):
            return F.center_crop_image_pil(input, self.output_size)
        else:
            return input


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
        width, height = F.get_image_size(image)
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
        if isinstance(input, (features.BoundingBox, features.SegmentationMask)):
            raise TypeError(f"{type(input).__name__}'s are not supported by {type(self).__name__}()")
        elif isinstance(input, features.Image):
            output = F.resized_crop_image_tensor(
                input, **params, size=list(self.size), interpolation=self.interpolation
            )
            return features.Image.new_like(input, output)
        elif isinstance(input, torch.Tensor):
            return F.resized_crop_image_tensor(input, **params, size=list(self.size), interpolation=self.interpolation)
        elif isinstance(input, PIL.Image.Image):
            return F.resized_crop_image_pil(input, **params, size=list(self.size), interpolation=self.interpolation)
        else:
            return input
