import math
import warnings
from typing import Any, Dict, List, Union, Sequence, Tuple, cast

import torch
from torchvision.prototype.transforms import Transform, InterpolationMode
from torchvision.prototype.utils._internal import query_recursively
from torchvision.transforms.transforms import _setup_size, _interpolation_modes_from_int

from . import functional as F


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

    def _get_params(self, sample: Any) -> Dict[str, Any]:
        return dict(size=self.size, interpolation=self.interpolation)


class CenterCrop(Transform):
    _DISPATCHER = F.center_crop

    def __init__(self, output_size: List[int]):
        super().__init__()
        self.output_size = output_size

    def _get_params(self, sample: Any) -> Dict[str, Any]:
        return dict(output_size=self.output_size)


class RandomResizedCrop(Transform):
    _DISPATCHER = F.resized_crop

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
        image: Any = next(
            query_recursively(
                lambda input: input if input in self._DISPATCHER else None, sample  # type: ignore[no-any-return]
            )
        )
        height, width = F.get_image_size(image)
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

        return dict(top=i, left=j, height=h, width=w, size=self.size)
