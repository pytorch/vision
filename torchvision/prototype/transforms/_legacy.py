from __future__ import annotations

import warnings
from typing import Any, Dict

from torchvision.prototype.features import ColorSpace
from torchvision.prototype.transforms import Transform
from typing_extensions import Literal

from ._meta import ConvertImageColorSpace
from ._transform import _RandomApplyTransform


class Grayscale(Transform):
    def __init__(self, num_output_channels: Literal[1, 3] = 1) -> None:
        warnings.warn(
            "The transform `Grayscale(num_output_channels=...)` is deprecated and will be removed in a future release. "
            "Please use "
            "`transforms.ConvertImageColorSpace(color_space=ColorSpace.GRAY, old_color_space=ColorSpace.RGB, "
            "gray_output_channels=...)` "
            "instead."
        )
        super().__init__()
        self.num_output_channels = num_output_channels
        self._rgb_to_gray = ConvertImageColorSpace(
            color_space=ColorSpace.GRAY, old_color_space=ColorSpace.RGB, gray_output_channels=num_output_channels
        )

    def _transform(self, input: Any, params: Dict[str, Any]) -> Any:
        return self._rgb_to_gray(input)


class RandomGrayscale(_RandomApplyTransform):
    def __init__(self, p: float = 0.1) -> None:
        super().__init__(p=p)
        self._rgb_to_gray = ConvertImageColorSpace(
            color_space=ColorSpace.GRAY, old_color_space=ColorSpace.RGB, gray_output_channels=3
        )

    def _transform(self, input: Any, params: Dict[str, Any]) -> Any:
        return self._rgb_to_gray(input)
