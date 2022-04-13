import warnings
from typing import Any, Dict, Optional

import numpy as np
import PIL.Image
from torchvision.prototype import features
from torchvision.prototype.features import ColorSpace
from torchvision.prototype.transforms import Transform
from torchvision.transforms import functional as _F
from typing_extensions import Literal

from ._meta import ConvertImageColorSpace
from ._transform import _RandomApplyTransform
from ._utils import is_simple_tensor


class ToTensor(Transform):
    def __init__(self) -> None:
        warnings.warn(
            "The transform `ToTensor()` is deprecated and will be removed in a future release. "
            "Instead, please use `transforms.ToImageTensor()`."
        )
        super().__init__()

    def _transform(self, input: Any, params: Dict[str, Any]) -> Any:
        if isinstance(input, (PIL.Image.Image, np.ndarray)):
            return _F.to_tensor(input)
        else:
            return input


class PILToTensor(Transform):
    def __init__(self) -> None:
        warnings.warn(
            "The transform `PILToTensor()` is deprecated and will be removed in a future release. "
            "Instead, please use `transforms.ToImageTensor()`."
        )
        super().__init__()

    def _transform(self, input: Any, params: Dict[str, Any]) -> Any:
        if isinstance(input, PIL.Image.Image):
            return _F.pil_to_tensor(input)
        else:
            return input


class ToPILImage(Transform):
    def __init__(self, mode: Optional[str] = None) -> None:
        warnings.warn(
            "The transform `ToPILImage()` is deprecated and will be removed in a future release. "
            "Instead, please use `transforms.ToImagePIL()`."
        )
        super().__init__()
        self.mode = mode

    def _transform(self, input: Any, params: Dict[str, Any]) -> Any:
        if is_simple_tensor(input) or isinstance(input, (features.Image, np.ndarray)):
            return _F.to_pil_image(input, mode=self.mode)
        else:
            return input


class Grayscale(Transform):
    def __init__(self, num_output_channels: Literal[1, 3] = 1) -> None:
        deprecation_msg = (
            f"The transform `Grayscale(num_output_channels={num_output_channels})` "
            f"is deprecated and will be removed in a future release."
        )
        if num_output_channels == 1:
            replacement_msg = (
                "transforms.ConvertImageColorSpace(old_color_space=ColorSpace.RGB, color_space=ColorSpace.GRAY)"
            )
        else:
            replacement_msg = (
                "transforms.Compose(\n"
                "    transforms.ConvertImageColorSpace(old_color_space=ColorSpace.RGB, color_space=ColorSpace.GRAY),\n"
                "    transforms.ConvertImageColorSpace(old_color_space=ColorSpace.GRAY, color_space=ColorSpace.RGB),\n"
                ")"
            )
        warnings.warn(f"{deprecation_msg} Instead, please use\n\n{replacement_msg}")

        super().__init__()
        self.num_output_channels = num_output_channels
        self._rgb_to_gray = ConvertImageColorSpace(old_color_space=ColorSpace.RGB, color_space=ColorSpace.GRAY)
        self._gray_to_rgb = ConvertImageColorSpace(old_color_space=ColorSpace.GRAY, color_space=ColorSpace.RGB)

    def _transform(self, input: Any, params: Dict[str, Any]) -> Any:
        output = self._rgb_to_gray(input)
        if self.num_output_channels == 3:
            output = self._gray_to_rgb(output)
        return output


class RandomGrayscale(_RandomApplyTransform):
    def __init__(self, p: float = 0.1) -> None:
        warnings.warn(
            "The transform `RandomGrayscale(p=...)` is deprecated and will be removed in a future release. "
            "Instead, please use\n\n"
            "transforms.RandomApply(\n"
            "    transforms.Compose(\n"
            "        transforms.ConvertImageColorSpace(old_color_space=ColorSpace.RGB, color_space=ColorSpace.GRAY),\n"
            "        transforms.ConvertImageColorSpace(old_color_space=ColorSpace.GRAY, color_space=ColorSpace.RGB),\n"
            "    )\n"
            "    p=...,\n"
            ")"
        )

        super().__init__(p=p)
        self._rgb_to_gray = ConvertImageColorSpace(old_color_space=ColorSpace.RGB, color_space=ColorSpace.GRAY)
        self._gray_to_rgb = ConvertImageColorSpace(old_color_space=ColorSpace.GRAY, color_space=ColorSpace.RGB)

    def _transform(self, input: Any, params: Dict[str, Any]) -> Any:
        return self._gray_to_rgb(self._rgb_to_gray(input))
