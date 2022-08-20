import warnings
from typing import Any, Dict, Optional

import numpy as np
import PIL.Image
import torch
import torchvision.prototype.transforms.functional as F
from torchvision.prototype import features
from torchvision.prototype.features import ColorSpace
from torchvision.prototype.transforms import Transform
from torchvision.transforms import functional as _F
from typing_extensions import Literal

from ._transform import _RandomApplyTransform
from ._utils import is_simple_tensor


class ToTensor(Transform):

    # Updated transformed types for ToTensor
    _transformed_types = (torch.Tensor, features._Feature, PIL.Image.Image, np.ndarray)

    def __init__(self) -> None:
        warnings.warn(
            "The transform `ToTensor()` is deprecated and will be removed in a future release. "
            "Instead, please use `transforms.ToImageTensor()`."
        )
        super().__init__()

    def _transform(self, inpt: Any, params: Dict[str, Any]) -> Any:
        if isinstance(inpt, (PIL.Image.Image, np.ndarray)):
            return _F.to_tensor(inpt)
        else:
            return inpt


class PILToTensor(Transform):
    def __init__(self) -> None:
        warnings.warn(
            "The transform `PILToTensor()` is deprecated and will be removed in a future release. "
            "Instead, please use `transforms.ToImageTensor()`."
        )
        super().__init__()

    def _transform(self, inpt: Any, params: Dict[str, Any]) -> Any:
        if isinstance(inpt, PIL.Image.Image):
            return _F.pil_to_tensor(inpt)
        else:
            return inpt


class ToPILImage(Transform):

    # Updated transformed types for ToPILImage
    _transformed_types = (torch.Tensor, features._Feature, PIL.Image.Image, np.ndarray)

    def __init__(self, mode: Optional[str] = None) -> None:
        warnings.warn(
            "The transform `ToPILImage()` is deprecated and will be removed in a future release. "
            "Instead, please use `transforms.ToImagePIL()`."
        )
        super().__init__()
        self.mode = mode

    def _transform(self, inpt: Any, params: Dict[str, Any]) -> Any:
        if is_simple_tensor(inpt) or isinstance(inpt, (features.Image, np.ndarray)):
            return _F.to_pil_image(inpt, mode=self.mode)
        else:
            return inpt


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

    def _transform(self, inpt: Any, params: Dict[str, Any]) -> Any:
        output = F.convert_color_space(inpt, color_space=ColorSpace.GRAY, old_color_space=ColorSpace.RGB)
        if self.num_output_channels == 3:
            output = F.convert_color_space(inpt, color_space=ColorSpace.RGB, old_color_space=ColorSpace.GRAY)
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

    def _transform(self, inpt: Any, params: Dict[str, Any]) -> Any:
        output = F.convert_color_space(inpt, color_space=ColorSpace.GRAY, old_color_space=ColorSpace.RGB)
        return F.convert_color_space(output, color_space=ColorSpace.RGB, old_color_space=ColorSpace.GRAY)
