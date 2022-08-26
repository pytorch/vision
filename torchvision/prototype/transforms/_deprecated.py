import warnings
from typing import Any, Dict, Optional

import numpy as np
import PIL.Image
import torch

from torchvision.prototype import features
from torchvision.prototype.transforms import Transform
from torchvision.transforms import functional as _F
from typing_extensions import Literal

from ._transform import _RandomApplyTransform
from ._utils import is_simple_tensor, query_chw


class ToTensor(Transform):
    _transformed_types = (PIL.Image.Image, np.ndarray)

    def __init__(self) -> None:
        warnings.warn(
            "The transform `ToTensor()` is deprecated and will be removed in a future release. "
            "Instead, please use `transforms.ToImageTensor()`."
        )
        super().__init__()

    def _transform(self, inpt: Any, params: Dict[str, Any]) -> torch.Tensor:
        return _F.to_tensor(inpt)


class PILToTensor(Transform):
    _transformed_types = (PIL.Image.Image,)

    def __init__(self) -> None:
        warnings.warn(
            "The transform `PILToTensor()` is deprecated and will be removed in a future release. "
            "Instead, please use `transforms.ToImageTensor()`."
        )
        super().__init__()

    def _transform(self, inpt: Any, params: Dict[str, Any]) -> torch.Tensor:
        return _F.pil_to_tensor(inpt)


class ToPILImage(Transform):
    _transformed_types = (is_simple_tensor, features.Image, np.ndarray)

    def __init__(self, mode: Optional[str] = None) -> None:
        warnings.warn(
            "The transform `ToPILImage()` is deprecated and will be removed in a future release. "
            "Instead, please use `transforms.ToImagePIL()`."
        )
        super().__init__()
        self.mode = mode

    def _transform(self, inpt: Any, params: Dict[str, Any]) -> PIL.Image:
        return _F.to_pil_image(inpt, mode=self.mode)


class Grayscale(Transform):
    _transformed_types = (features.Image, PIL.Image.Image, is_simple_tensor)

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
        return _F.rgb_to_grayscale(inpt, num_output_channels=self.num_output_channels)


class RandomGrayscale(_RandomApplyTransform):
    _transformed_types = (features.Image, PIL.Image.Image, is_simple_tensor)

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

    def _get_params(self, sample: Any) -> Dict[str, Any]:
        num_input_channels, _, _ = query_chw(sample)
        return dict(num_input_channels=num_input_channels)

    def _transform(self, inpt: Any, params: Dict[str, Any]) -> Any:
        return _F.rgb_to_grayscale(inpt, num_output_channels=params["num_input_channels"])
