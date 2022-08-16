from typing import Any, Dict, Optional, Union

import PIL.Image

import torch
from torchvision.prototype import features
from torchvision.prototype.transforms import functional as F, Transform
from torchvision.transforms.functional import convert_image_dtype

from ._utils import is_simple_tensor


class ConvertBoundingBoxFormat(Transform):
    def __init__(self, format: Union[str, features.BoundingBoxFormat]) -> None:
        super().__init__()
        if isinstance(format, str):
            format = features.BoundingBoxFormat[format]
        self.format = format

    def _transform(self, inpt: Any, params: Dict[str, Any]) -> Any:
        if isinstance(inpt, features.BoundingBox):
            output = F.convert_bounding_box_format(inpt, old_format=inpt.format, new_format=params["format"])
            return features.BoundingBox.new_like(inpt, output, format=params["format"])
        else:
            return inpt


class ConvertImageDtype(Transform):
    def __init__(self, dtype: torch.dtype = torch.float32) -> None:
        super().__init__()
        self.dtype = dtype

    def _transform(self, inpt: Any, params: Dict[str, Any]) -> Any:
        if isinstance(inpt, features.Image):
            output = convert_image_dtype(inpt, dtype=self.dtype)
            return features.Image.new_like(inpt, output, dtype=self.dtype)
        elif is_simple_tensor(inpt):
            return convert_image_dtype(inpt, dtype=self.dtype)
        else:
            return inpt


class ConvertColorSpace(Transform):
    # F.convert_color_space does NOT handle `_Feature`'s in general
    _transformed_types = (torch.Tensor, features.Image, PIL.Image.Image)

    def __init__(
        self,
        color_space: Union[str, features.ColorSpace],
        old_color_space: Optional[Union[str, features.ColorSpace]] = None,
        copy: bool = True,
    ) -> None:
        super().__init__()

        if isinstance(color_space, str):
            color_space = features.ColorSpace.from_str(color_space)
        self.color_space = color_space

        if isinstance(old_color_space, str):
            old_color_space = features.ColorSpace.from_str(old_color_space)
        self.old_color_space = old_color_space

        self.copy = copy

    def _transform(self, inpt: Any, params: Dict[str, Any]) -> Any:
        return F.convert_color_space(
            inpt, color_space=self.color_space, old_color_space=self.old_color_space, copy=self.copy
        )
