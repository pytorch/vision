from typing import Any, Dict, Optional, Union

import PIL.Image

import torch
from torchvision.prototype import features
from torchvision.prototype.transforms import functional as F, Transform


class ConvertBoundingBoxFormat(Transform):
    _transformed_types = (features.BoundingBox,)

    def __init__(self, format: Union[str, features.BoundingBoxFormat]) -> None:
        super().__init__()
        if isinstance(format, str):
            format = features.BoundingBoxFormat[format]
        self.format = format

    def _transform(self, inpt: Any, params: Dict[str, Any]) -> Any:
        output = F.convert_bounding_box_format(inpt, old_format=inpt.format, new_format=params["format"])
        return features.BoundingBox.new_like(inpt, output, format=params["format"])


class ConvertImageDtype(Transform):
    _transformed_types = (features.is_simple_tensor, features.Image)

    def __init__(self, dtype: torch.dtype = torch.float32) -> None:
        super().__init__()
        self.dtype = dtype

    def _transform(self, inpt: Any, params: Dict[str, Any]) -> Any:
        output = F.convert_image_dtype(inpt, dtype=self.dtype)
        return output if features.is_simple_tensor(inpt) else features.Image.new_like(inpt, output, dtype=self.dtype)


class ConvertColorSpace(Transform):
    _transformed_types = (features.is_simple_tensor, features.Image, PIL.Image.Image)

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


class ClampBoundingBoxes(Transform):
    _transformed_types = (features.BoundingBox,)

    def _transform(self, inpt: Any, params: Dict[str, Any]) -> Any:
        output = F.clamp_bounding_box(inpt, format=inpt.format, image_size=inpt.image_size)
        return features.BoundingBox.new_like(inpt, output)
