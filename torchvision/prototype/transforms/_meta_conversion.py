from typing import Union, Any, Dict

import torch
from torchvision.prototype import features
from torchvision.prototype.transforms import Transform, kernels as K
from torchvision.transforms import functional as _F


class ConvertBoundingBoxFormat(Transform):
    def __init__(self, format: Union[str, features.BoundingBoxFormat]) -> None:
        super().__init__()
        if isinstance(format, str):
            format = features.BoundingBoxFormat[format]
        self.format = format

    def _transform(self, input: Any, params: Dict[str, Any]) -> Any:
        if type(input) is features.BoundingBox:
            output = K.convert_bounding_box_format(input, old_format=input.format, new_format=params["format"])
            return features.BoundingBox.new_like(input, output, format=params["format"])
        else:
            return input

    def extra_repr(self) -> str:
        return self._extra_repr_from_attrs("format")


class ConvertImageDtype(Transform):
    def __init__(self, dtype: torch.dtype = torch.float32) -> None:
        super().__init__()
        self.dtype = dtype

    def _transform(self, input: Any, params: Dict[str, Any]) -> Any:
        if type(input) is features.Image:
            output = _F.convert_image_dtype(input, dtype=self.dtype)
            return features.Image.new_like(input, output, dtype=self.dtype)
        else:
            return input

    def extra_repr(self) -> str:
        return self._extra_repr_from_attrs("dtype")


class ConvertColorSpace(Transform):
    def __init__(self, color_space: Union[str, features.ColorSpace]) -> None:
        super().__init__()
        if isinstance(color_space, str):
            color_space = features.ColorSpace[color_space]
        self.color_space = color_space

    def _transform(self, input: Any, params: Dict[str, Any]) -> Any:
        if type(input) is features.Image:
            output = K.convert_color_space(input, old_color_space=input.color_space, new_color_space=self.color_space)
            return features.Image.new_like(input, output, color_space=self.color_space)
        else:
            return input

    def extra_repr(self) -> str:
        return self._extra_repr_from_attrs("color_space")
