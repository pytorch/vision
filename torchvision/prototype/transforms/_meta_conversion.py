from typing import Union, Any, Dict, Optional

import torch
import torchvision.prototype.transforms.functional as F
from torchvision.prototype import features
from torchvision.prototype.transforms import Transform
from torchvision.transforms.functional import convert_image_dtype


class ConvertBoundingBoxFormat(Transform):
    _DISPATCHER = F.convert_format

    def __init__(
        self,
        format: Union[str, features.BoundingBoxFormat],
        old_format: Optional[Union[str, features.BoundingBoxFormat]] = None,
    ) -> None:
        super().__init__()
        if isinstance(format, str):
            format = features.BoundingBoxFormat[format]
        self.format = format

        if isinstance(old_format, str):
            old_format = features.BoundingBoxFormat[old_format]
        self.old_format = old_format

    def _get_params(self, sample: Any) -> Dict[str, Any]:
        return dict(format=self.format, old_format=self.old_format)


class ConvertImageDtype(Transform):
    def __init__(self, dtype: torch.dtype = torch.float32) -> None:
        super().__init__()
        self.dtype = dtype

    def _transform(self, input: Any, params: Dict[str, Any]) -> Any:
        if not isinstance(input, features.Image):
            return input

        output = convert_image_dtype(input, dtype=self.dtype)
        return features.Image.new_like(input, output, dtype=self.dtype)


class ConvertColorSpace(Transform):
    _DISPATCHER = F.convert_color_space

    def __init__(self, color_space: Union[str, features.ColorSpace]) -> None:
        super().__init__()
        if isinstance(color_space, str):
            color_space = features.ColorSpace[color_space]
        self.color_space = color_space

    def _get_params(self, sample: Any) -> Dict[str, Any]:
        return dict(color_space=self.color_space)
