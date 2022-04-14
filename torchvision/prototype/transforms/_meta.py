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

    def _transform(self, input: Any, params: Dict[str, Any]) -> Any:
        if isinstance(input, features.BoundingBox):
            output = F.convert_bounding_box_format(input, old_format=input.format, new_format=params["format"])
            return features.BoundingBox.new_like(input, output, format=params["format"])
        else:
            return input


class ConvertImageDtype(Transform):
    def __init__(self, dtype: torch.dtype = torch.float32) -> None:
        super().__init__()
        self.dtype = dtype

    def _transform(self, input: Any, params: Dict[str, Any]) -> Any:
        if isinstance(input, features.Image):
            output = convert_image_dtype(input, dtype=self.dtype)
            return features.Image.new_like(input, output, dtype=self.dtype)
        elif is_simple_tensor(input):
            return convert_image_dtype(input, dtype=self.dtype)
        else:
            return input


class ConvertImageColorSpace(Transform):
    def __init__(
        self,
        color_space: Union[str, features.ColorSpace],
        old_color_space: Optional[Union[str, features.ColorSpace]] = None,
    ) -> None:
        super().__init__()

        if isinstance(color_space, str):
            color_space = features.ColorSpace.from_str(color_space)
        self.color_space = color_space

        if isinstance(old_color_space, str):
            old_color_space = features.ColorSpace.from_str(old_color_space)
        self.old_color_space = old_color_space

    def _transform(self, input: Any, params: Dict[str, Any]) -> Any:
        if isinstance(input, features.Image):
            output = F.convert_image_color_space_tensor(
                input, old_color_space=input.color_space, new_color_space=self.color_space
            )
            return features.Image.new_like(input, output, color_space=self.color_space)
        elif is_simple_tensor(input):
            if self.old_color_space is None:
                raise RuntimeError(
                    f"In order to convert simple tensor images, `{type(self).__name__}(...)` "
                    f"needs to be constructed with the `old_color_space=...` parameter."
                )

            return F.convert_image_color_space_tensor(
                input, old_color_space=self.old_color_space, new_color_space=self.color_space
            )
        elif isinstance(input, PIL.Image.Image):
            return F.convert_image_color_space_pil(input, color_space=self.color_space)
        else:
            return input
