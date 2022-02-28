from typing import Union, Any, Dict, Optional

import PIL.Image
import torch
from torchvision.prototype import features
from torchvision.prototype.transforms import Transform, functional as F
from torchvision.transforms.functional import convert_image_dtype  # We should have our an alias for this on the new API


class ConvertBoundingBoxFormat(Transform):
    def __init__(self, format: Union[str, features.BoundingBoxFormat]) -> None:
        super().__init__()
        if isinstance(format, str):
            format = features.BoundingBoxFormat[format]
        self.format = format

    def _transform(self, input: Any, params: Dict[str, Any]) -> Any:
        if type(input) is features.BoundingBox:
            output = F.convert_bounding_box_format(input, old_format=input.format, new_format=params["format"])
            return features.BoundingBox.new_like(input, output, format=params["format"])
        else:
            return input


class ConvertImageDtype(Transform):
    # Question: Why do we have both this and a ToDtype Transform? Is this due to BC? Ideally we could move people off
    # from methods that did an implicit normalization of the values (like this one, or to_tensor). cc @vfdev-5
    # If that's the case, we should move to _legacy and add deprecation warnings from day one to push people to use
    # the new methods.
    def __init__(self, dtype: torch.dtype = torch.float32) -> None:
        super().__init__()
        self.dtype = dtype

    def _transform(self, input: Any, params: Dict[str, Any]) -> Any:
        if type(input) is features.Image:
            output = convert_image_dtype(input, dtype=self.dtype)
            return features.Image.new_like(input, output, dtype=self.dtype)
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
            color_space = features.ColorSpace[color_space]
        self.color_space = color_space

        if isinstance(old_color_space, str):
            old_color_space = features.ColorSpace[old_color_space]
        self.old_color_space = old_color_space

    def _transform(self, input: Any, params: Dict[str, Any]) -> Any:
        if isinstance(input, features.Image):
            output = F.convert_image_color_space_tensor(
                input, old_color_space=input.color_space, new_color_space=self.color_space
            )
            return features.Image.new_like(input, output, color_space=self.color_space)
        elif isinstance(input, torch.Tensor):
            if self.old_color_space is None:
                raise RuntimeError("")  # Add better exception message

            return F.convert_image_color_space_tensor(
                input, old_color_space=self.old_color_space, new_color_space=self.color_space
            )
        elif isinstance(input, PIL.Image.Image):
            old_color_space = {
                "L": features.ColorSpace.GRAYSCALE,
                "RGB": features.ColorSpace.RGB,
            }.get(input.mode, features.ColorSpace.OTHER)

            return F.convert_image_color_space_pil(
                input, old_color_space=old_color_space, new_color_space=self.color_space
            )
        else:
            return input
