from typing import Union, Any, Dict

import torch
import torchvision.prototype.transforms.kernels as K
from torchvision.prototype import features
from torchvision.prototype.transforms import ConstantParamTransform
from torchvision.transforms.functional import convert_image_dtype


class ConvertBoundingBoxFormat(ConstantParamTransform):
    def __init__(self, format: Union[str, features.BoundingBoxFormat]) -> None:
        if isinstance(format, str):
            format = features.BoundingBoxFormat[format]
        super().__init__(format=format)

    def _supports(self, obj: Any) -> bool:
        return (obj if isinstance(obj, type) else type(obj)) is features.BoundingBox

    def _dispatch(self, feature: features.BoundingBox, params: Dict[str, Any]) -> features.BoundingBox:
        output = K.convert_bounding_box_format(feature, old_format=feature.format, new_format=params["format"])
        return features.BoundingBox.new_like(feature, output, format=params["format"])


class ConvertImageDtype(ConstantParamTransform):
    def __init__(self, dtype: torch.dtype = torch.float32) -> None:
        super().__init__(dtype=dtype)

    def _supports(self, obj: Any) -> bool:
        return (obj if isinstance(obj, type) else type(obj)) is features.Image

    def _dispatch(self, feature: features.Image, params: Dict[str, Any]) -> features.Image:
        output = convert_image_dtype(feature, dtype=params["dtype"])
        return features.Image.new_like(feature, output, dtype=params["dtype"])


class ConvertColorSpace(ConstantParamTransform):
    def __init__(self, color_space: Union[str, features.ColorSpace]) -> None:
        if isinstance(color_space, str):
            color_space = features.ColorSpace[color_space]
        super().__init__(color_space=color_space)

    def _supports(self, obj: Any) -> bool:
        return (obj if isinstance(obj, type) else type(obj)) is features.Image

    def _dispatch(self, feature: features.Image, params: Dict[str, Any]) -> features.Image:
        output = K.convert_color_space(
            feature, old_color_space=feature.color_space, new_color_space=params["color_space"]
        )
        return features.Image.new_like(feature, output, color_space=params["color_space"])
