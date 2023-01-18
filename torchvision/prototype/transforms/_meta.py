from typing import Any, Dict, Optional, Union

import PIL.Image

import torch

from torchvision.prototype import datapoints
from torchvision.prototype.transforms import functional as F, Transform

from .utils import is_simple_tensor


class ConvertBoundingBoxFormat(Transform):
    _transformed_types = (datapoints.BoundingBox,)

    def __init__(self, format: Union[str, datapoints.BoundingBoxFormat]) -> None:
        super().__init__()
        if isinstance(format, str):
            format = datapoints.BoundingBoxFormat[format]
        self.format = format

    def _transform(self, inpt: datapoints.BoundingBox, params: Dict[str, Any]) -> datapoints.BoundingBox:
        # We need to unwrap here to avoid unnecessary `__torch_function__` calls,
        # since `convert_format_bounding_box` does not have a dispatcher function that would do that for us
        output = F.convert_format_bounding_box(
            inpt.as_subclass(torch.Tensor), old_format=inpt.format, new_format=params["format"]
        )
        return datapoints.BoundingBox.wrap_like(inpt, output, format=params["format"])


class ConvertDtype(Transform):
    _transformed_types = (is_simple_tensor, datapoints.Image, datapoints.Video)

    def __init__(self, dtype: torch.dtype = torch.float32) -> None:
        super().__init__()
        self.dtype = dtype

    def _transform(
        self, inpt: Union[datapoints.TensorImageType, datapoints.TensorVideoType], params: Dict[str, Any]
    ) -> Union[datapoints.TensorImageType, datapoints.TensorVideoType]:
        return F.convert_dtype(inpt, self.dtype)


# We changed the name to align it with the new naming scheme. Still, `ConvertImageDtype` is
# prevalent and well understood. Thus, we just alias it without deprecating the old name.
ConvertImageDtype = ConvertDtype


class ConvertColorSpace(Transform):
    _transformed_types = (
        is_simple_tensor,
        datapoints.Image,
        PIL.Image.Image,
        datapoints.Video,
    )

    def __init__(
        self,
        color_space: Union[str, datapoints.ColorSpace],
        old_color_space: Optional[Union[str, datapoints.ColorSpace]] = None,
    ) -> None:
        super().__init__()

        if isinstance(color_space, str):
            color_space = datapoints.ColorSpace.from_str(color_space)
        self.color_space = color_space

        if isinstance(old_color_space, str):
            old_color_space = datapoints.ColorSpace.from_str(old_color_space)
        self.old_color_space = old_color_space

    def _transform(
        self, inpt: Union[datapoints.ImageType, datapoints.VideoType], params: Dict[str, Any]
    ) -> Union[datapoints.ImageType, datapoints.VideoType]:
        return F.convert_color_space(inpt, color_space=self.color_space, old_color_space=self.old_color_space)


class ClampBoundingBoxes(Transform):
    _transformed_types = (datapoints.BoundingBox,)

    def _transform(self, inpt: datapoints.BoundingBox, params: Dict[str, Any]) -> datapoints.BoundingBox:
        # We need to unwrap here to avoid unnecessary `__torch_function__` calls,
        # since `clamp_bounding_box` does not have a dispatcher function that would do that for us
        output = F.clamp_bounding_box(
            inpt.as_subclass(torch.Tensor), format=inpt.format, spatial_size=inpt.spatial_size
        )
        return datapoints.BoundingBox.wrap_like(inpt, output)
