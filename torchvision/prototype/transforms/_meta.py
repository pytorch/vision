from typing import Any, Dict, Union

import torch

from torchvision import transforms as _transforms
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
    _v1_transform_cls = _transforms.ConvertImageDtype

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


class ClampBoundingBoxes(Transform):
    _transformed_types = (datapoints.BoundingBox,)

    def _transform(self, inpt: datapoints.BoundingBox, params: Dict[str, Any]) -> datapoints.BoundingBox:
        return F.clamp_bounding_box(inpt)  # type: ignore[return-value]
