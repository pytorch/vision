from typing import Any, Dict, Union

import torch

from torchvision import datapoints, transforms as _transforms
from torchvision.transforms.v2 import functional as F, Transform

from .utils import is_simple_tensor


class ConvertBoundingBoxFormat(Transform):
    """[BETA] Convert bounding box coordinates to the given ``format``, eg from "CXCYWH" to "XYXY".

    .. v2betastatus:: ConvertBoundingBoxFormat transform

    Args:
        format (str or datapoints.BoundingBoxFormat): output bounding box format.
            Possible values are defined by :class:`~torchvision.datapoints.BoundingBoxFormat` and
            string values match the enums, e.g. "XYXY" or "XYWH" etc.
    """

    _transformed_types = (datapoints.BoundingBox,)

    def __init__(self, format: Union[str, datapoints.BoundingBoxFormat]) -> None:
        super().__init__()
        if isinstance(format, str):
            format = datapoints.BoundingBoxFormat[format]
        self.format = format

    def _transform(self, inpt: datapoints.BoundingBox, params: Dict[str, Any]) -> datapoints.BoundingBox:
        return F.convert_format_bounding_box(inpt, new_format=self.format)  # type: ignore[return-value]


class ConvertImageDtype(Transform):
    """[BETA] Convert input image to the given ``dtype`` and scale the values accordingly.

    .. v2betastatus:: ConvertImageDtype transform

    .. warning::
        Consider using ToDtype(dtype, scale=True) instead.

    This function does not support PIL Image.

    Args:
        dtype (torch.dtype): Desired data type of the output

    .. note::

        When converting from a smaller to a larger integer ``dtype`` the maximum values are **not** mapped exactly.
        If converted back and forth, this mismatch has no effect.

    Raises:
        RuntimeError: When trying to cast :class:`torch.float32` to :class:`torch.int32` or :class:`torch.int64` as
            well as for trying to cast :class:`torch.float64` to :class:`torch.int64`. These conversions might lead to
            overflow errors since the floating point ``dtype`` cannot store consecutive integers over the whole range
            of the integer ``dtype``.
    """

    _v1_transform_cls = _transforms.ConvertImageDtype

    _transformed_types = (is_simple_tensor, datapoints.Image)

    def __init__(self, dtype: torch.dtype = torch.float32) -> None:
        super().__init__()
        self.dtype = dtype

    def _transform(self, inpt: Any, params: Dict[str, Any]) -> Any:
        return F.to_dtype(inpt, dtype=self.dtype, scale=True)


class ClampBoundingBox(Transform):
    """[BETA] Clamp bounding boxes to their corresponding image dimensions.

    The clamping is done according to the bounding boxes' ``spatial_size`` meta-data.

    .. v2betastatus:: ClampBoundingBox transform

    """

    _transformed_types = (datapoints.BoundingBox,)

    def _transform(self, inpt: datapoints.BoundingBox, params: Dict[str, Any]) -> datapoints.BoundingBox:
        return F.clamp_bounding_box(inpt)  # type: ignore[return-value]
