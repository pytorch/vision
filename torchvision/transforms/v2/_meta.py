from typing import Any, Dict, Union

from torchvision import datapoints
from torchvision.transforms.v2 import functional as F, Transform


class ConvertBoundingBoxFormat(Transform):
    """[BETA] Convert bounding box coordinates to the given ``format``, eg from "CXCYWH" to "XYXY".

    .. v2betastatus:: ConvertBoundingBoxFormat transform

    Args:
        format (str or datapoints.BoundingBoxFormat): output bounding box format.
            Possible values are defined by :class:`~torchvision.datapoints.BoundingBoxFormat` and
            string values match the enums, e.g. "XYXY" or "XYWH" etc.
    """

    _transformed_types = (datapoints.BoundingBoxes,)

    def __init__(self, format: Union[str, datapoints.BoundingBoxFormat]) -> None:
        super().__init__()
        if isinstance(format, str):
            format = datapoints.BoundingBoxFormat[format]
        self.format = format

    def _transform(self, inpt: datapoints.BoundingBoxes, params: Dict[str, Any]) -> datapoints.BoundingBoxes:
        return F.convert_format_bounding_boxes(inpt, new_format=self.format)  # type: ignore[return-value]


class ClampBoundingBoxes(Transform):
    """[BETA] Clamp bounding boxes to their corresponding image dimensions.

    The clamping is done according to the bounding boxes' ``canvas_size`` meta-data.

    .. v2betastatus:: ClampBoundingBoxes transform

    """

    _transformed_types = (datapoints.BoundingBoxes,)

    def _transform(self, inpt: datapoints.BoundingBoxes, params: Dict[str, Any]) -> datapoints.BoundingBoxes:
        return F.clamp_bounding_boxes(inpt)  # type: ignore[return-value]
