from typing import Any, Dict, Union

from torchvision import datapoints
from torchvision.transforms.v2 import functional as F, Transform


class ConvertBBoxFormat(Transform):
    """[BETA] Convert bounding box coordinates to the given ``format``, eg from "CXCYWH" to "XYXY".

    .. v2betastatus:: ConvertBBoxFormat transform

    Args:
        format (str or datapoints.BBoxFormat): output bounding box format.
            Possible values are defined by :class:`~torchvision.datapoints.BBoxFormat` and
            string values match the enums, e.g. "XYXY" or "XYWH" etc.
    """

    _transformed_types = (datapoints.BBoxes,)

    def __init__(self, format: Union[str, datapoints.BBoxFormat]) -> None:
        super().__init__()
        if isinstance(format, str):
            format = datapoints.BBoxFormat[format]
        self.format = format

    def _transform(self, inpt: datapoints.BBoxes, params: Dict[str, Any]) -> datapoints.BBoxes:
        return F.convert_format_bounding_boxes(inpt, new_format=self.format)  # type: ignore[return-value]


class ClampBBoxes(Transform):
    """[BETA] Clamp bounding boxes to their corresponding image dimensions.

    The clamping is done according to the bounding boxes' ``spatial_size`` meta-data.

    .. v2betastatus:: ClampBBoxes transform

    """

    _transformed_types = (datapoints.BBoxes,)

    def _transform(self, inpt: datapoints.BBoxes, params: Dict[str, Any]) -> datapoints.BBoxes:
        return F.clamp_bounding_boxes(inpt)  # type: ignore[return-value]
