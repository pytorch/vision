from typing import Any, Dict, Union

from torchvision import vision_tensors
from torchvision.transforms.v2 import functional as F, Transform


class ConvertBoundingBoxFormat(Transform):
    """[BETA] Convert bounding box coordinates to the given ``format``, eg from "CXCYWH" to "XYXY".

    .. v2betastatus:: ConvertBoundingBoxFormat transform

    Args:
        format (str or vision_tensors.BoundingBoxFormat): output bounding box format.
            Possible values are defined by :class:`~torchvision.vision_tensors.BoundingBoxFormat` and
            string values match the enums, e.g. "XYXY" or "XYWH" etc.
    """

    _transformed_types = (vision_tensors.BoundingBoxes,)

    def __init__(self, format: Union[str, vision_tensors.BoundingBoxFormat]) -> None:
        super().__init__()
        if isinstance(format, str):
            format = vision_tensors.BoundingBoxFormat[format]
        self.format = format

    def _transform(self, inpt: vision_tensors.BoundingBoxes, params: Dict[str, Any]) -> vision_tensors.BoundingBoxes:
        return F.convert_bounding_box_format(inpt, new_format=self.format)  # type: ignore[return-value]


class ClampBoundingBoxes(Transform):
    """[BETA] Clamp bounding boxes to their corresponding image dimensions.

    The clamping is done according to the bounding boxes' ``canvas_size`` meta-data.

    .. v2betastatus:: ClampBoundingBoxes transform

    """

    _transformed_types = (vision_tensors.BoundingBoxes,)

    def _transform(self, inpt: vision_tensors.BoundingBoxes, params: Dict[str, Any]) -> vision_tensors.BoundingBoxes:
        return F.clamp_bounding_boxes(inpt)  # type: ignore[return-value]
