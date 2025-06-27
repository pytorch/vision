from typing import Any, Union

from torchvision import tv_tensors
from torchvision.transforms.v2 import functional as F, Transform
from torchvision.tv_tensors._bounding_boxes import CLAMPING_MODE_TYPE 


class ConvertBoundingBoxFormat(Transform):
    """Convert bounding box coordinates to the given ``format``, eg from "CXCYWH" to "XYXY".

    Args:
        format (str or tv_tensors.BoundingBoxFormat): output bounding box format.
            Possible values are defined by :class:`~torchvision.tv_tensors.BoundingBoxFormat` and
            string values match the enums, e.g. "XYXY" or "XYWH" etc.
    """

    _transformed_types = (tv_tensors.BoundingBoxes,)

    def __init__(self, format: Union[str, tv_tensors.BoundingBoxFormat]) -> None:
        super().__init__()
        self.format = format

    def transform(self, inpt: tv_tensors.BoundingBoxes, params: dict[str, Any]) -> tv_tensors.BoundingBoxes:
        return F.convert_bounding_box_format(inpt, new_format=self.format)  # type: ignore[return-value, arg-type]


class ClampBoundingBoxes(Transform):
    """Clamp bounding boxes to their corresponding image dimensions.

    The clamping is done according to the bounding boxes' ``canvas_size`` meta-data.

    Args:
        clamping_mode: TODOBB more docs. Default is None which relies on the input box' .clamping_mode attribute.

    """
    def __init__(self, clamping_mode: CLAMPING_MODE_TYPE = None) -> None:
        super().__init__()
        self.clamping_mode = clamping_mode

    _transformed_types = (tv_tensors.BoundingBoxes,)

    def transform(self, inpt: tv_tensors.BoundingBoxes, params: dict[str, Any]) -> tv_tensors.BoundingBoxes:
        return F.clamp_bounding_boxes(inpt, clamping_mode=self.clamping_mode)  # type: ignore[return-value]


class ClampKeyPoints(Transform):
    """Clamp keypoints to their corresponding image dimensions.

    The clamping is done according to the keypoints' ``canvas_size`` meta-data.
    """

    _transformed_types = (tv_tensors.KeyPoints,)

    def transform(self, inpt: tv_tensors.KeyPoints, params: dict[str, Any]) -> tv_tensors.KeyPoints:
        return F.clamp_keypoints(inpt)  # type: ignore[return-value]
