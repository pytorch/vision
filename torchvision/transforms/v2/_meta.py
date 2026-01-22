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

    Args:
        clamping_mode: Default is "auto" which relies on the input box'
            ``clamping_mode`` attribute. Read more in :ref:`clamping_mode_tuto`
            for more details on how to use this transform.
    """

    def __init__(self, clamping_mode: Union[CLAMPING_MODE_TYPE, str] = "auto") -> None:
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


class SetClampingMode(Transform):
    """Sets the ``clamping_mode`` attribute of the bounding boxes for future transforms.



    Args:
        clamping_mode: The clamping mode to set. Possible values are: "soft",
            "hard", or ``None``. Read more in :ref:`clamping_mode_tuto` for more
            details on how to use this transform.
    """

    def __init__(self, clamping_mode: CLAMPING_MODE_TYPE) -> None:
        super().__init__()
        self.clamping_mode = clamping_mode

        if self.clamping_mode not in (None, "soft", "hard"):
            raise ValueError(f"clamping_mode must be soft, hard or None, got {clamping_mode}")

    _transformed_types = (tv_tensors.BoundingBoxes,)

    def transform(self, inpt: tv_tensors.BoundingBoxes, params: dict[str, Any]) -> tv_tensors.BoundingBoxes:
        out: tv_tensors.BoundingBoxes = inpt.clone()  # type: ignore[assignment]
        out.clamping_mode = self.clamping_mode
        return out
