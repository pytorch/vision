from typing import Any, Tuple, Union, Optional

import torch
from torchvision.prototype.utils._internal import StrEnum

from ._feature import Feature


class BoundingBoxFormat(StrEnum):
    # this is just for test purposes
    _SENTINEL = -1
    XYXY = StrEnum.auto()
    XYWH = StrEnum.auto()
    CXCYWH = StrEnum.auto()


class BoundingBox(Feature):
    formats = BoundingBoxFormat  # Couldn't find a use of this in code. Is there a reason why we don't just let people access the enums directly?
    format: BoundingBoxFormat
    image_size: Tuple[int, int]

    def __new__(
        cls,
        data: Any,
        *,
        dtype: Optional[torch.dtype] = None,
        device: Optional[torch.device] = None,
        format: Union[BoundingBoxFormat, str],
        image_size: Tuple[int, int],
    ):
        bounding_box = super().__new__(cls, data, dtype=dtype, device=device)

        if isinstance(format, str):
            format = BoundingBoxFormat[format]

        bounding_box._metadata.update(dict(format=format, image_size=image_size))

        return bounding_box

    def to_format(self, format: Union[str, BoundingBoxFormat]) -> "BoundingBox":
        # import at runtime to avoid cyclic imports
        from torchvision.prototype.transforms.functional import convert_bounding_box_format
        # I think we can avoid this by not having a `to_format` method but instead require users to explicitly call the
        # convert method. As far as I see, the specific method is used only once on the code, so it is something we
        # could avoid all together.

        if isinstance(format, str):
            format = BoundingBoxFormat[format]

        return BoundingBox.new_like(
            self, convert_bounding_box_format(self, old_format=self.format, new_format=format), format=format
        )
