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
    formats = BoundingBoxFormat
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
