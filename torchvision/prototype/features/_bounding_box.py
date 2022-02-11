from __future__ import annotations

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
    ) -> BoundingBox:
        bounding_box = super().__new__(cls, data, dtype=dtype, device=device)

        if isinstance(format, str):
            format = BoundingBoxFormat[format]

        bounding_box._metadata.update(dict(format=format, image_size=image_size))

        return bounding_box

    def to_format(self, format: Union[str, BoundingBoxFormat]) -> BoundingBox:
        # TODO: this is useful for developing and debugging but we should remove or at least revisit this before we
        #  promote this out of the prototype state

        # import at runtime to avoid cyclic imports
        from torchvision.prototype.transforms.kernels import convert_bounding_box_format

        if isinstance(format, str):
            format = BoundingBoxFormat[format]

        return BoundingBox.new_like(
            self, convert_bounding_box_format(self, old_format=self.format, new_format=format), format=format
        )
