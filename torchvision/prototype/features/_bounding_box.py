from typing import Any, Dict, Tuple, Union

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

    @classmethod
    def _prepare_meta_data(
        cls,
        data: torch.Tensor,
        meta_data: Dict[str, Any],
    ) -> Dict[str, Any]:
        if isinstance(meta_data["format"], str):
            meta_data["format"] = BoundingBoxFormat[meta_data["format"]]

        # TODO: input validation

        return meta_data

    def to_format(self, format: Union[str, BoundingBoxFormat]) -> "BoundingBox":
        # import at runtime to avoid cyclic imports
        from torchvision.transforms.functional import convert_format

        if isinstance(format, str):
            format = BoundingBoxFormat[format]

        return convert_format(self, new_format=format)
