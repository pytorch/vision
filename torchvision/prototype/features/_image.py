from typing import Dict, Any, Union, Tuple

import torch
from torchvision.prototype.utils._internal import StrEnum

from ._feature import Feature, DEFAULT


class ColorSpace(StrEnum):
    # this is just for test purposes
    _SENTINEL = -1
    OTHER = 0
    GRAYSCALE = 1
    RGB = 3


class Image(Feature):
    color_spaces = ColorSpace
    color_space: ColorSpace

    @classmethod
    def _to_tensor(cls, data, *, dtype, device):
        tensor = torch.as_tensor(data, dtype=dtype, device=device)
        if tensor.ndim == 2:
            tensor = tensor.unsqueeze(0)
        elif tensor.ndim != 3:
            raise ValueError("Only single images with 2 or 3 dimensions are allowed.")
        return tensor

    @classmethod
    def _parse_meta_data(
        cls,
        color_space: Union[str, ColorSpace] = DEFAULT,  # type: ignore[assignment]
    ) -> Dict[str, Tuple[Any, Any]]:
        if isinstance(color_space, str):
            color_space = ColorSpace[color_space]
        return dict(color_space=(color_space, cls.guess_color_space))

    @staticmethod
    def guess_color_space(data: torch.Tensor) -> ColorSpace:
        if data.ndim < 2:
            return ColorSpace.OTHER
        elif data.ndim == 2:
            return ColorSpace.GRAYSCALE

        num_channels = data.shape[-3]
        if num_channels == 1:
            return ColorSpace.GRAYSCALE
        elif num_channels == 3:
            return ColorSpace.RGB
        else:
            return ColorSpace.OTHER
