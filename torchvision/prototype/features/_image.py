from __future__ import annotations

import warnings
from typing import Any, Optional, Union, Tuple, cast

import torch
from torchvision.prototype.utils._internal import StrEnum
from torchvision.transforms.functional import to_pil_image
from torchvision.utils import draw_bounding_boxes
from torchvision.utils import make_grid

from ._bounding_box import BoundingBox
from ._feature import Feature


class ColorSpace(StrEnum):
    # this is just for test purposes
    _SENTINEL = -1
    OTHER = 0
    GRAYSCALE = 1
    RGB = 3


class Image(Feature):
    color_space: ColorSpace

    def __new__(
        cls,
        data: Any,
        *,
        dtype: Optional[torch.dtype] = None,
        device: Optional[torch.device] = None,
        color_space: Optional[Union[ColorSpace, str]] = None,
    ) -> Image:
        image = super().__new__(cls, data, dtype=dtype, device=device)

        if color_space is None:
            color_space = cls.guess_color_space(image)
            if color_space == ColorSpace.OTHER:
                warnings.warn("Unable to guess a specific color space. Consider passing it explicitly.")
        elif isinstance(color_space, str):
            color_space = ColorSpace[color_space]

        image._metadata.update(dict(color_space=color_space))

        return image

    @classmethod
    def _to_tensor(cls, data: Any, *, dtype: Optional[torch.dtype], device: Optional[torch.device]) -> torch.Tensor:
        tensor = super()._to_tensor(data, dtype=dtype, device=device)
        if tensor.ndim < 2:
            raise ValueError
        elif tensor.ndim == 2:
            tensor = tensor.unsqueeze(0)
        return tensor

    @property
    def image_size(self) -> Tuple[int, int]:
        return cast(Tuple[int, int], self.shape[-2:])

    @property
    def num_channels(self) -> int:
        return self.shape[-3]

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

    def show(self) -> None:
        # TODO: this is useful for developing and debugging but we should remove or at least revisit this before we
        #  promote this out of the prototype state
        to_pil_image(make_grid(self.view(-1, *self.shape[-3:]))).show()

    def draw_bounding_box(self, bounding_box: BoundingBox, **kwargs: Any) -> Image:
        # TODO: this is useful for developing and debugging but we should remove or at least revisit this before we
        #  promote this out of the prototype state
        return Image.new_like(self, draw_bounding_boxes(self, bounding_box.to_format("xyxy").view(-1, 4), **kwargs))
