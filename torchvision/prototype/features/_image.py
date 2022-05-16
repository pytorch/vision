from __future__ import annotations

import warnings
from typing import Any, cast, Optional, Tuple, Union

import torch
from torchvision._utils import StrEnum
from torchvision.transforms.functional import to_pil_image
from torchvision.utils import draw_bounding_boxes, make_grid

from ._bounding_box import BoundingBox
from ._feature import _Feature


class ColorSpace(StrEnum):
    OTHER = StrEnum.auto()
    GRAY = StrEnum.auto()
    GRAY_ALPHA = StrEnum.auto()
    RGB = StrEnum.auto()
    RGB_ALPHA = StrEnum.auto()

    @classmethod
    def from_pil_mode(cls, mode: str) -> ColorSpace:
        if mode == "L":
            return cls.GRAY
        elif mode == "LA":
            return cls.GRAY_ALPHA
        elif mode == "RGB":
            return cls.RGB
        elif mode == "RGBA":
            return cls.RGB_ALPHA
        else:
            return cls.OTHER


class Image(_Feature):
    color_space: ColorSpace

    def __new__(
        cls,
        data: Any,
        *,
        color_space: Optional[Union[ColorSpace, str]] = None,
        dtype: Optional[torch.dtype] = None,
        device: Optional[Union[torch.device, str, int]] = None,
        requires_grad: bool = False,
    ) -> Image:
        data = torch.as_tensor(data, dtype=dtype, device=device)  # type: ignore[arg-type]
        if data.ndim < 2:
            raise ValueError
        elif data.ndim == 2:
            data = data.unsqueeze(0)
        image = super().__new__(cls, data, requires_grad=requires_grad)

        if color_space is None:
            color_space = cls.guess_color_space(image)
            if color_space == ColorSpace.OTHER:
                warnings.warn("Unable to guess a specific color space. Consider passing it explicitly.")
        elif isinstance(color_space, str):
            color_space = ColorSpace.from_str(color_space.upper())
        elif not isinstance(color_space, ColorSpace):
            raise ValueError
        image.color_space = color_space

        return image

    @classmethod
    def new_like(
        cls, other: Image, data: Any, *, color_space: Optional[Union[ColorSpace, str]] = None, **kwargs: Any
    ) -> Image:
        return super().new_like(
            other, data, color_space=color_space if color_space is not None else other.color_space, **kwargs
        )

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
            return ColorSpace.GRAY

        num_channels = data.shape[-3]
        if num_channels == 1:
            return ColorSpace.GRAY
        elif num_channels == 2:
            return ColorSpace.GRAY_ALPHA
        elif num_channels == 3:
            return ColorSpace.RGB
        elif num_channels == 4:
            return ColorSpace.RGB_ALPHA
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
