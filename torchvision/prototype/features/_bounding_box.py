from __future__ import annotations

from typing import Any, Tuple, Union, Optional

import torch
from torchvision._utils import StrEnum

from ._feature import _Feature


class BoundingBoxFormat(StrEnum):
    XYXY = StrEnum.auto()
    XYWH = StrEnum.auto()
    CXCYWH = StrEnum.auto()


class BoundingBox(_Feature):
    format: BoundingBoxFormat
    image_size: Tuple[int, int]

    def __new__(
        cls,
        data: Any,
        *,
        format: Union[BoundingBoxFormat, str],
        image_size: Tuple[int, int],
        dtype: Optional[torch.dtype] = None,
        device: Optional[Union[torch.device, str, int]] = None,
        requires_grad: bool = False,
    ) -> BoundingBox:
        bounding_box = super().__new__(cls, data, dtype=dtype, device=device, requires_grad=requires_grad)

        if isinstance(format, str):
            format = BoundingBoxFormat.from_str(format.upper())
        bounding_box.format = format

        bounding_box.image_size = image_size

        return bounding_box

    @classmethod
    def new_like(
        cls,
        other: BoundingBox,
        data: Any,
        *,
        format: Optional[Union[BoundingBoxFormat, str]] = None,
        image_size: Optional[Tuple[int, int]] = None,
        **kwargs: Any,
    ) -> BoundingBox:
        return super().new_like(
            other,
            data,
            format=format if format is not None else other.format,
            image_size=image_size if image_size is not None else other.image_size,
            **kwargs,
        )

    def to_format(self, format: Union[str, BoundingBoxFormat]) -> BoundingBox:
        # TODO: this is useful for developing and debugging but we should remove or at least revisit this before we
        #  promote this out of the prototype state

        # import at runtime to avoid cyclic imports
        from torchvision.prototype.transforms.functional import convert_bounding_box_format

        if isinstance(format, str):
            format = BoundingBoxFormat.from_str(format.upper())

        return BoundingBox.new_like(
            self, convert_bounding_box_format(self, old_format=self.format, new_format=format), format=format
        )

    def horizontal_flip(self) -> BoundingBox:
        output = self._F.horizontal_flip_bounding_box(self, format=self.format, image_size=self.image_size)
        return BoundingBox.new_like(self, output)

    def vertical_flip(self) -> BoundingBox:
        output = self._F.vertical_flip_bounding_box(self, format=self.format, image_size=self.image_size)
        return BoundingBox.new_like(self, output)

    def resize(self, size, *, interpolation, max_size, antialias) -> BoundingBox:
        interpolation, antialias  # unused
        output = self._F.resize_bounding_box(self, size, image_size=self.image_size, max_size=max_size)
        return BoundingBox.new_like(self, output, image_size=size, dtype=output.dtype)

    def crop(self, top: int, left: int, height: int, width: int) -> BoundingBox:
        output = self._F.crop_bounding_box(self, self.format, top, left)
        return BoundingBox.new_like(self, output, image_size=(height, width))

    def center_crop(self, output_size) -> BoundingBox:
        output = self._F.center_crop_bounding_box(
            self, format=self.format, output_size=output_size, image_size=self.image_size
        )
        return BoundingBox.new_like(self, output, image_size=output_size)

    def resized_crop(self, top, left, height, width, *, size, interpolation, antialias) -> BoundingBox:
        interpolation, antialias  # unused
        output = self._F.resized_crop_bounding_box(self, self.format, top, left, height, width, size=size)
        return BoundingBox.new_like(self, output, image_size=size, dtype=output.dtype)

    def pad(self, padding, *, fill, padding_mode) -> BoundingBox:
        fill  # unused
        if padding_mode not in ["constant"]:
            raise ValueError(f"Padding mode '{padding_mode}' is not supported with bounding boxes")

        output = self._F.pad_bounding_box(self, padding, format=self.format)

        # Update output image size:
        # TODO: remove the import below and make _parse_pad_padding available
        from torchvision.transforms.functional_tensor import _parse_pad_padding

        left, top, right, bottom = _parse_pad_padding(padding)
        height, width = self.image_size
        height += top + bottom
        width += left + right

        return BoundingBox.new_like(self, output, image_size=(height, width))

    def rotate(self, angle, *, interpolation, expand, fill, center) -> BoundingBox:
        interpolation, fill  # unused
        output = self._F.rotate_bounding_box(
            self, format=self.format, image_size=self.image_size, angle=angle, expand=expand, center=center
        )
        # TODO: update output image size if expand is True
        if expand:
            raise RuntimeError("Not yet implemented")
        return BoundingBox.new_like(self, output, dtype=output.dtype)

    def affine(self, angle, *, translate, scale, shear, interpolation, fill, center) -> BoundingBox:
        interpolation, fill  # unused
        output = self._F.affine_bounding_box(
            self,
            self.format,
            self.image_size,
            angle,
            translate=translate,
            scale=scale,
            shear=shear,
            center=center,
        )
        return BoundingBox.new_like(self, output, dtype=output.dtype)

    def perspective(self, perspective_coeffs, *, interpolation, fill) -> BoundingBox:
        interpolation, fill  # unused
        output = self._F.perspective_bounding_box(self, self.format, perspective_coeffs)
        return BoundingBox.new_like(self, output, dtype=output.dtype)

    def erase(self, *args) -> BoundingBox:
        raise TypeError("Erase transformation does not support bounding boxes")

    def mixup(self, *args) -> BoundingBox:
        raise TypeError("Mixup transformation does not support bounding boxes")

    def cutmix(self, *args) -> BoundingBox:
        raise TypeError("Cutmix transformation does not support bounding boxes")
