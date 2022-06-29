from __future__ import annotations

import warnings
from typing import Any, List, Optional, Union, Tuple, cast

import torch
from torchvision._utils import StrEnum
from torchvision.transforms.functional import to_pil_image, InterpolationMode
from torchvision.utils import draw_bounding_boxes
from torchvision.utils import make_grid

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

    def horizontal_flip(self) -> Image:
        output = self._F.horizontal_flip_image_tensor(self)  # type: ignore[attr-defined]
        return Image.new_like(self, output)

    def vertical_flip(self) -> Image:
        output = self._F.vertical_flip_image_tensor(self)  # type: ignore[attr-defined]
        return Image.new_like(self, output)

    def resize(  # type: ignore[override]
        self,
        size: List[int],
        interpolation: InterpolationMode = InterpolationMode.BILINEAR,
        max_size: Optional[int] = None,
        antialias: bool = False,
    ) -> Image:
        output = self._F.resize_image_tensor(  # type: ignore[attr-defined]
            self, size, interpolation=interpolation, max_size=max_size, antialias=antialias
        )
        return Image.new_like(self, output)

    def crop(self, top: int, left: int, height: int, width: int) -> Image:
        output = self._F.crop_image_tensor(self, top, left, height, width)  # type: ignore[attr-defined]
        return Image.new_like(self, output)

    def center_crop(self, output_size: List[int]) -> Image:
        output = self._F.center_crop_image_tensor(self, output_size=output_size)  # type: ignore[attr-defined]
        return Image.new_like(self, output)

    def resized_crop(
        self,
        top: int,
        left: int,
        height: int,
        width: int,
        size: List[int],
        interpolation: InterpolationMode = InterpolationMode.BILINEAR,
        antialias: bool = False,
    ) -> Image:
        output = self._F.resized_crop_image_tensor(  # type: ignore[attr-defined]
            self, top, left, height, width, size=list(size), interpolation=interpolation, antialias=antialias
        )
        return Image.new_like(self, output)

    def pad(self, padding: List[int], fill: int = 0, padding_mode: str = "constant") -> Image:
        output = self._F.pad_image_tensor(self, padding, fill=fill, padding_mode=padding_mode)  # type: ignore[attr-defined]
        return Image.new_like(self, output)

    def rotate(
        self,
        angle: float,
        interpolation: InterpolationMode = InterpolationMode.NEAREST,
        expand: bool = False,
        fill: Optional[List[float]] = None,
        center: Optional[List[float]] = None,
    ) -> Image:
        output = self._F.rotate_image_tensor(  # type: ignore[attr-defined]
            self, angle, interpolation=interpolation, expand=expand, fill=fill, center=center
        )
        return Image.new_like(self, output)

    def affine(
        self,
        angle: float,
        translate: List[float],
        scale: float,
        shear: List[float],
        interpolation: InterpolationMode = InterpolationMode.NEAREST,
        fill: Optional[List[float]] = None,
        center: Optional[List[float]] = None,
    ) -> Image:
        output = self._F.affine_image_tensor(  # type: ignore[attr-defined]
            self,
            angle,
            translate=translate,
            scale=scale,
            shear=shear,
            interpolation=interpolation,
            fill=fill,
            center=center,
        )
        return Image.new_like(self, output)

    def perspective(
        self,
        perspective_coeffs: List[float],
        interpolation: InterpolationMode = InterpolationMode.BILINEAR,
        fill: Optional[List[float]] = None,
    ) -> Image:
        output = self._F.perspective_image_tensor(self, perspective_coeffs, interpolation=interpolation, fill=fill)  # type: ignore[attr-defined]
        return Image.new_like(self, output)

    def adjust_brightness(self, brightness_factor: float) -> Image:
        output = self._F.adjust_brightness_image_tensor(self, brightness_factor=brightness_factor)  # type: ignore[attr-defined]
        return Image.new_like(self, output)

    def adjust_saturation(self, saturation_factor: float) -> Image:
        output = self._F.adjust_saturation_image_tensor(self, saturation_factor=saturation_factor)  # type: ignore[attr-defined]
        return Image.new_like(self, output)

    def adjust_contrast(self, contrast_factor: float) -> Image:
        output = self._F.adjust_contrast_image_tensor(self, contrast_factor=contrast_factor)  # type: ignore[attr-defined]
        return Image.new_like(self, output)

    def adjust_sharpness(self, sharpness_factor: float) -> Image:
        output = self._F.adjust_sharpness_image_tensor(self, sharpness_factor=sharpness_factor)  # type: ignore[attr-defined]
        return Image.new_like(self, output)

    def adjust_hue(self, hue_factor: float) -> Image:
        output = self._F.adjust_hue_image_tensor(self, hue_factor=hue_factor)  # type: ignore[attr-defined]
        return Image.new_like(self, output)

    def adjust_gamma(self, gamma: float, gain: float = 1) -> Image:
        output = self._F.adjust_gamma_image_tensor(self, gamma=gamma, gain=gain)  # type: ignore[attr-defined]
        return Image.new_like(self, output)

    def posterize(self, bits: int) -> Image:
        output = self._F.posterize_image_tensor(self, bits=bits)  # type: ignore[attr-defined]
        return Image.new_like(self, output)

    def solarize(self, threshold: float) -> Image:
        output = self._F.solarize_image_tensor(self, threshold=threshold)  # type: ignore[attr-defined]
        return Image.new_like(self, output)

    def autocontrast(self) -> Image:
        output = self._F.autocontrast_image_tensor(self)  # type: ignore[attr-defined]
        return Image.new_like(self, output)

    def equalize(self) -> Image:
        output = self._F.equalize_image_tensor(self)  # type: ignore[attr-defined]
        return Image.new_like(self, output)

    def invert(self) -> Image:
        output = self._F.invert_image_tensor(self)  # type: ignore[attr-defined]
        return Image.new_like(self, output)

    def erase(self, i: int, j: int, h: int, w: int, v: torch.Tensor) -> Image:
        output = self._F.erase_image_tensor(self, i, j, h, w, v)  # type: ignore[attr-defined]
        return Image.new_like(self, output)

    def mixup(self, lam: float) -> Image:
        if self.ndim < 4:
            raise ValueError("Need a batch of images")
        output = self.clone()
        output = output.roll(1, -4).mul_(1 - lam).add_(output.mul_(lam))
        return Image.new_like(self, output)

    def cutmix(self, box: Tuple[int, int, int, int], lam_adjusted: float) -> Image:
        if self.ndim < 4:
            raise ValueError("Need a batch of images")
        x1, y1, x2, y2 = box
        image_rolled = self.roll(1, -4)
        output = self.clone()
        output[..., y1:y2, x1:x2] = image_rolled[..., y1:y2, x1:x2]
        return Image.new_like(self, output)
