from __future__ import annotations

import warnings
from typing import Any, cast, List, Optional, Tuple, Union

import PIL.Image
import torch
from torchvision._utils import StrEnum
from torchvision.transforms.functional import InterpolationMode, to_pil_image
from torchvision.utils import draw_bounding_boxes, make_grid

from ._bounding_box import BoundingBox
from ._feature import _Feature, FillTypeJIT


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

    @staticmethod
    def from_tensor_shape(shape: List[int]) -> ColorSpace:
        return _from_tensor_shape(shape)


def _from_tensor_shape(shape: List[int]) -> ColorSpace:
    # Needed as a standalone method for JIT
    ndim = len(shape)
    if ndim < 2:
        return ColorSpace.OTHER
    elif ndim == 2:
        return ColorSpace.GRAY

    num_channels = shape[-3]
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
        data = torch.as_tensor(data, dtype=dtype, device=device)
        if data.ndim < 2:
            raise ValueError
        elif data.ndim == 2:
            data = data.unsqueeze(0)
        image = super().__new__(cls, data, requires_grad=requires_grad)

        if color_space is None:
            color_space = ColorSpace.from_tensor_shape(image.shape)  # type: ignore[arg-type]
            if color_space == ColorSpace.OTHER:
                warnings.warn("Unable to guess a specific color space. Consider passing it explicitly.")
        elif isinstance(color_space, str):
            color_space = ColorSpace.from_str(color_space.upper())
        elif not isinstance(color_space, ColorSpace):
            raise ValueError
        image.color_space = color_space

        return image

    def __repr__(self, *, tensor_contents: Any = None) -> str:  # type: ignore[override]
        return self._make_repr(color_space=self.color_space)

    @classmethod
    def new_like(
        cls, other: Image, data: Any, *, color_space: Optional[Union[ColorSpace, str]] = None, **kwargs: Any
    ) -> Image:
        return super().new_like(
            other, data, color_space=color_space if color_space is not None else other.color_space, **kwargs
        )

    @property
    def image_size(self) -> Tuple[int, int]:
        return cast(Tuple[int, int], tuple(self.shape[-2:]))

    @property
    def num_channels(self) -> int:
        return self.shape[-3]

    def to_color_space(self, color_space: Union[str, ColorSpace], copy: bool = True) -> Image:
        if isinstance(color_space, str):
            color_space = ColorSpace.from_str(color_space.upper())

        return Image.new_like(
            self,
            self._F.convert_color_space_image_tensor(
                self, old_color_space=self.color_space, new_color_space=color_space, copy=copy
            ),
            color_space=color_space,
        )

    def show(self) -> None:
        # TODO: this is useful for developing and debugging but we should remove or at least revisit this before we
        #  promote this out of the prototype state
        to_pil_image(make_grid(self.view(-1, *self.shape[-3:]))).show()

    def draw_bounding_box(self, bounding_box: BoundingBox, **kwargs: Any) -> Image:
        # TODO: this is useful for developing and debugging but we should remove or at least revisit this before we
        #  promote this out of the prototype state
        return Image.new_like(self, draw_bounding_boxes(self, bounding_box.to_format("xyxy").view(-1, 4), **kwargs))

    def horizontal_flip(self) -> Image:
        output = self._F.horizontal_flip_image_tensor(self)
        return Image.new_like(self, output)

    def vertical_flip(self) -> Image:
        output = self._F.vertical_flip_image_tensor(self)
        return Image.new_like(self, output)

    def resize(  # type: ignore[override]
        self,
        size: List[int],
        interpolation: InterpolationMode = InterpolationMode.BILINEAR,
        max_size: Optional[int] = None,
        antialias: bool = False,
    ) -> Image:
        output = self._F.resize_image_tensor(
            self, size, interpolation=interpolation, max_size=max_size, antialias=antialias
        )
        return Image.new_like(self, output)

    def crop(self, top: int, left: int, height: int, width: int) -> Image:
        output = self._F.crop_image_tensor(self, top, left, height, width)
        return Image.new_like(self, output)

    def center_crop(self, output_size: List[int]) -> Image:
        output = self._F.center_crop_image_tensor(self, output_size=output_size)
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
        output = self._F.resized_crop_image_tensor(
            self, top, left, height, width, size=list(size), interpolation=interpolation, antialias=antialias
        )
        return Image.new_like(self, output)

    def pad(
        self,
        padding: Union[int, List[int]],
        fill: FillTypeJIT = None,
        padding_mode: str = "constant",
    ) -> Image:
        output = self._F.pad_image_tensor(self, padding, fill=fill, padding_mode=padding_mode)
        return Image.new_like(self, output)

    def rotate(
        self,
        angle: float,
        interpolation: InterpolationMode = InterpolationMode.NEAREST,
        expand: bool = False,
        fill: FillTypeJIT = None,
        center: Optional[List[float]] = None,
    ) -> Image:
        output = self._F._geometry.rotate_image_tensor(
            self, angle, interpolation=interpolation, expand=expand, fill=fill, center=center
        )
        return Image.new_like(self, output)

    def affine(
        self,
        angle: Union[int, float],
        translate: List[float],
        scale: float,
        shear: List[float],
        interpolation: InterpolationMode = InterpolationMode.NEAREST,
        fill: FillTypeJIT = None,
        center: Optional[List[float]] = None,
    ) -> Image:
        output = self._F._geometry.affine_image_tensor(
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
        fill: FillTypeJIT = None,
    ) -> Image:
        output = self._F._geometry.perspective_image_tensor(
            self, perspective_coeffs, interpolation=interpolation, fill=fill
        )
        return Image.new_like(self, output)

    def elastic(
        self,
        displacement: torch.Tensor,
        interpolation: InterpolationMode = InterpolationMode.BILINEAR,
        fill: FillTypeJIT = None,
    ) -> Image:
        output = self._F._geometry.elastic_image_tensor(self, displacement, interpolation=interpolation, fill=fill)
        return Image.new_like(self, output)

    def adjust_brightness(self, brightness_factor: float) -> Image:
        output = self._F.adjust_brightness_image_tensor(self, brightness_factor=brightness_factor)
        return Image.new_like(self, output)

    def adjust_saturation(self, saturation_factor: float) -> Image:
        output = self._F.adjust_saturation_image_tensor(self, saturation_factor=saturation_factor)
        return Image.new_like(self, output)

    def adjust_contrast(self, contrast_factor: float) -> Image:
        output = self._F.adjust_contrast_image_tensor(self, contrast_factor=contrast_factor)
        return Image.new_like(self, output)

    def adjust_sharpness(self, sharpness_factor: float) -> Image:
        output = self._F.adjust_sharpness_image_tensor(self, sharpness_factor=sharpness_factor)
        return Image.new_like(self, output)

    def adjust_hue(self, hue_factor: float) -> Image:
        output = self._F.adjust_hue_image_tensor(self, hue_factor=hue_factor)
        return Image.new_like(self, output)

    def adjust_gamma(self, gamma: float, gain: float = 1) -> Image:
        output = self._F.adjust_gamma_image_tensor(self, gamma=gamma, gain=gain)
        return Image.new_like(self, output)

    def posterize(self, bits: int) -> Image:
        output = self._F.posterize_image_tensor(self, bits=bits)
        return Image.new_like(self, output)

    def solarize(self, threshold: float) -> Image:
        output = self._F.solarize_image_tensor(self, threshold=threshold)
        return Image.new_like(self, output)

    def autocontrast(self) -> Image:
        output = self._F.autocontrast_image_tensor(self)
        return Image.new_like(self, output)

    def equalize(self) -> Image:
        output = self._F.equalize_image_tensor(self)
        return Image.new_like(self, output)

    def invert(self) -> Image:
        output = self._F.invert_image_tensor(self)
        return Image.new_like(self, output)

    def gaussian_blur(self, kernel_size: List[int], sigma: Optional[List[float]] = None) -> Image:
        output = self._F.gaussian_blur_image_tensor(self, kernel_size=kernel_size, sigma=sigma)
        return Image.new_like(self, output)


ImageType = Union[torch.Tensor, PIL.Image.Image, Image]
ImageTypeJIT = torch.Tensor
LegacyImageType = Union[torch.Tensor, PIL.Image.Image]
LegacyImageTypeJIT = torch.Tensor
TensorImageType = Union[torch.Tensor, Image]
TensorImageTypeJIT = torch.Tensor
