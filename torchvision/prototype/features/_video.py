from __future__ import annotations

import warnings
from typing import Any, cast, List, Optional, Tuple, Union

import torch
from torchvision.transforms.functional import InterpolationMode

from ._feature import _Feature, FillTypeJIT
from ._image import ColorSpace


class Video(_Feature):
    color_space: ColorSpace

    @classmethod
    def _wrap(cls, tensor: torch.Tensor, *, color_space: ColorSpace) -> Video:
        video = tensor.as_subclass(cls)
        video.color_space = color_space
        return video

    def __new__(
        cls,
        data: Any,
        *,
        color_space: Optional[Union[ColorSpace, str]] = None,
        dtype: Optional[torch.dtype] = None,
        device: Optional[Union[torch.device, str, int]] = None,
        requires_grad: bool = False,
    ) -> Video:
        tensor = cls._to_tensor(data, dtype=dtype, device=device, requires_grad=requires_grad)
        if data.ndim < 4:
            raise ValueError
        video = super().__new__(cls, data, requires_grad=requires_grad)

        if color_space is None:
            color_space = ColorSpace.from_tensor_shape(video.shape)  # type: ignore[arg-type]
            if color_space == ColorSpace.OTHER:
                warnings.warn("Unable to guess a specific color space. Consider passing it explicitly.")
        elif isinstance(color_space, str):
            color_space = ColorSpace.from_str(color_space.upper())
        elif not isinstance(color_space, ColorSpace):
            raise ValueError

        return cls._wrap(tensor, color_space=color_space)

    @classmethod
    def wrap_like(cls, other: Video, tensor: torch.Tensor, *, color_space: Optional[ColorSpace] = None) -> Video:
        return cls._wrap(
            tensor,
            color_space=color_space if color_space is not None else other.color_space,
        )

    def __repr__(self, *, tensor_contents: Any = None) -> str:  # type: ignore[override]
        return self._make_repr(color_space=self.color_space)

    @property
    def spatial_size(self) -> Tuple[int, int]:
        return cast(Tuple[int, int], tuple(self.shape[-2:]))

    @property
    def num_channels(self) -> int:
        return self.shape[-3]

    @property
    def num_frames(self) -> int:
        return self.shape[-4]

    def to_color_space(self, color_space: Union[str, ColorSpace], copy: bool = True) -> Video:
        if isinstance(color_space, str):
            color_space = ColorSpace.from_str(color_space.upper())

        return Video.wrap_like(
            self,
            self._F.convert_color_space_video(
                self, old_color_space=self.color_space, new_color_space=color_space, copy=copy
            ),
            color_space=color_space,
        )

    def horizontal_flip(self) -> Video:
        output = self._F.horizontal_flip_video(self)
        return Video.wrap_like(self, output)

    def vertical_flip(self) -> Video:
        output = self._F.vertical_flip_video(self)
        return Video.wrap_like(self, output)

    def resize(  # type: ignore[override]
        self,
        size: List[int],
        interpolation: InterpolationMode = InterpolationMode.BILINEAR,
        max_size: Optional[int] = None,
        antialias: bool = False,
    ) -> Video:
        output = self._F.resize_video(self, size, interpolation=interpolation, max_size=max_size, antialias=antialias)
        return Video.wrap_like(self, output)

    def crop(self, top: int, left: int, height: int, width: int) -> Video:
        output = self._F.crop_video(self, top, left, height, width)
        return Video.wrap_like(self, output)

    def center_crop(self, output_size: List[int]) -> Video:
        output = self._F.center_crop_video(self, output_size=output_size)
        return Video.wrap_like(self, output)

    def resized_crop(
        self,
        top: int,
        left: int,
        height: int,
        width: int,
        size: List[int],
        interpolation: InterpolationMode = InterpolationMode.BILINEAR,
        antialias: bool = False,
    ) -> Video:
        output = self._F.resized_crop_video(
            self, top, left, height, width, size=list(size), interpolation=interpolation, antialias=antialias
        )
        return Video.wrap_like(self, output)

    def pad(
        self,
        padding: Union[int, List[int]],
        fill: FillTypeJIT = None,
        padding_mode: str = "constant",
    ) -> Video:
        output = self._F.pad_video(self, padding, fill=fill, padding_mode=padding_mode)
        return Video.wrap_like(self, output)

    def rotate(
        self,
        angle: float,
        interpolation: InterpolationMode = InterpolationMode.NEAREST,
        expand: bool = False,
        fill: FillTypeJIT = None,
        center: Optional[List[float]] = None,
    ) -> Video:
        output = self._F._geometry.rotate_video(
            self, angle, interpolation=interpolation, expand=expand, fill=fill, center=center
        )
        return Video.wrap_like(self, output)

    def affine(
        self,
        angle: Union[int, float],
        translate: List[float],
        scale: float,
        shear: List[float],
        interpolation: InterpolationMode = InterpolationMode.NEAREST,
        fill: FillTypeJIT = None,
        center: Optional[List[float]] = None,
    ) -> Video:
        output = self._F._geometry.affine_video(
            self,
            angle,
            translate=translate,
            scale=scale,
            shear=shear,
            interpolation=interpolation,
            fill=fill,
            center=center,
        )
        return Video.wrap_like(self, output)

    def perspective(
        self,
        perspective_coeffs: List[float],
        interpolation: InterpolationMode = InterpolationMode.BILINEAR,
        fill: FillTypeJIT = None,
    ) -> Video:
        output = self._F._geometry.perspective_video(self, perspective_coeffs, interpolation=interpolation, fill=fill)
        return Video.wrap_like(self, output)

    def elastic(
        self,
        displacement: torch.Tensor,
        interpolation: InterpolationMode = InterpolationMode.BILINEAR,
        fill: FillTypeJIT = None,
    ) -> Video:
        output = self._F._geometry.elastic_video(self, displacement, interpolation=interpolation, fill=fill)
        return Video.wrap_like(self, output)

    def adjust_brightness(self, brightness_factor: float) -> Video:
        output = self._F.adjust_brightness_video(self, brightness_factor=brightness_factor)
        return Video.wrap_like(self, output)

    def adjust_saturation(self, saturation_factor: float) -> Video:
        output = self._F.adjust_saturation_video(self, saturation_factor=saturation_factor)
        return Video.wrap_like(self, output)

    def adjust_contrast(self, contrast_factor: float) -> Video:
        output = self._F.adjust_contrast_video(self, contrast_factor=contrast_factor)
        return Video.wrap_like(self, output)

    def adjust_sharpness(self, sharpness_factor: float) -> Video:
        output = self._F.adjust_sharpness_video(self, sharpness_factor=sharpness_factor)
        return Video.wrap_like(self, output)

    def adjust_hue(self, hue_factor: float) -> Video:
        output = self._F.adjust_hue_video(self, hue_factor=hue_factor)
        return Video.wrap_like(self, output)

    def adjust_gamma(self, gamma: float, gain: float = 1) -> Video:
        output = self._F.adjust_gamma_video(self, gamma=gamma, gain=gain)
        return Video.wrap_like(self, output)

    def posterize(self, bits: int) -> Video:
        output = self._F.posterize_video(self, bits=bits)
        return Video.wrap_like(self, output)

    def solarize(self, threshold: float) -> Video:
        output = self._F.solarize_video(self, threshold=threshold)
        return Video.wrap_like(self, output)

    def autocontrast(self) -> Video:
        output = self._F.autocontrast_video(self)
        return Video.wrap_like(self, output)

    def equalize(self) -> Video:
        output = self._F.equalize_video(self)
        return Video.wrap_like(self, output)

    def invert(self) -> Video:
        output = self._F.invert_video(self)
        return Video.wrap_like(self, output)

    def gaussian_blur(self, kernel_size: List[int], sigma: Optional[List[float]] = None) -> Video:
        output = self._F.gaussian_blur_video(self, kernel_size=kernel_size, sigma=sigma)
        return Video.wrap_like(self, output)


VideoType = Union[torch.Tensor, Video]
VideoTypeJIT = torch.Tensor
LegacyVideoType = torch.Tensor
LegacyVideoTypeJIT = torch.Tensor
TensorVideoType = Union[torch.Tensor, Video]
TensorVideoTypeJIT = torch.Tensor
