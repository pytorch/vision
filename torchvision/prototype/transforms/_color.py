import collections.abc
import functools
from typing import Any, Dict, Union, Tuple, Optional, Sequence, Callable, TypeVar

import PIL.Image
import torch
from torchvision.prototype import features
from torchvision.prototype.transforms import Transform, functional as F

from ._utils import is_simple_tensor

T = TypeVar("T", features.Image, torch.Tensor, PIL.Image.Image)


class ColorJitter(Transform):
    def __init__(
        self,
        brightness: Optional[Union[float, Sequence[float]]] = None,
        contrast: Optional[Union[float, Sequence[float]]] = None,
        saturation: Optional[Union[float, Sequence[float]]] = None,
        hue: Optional[Union[float, Sequence[float]]] = None,
    ) -> None:
        super().__init__()
        self.brightness = self._check_input(brightness, "brightness")
        self.contrast = self._check_input(contrast, "contrast")
        self.saturation = self._check_input(saturation, "saturation")
        self.hue = self._check_input(hue, "hue", center=0, bound=(-0.5, 0.5), clip_first_on_zero=False)

    def _check_input(
        self,
        value: Optional[Union[float, Sequence[float]]],
        name: str,
        center: float = 1.0,
        bound: Tuple[float, float] = (0, float("inf")),
        clip_first_on_zero: bool = True,
    ) -> Optional[Tuple[float, float]]:
        if value is None:
            return None

        if isinstance(value, float):
            if value < 0:
                raise ValueError(f"If {name} is a single number, it must be non negative.")
            value = [center - value, center + value]
            if clip_first_on_zero:
                value[0] = max(value[0], 0.0)
        elif isinstance(value, collections.abc.Sequence) and len(value) == 2:
            if not bound[0] <= value[0] <= value[1] <= bound[1]:
                raise ValueError(f"{name} values should be between {bound}")
        else:
            raise TypeError(f"{name} should be a single number or a sequence with length 2.")

        return None if value[0] == value[1] == center else (float(value[0]), float(value[1]))

    def _image_transform(
        self,
        input: T,
        *,
        kernel_tensor: Callable[..., torch.Tensor],
        kernel_pil: Callable[..., PIL.Image.Image],
        **kwargs: Any,
    ) -> T:
        if isinstance(input, features.Image):
            output = kernel_tensor(input, **kwargs)
            return features.Image.new_like(input, output)
        elif is_simple_tensor(input):
            return kernel_tensor(input, **kwargs)
        elif isinstance(input, PIL.Image.Image):
            return kernel_pil(input, **kwargs)  # type: ignore[no-any-return]
        else:
            raise RuntimeError

    def _get_params(self, sample: Any) -> Dict[str, Any]:
        image_transforms = []
        if self.brightness is not None:
            image_transforms.append(
                functools.partial(
                    self._image_transform,
                    kernel_tensor=F.adjust_brightness_image_tensor,
                    kernel_pil=F.adjust_brightness_image_pil,
                    brightness_factor=float(
                        torch.distributions.Uniform(self.brightness[0], self.brightness[1]).sample()
                    ),
                )
            )
        if self.contrast is not None:
            image_transforms.append(
                functools.partial(
                    self._image_transform,
                    kernel_tensor=F.adjust_contrast_image_tensor,
                    kernel_pil=F.adjust_contrast_image_pil,
                    contrast_factor=float(torch.distributions.Uniform(self.contrast[0], self.contrast[1]).sample()),
                )
            )
        if self.saturation is not None:
            image_transforms.append(
                functools.partial(
                    self._image_transform,
                    kernel_tensor=F.adjust_saturation_image_tensor,
                    kernel_pil=F.adjust_saturation_image_pil,
                    saturation_factor=float(
                        torch.distributions.Uniform(self.saturation[0], self.saturation[1]).sample()
                    ),
                )
            )
        if self.hue is not None:
            image_transforms.append(
                functools.partial(
                    self._image_transform,
                    kernel_tensor=F.adjust_hue_image_tensor,
                    kernel_pil=F.adjust_hue_image_pil,
                    hue_factor=float(torch.distributions.Uniform(self.hue[0], self.hue[1]).sample()),
                )
            )

        return dict(image_transforms=[image_transforms[idx] for idx in torch.randperm(len(image_transforms))])

    def _transform(self, input: Any, params: Dict[str, Any]) -> Any:
        if not (isinstance(input, (features.Image, PIL.Image.Image)) or is_simple_tensor(input)):
            return input

        for transform in params["image_transforms"]:
            input = transform(input)

        return input
