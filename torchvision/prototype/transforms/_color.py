import collections.abc
from typing import Any, Dict, Optional, Sequence, Tuple, TypeVar, Union

import PIL.Image
import torch
from torchvision.prototype import features
from torchvision.prototype.transforms import functional as F, Transform

from ._transform import _RandomApplyTransform
from ._utils import query_chw

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

    @staticmethod
    def _generate_value(left: float, right: float) -> float:
        return float(torch.distributions.Uniform(left, right).sample())

    def _get_params(self, sample: Any) -> Dict[str, Any]:
        fn_idx = torch.randperm(4)

        b = None if self.brightness is None else self._generate_value(self.brightness[0], self.brightness[1])
        c = None if self.contrast is None else self._generate_value(self.contrast[0], self.contrast[1])
        s = None if self.saturation is None else self._generate_value(self.saturation[0], self.saturation[1])
        h = None if self.hue is None else self._generate_value(self.hue[0], self.hue[1])

        return dict(fn_idx=fn_idx, brightness_factor=b, contrast_factor=c, saturation_factor=s, hue_factor=h)

    def _transform(self, inpt: Any, params: Dict[str, Any]) -> Any:
        output = inpt
        brightness_factor = params["brightness_factor"]
        contrast_factor = params["contrast_factor"]
        saturation_factor = params["saturation_factor"]
        hue_factor = params["hue_factor"]
        for fn_id in params["fn_idx"]:
            if fn_id == 0 and brightness_factor is not None:
                output = F.adjust_brightness(output, brightness_factor=brightness_factor)
            elif fn_id == 1 and contrast_factor is not None:
                output = F.adjust_contrast(output, contrast_factor=contrast_factor)
            elif fn_id == 2 and saturation_factor is not None:
                output = F.adjust_saturation(output, saturation_factor=saturation_factor)
            elif fn_id == 3 and hue_factor is not None:
                output = F.adjust_hue(output, hue_factor=hue_factor)
        return output


class RandomPhotometricDistort(Transform):
    _transformed_types = (features.Image, PIL.Image.Image, features.is_simple_tensor)

    def __init__(
        self,
        contrast: Tuple[float, float] = (0.5, 1.5),
        saturation: Tuple[float, float] = (0.5, 1.5),
        hue: Tuple[float, float] = (-0.05, 0.05),
        brightness: Tuple[float, float] = (0.875, 1.125),
        p: float = 0.5,
    ):
        super().__init__()
        self.brightness = brightness
        self.contrast = contrast
        self.hue = hue
        self.saturation = saturation
        self.p = p

    def _get_params(self, sample: Any) -> Dict[str, Any]:
        num_channels, _, _ = query_chw(sample)
        return dict(
            zip(
                ["brightness", "contrast1", "saturation", "hue", "contrast2"],
                (torch.rand(5) < self.p).tolist(),
            ),
            contrast_before=bool(torch.rand(()) < 0.5),
            channel_permutation=torch.randperm(num_channels) if torch.rand(()) < self.p else None,
        )

    def _permute_channels(self, inpt: Any, *, permutation: torch.Tensor) -> Any:
        if isinstance(inpt, PIL.Image.Image):
            inpt = F.to_image_tensor(inpt)

        output = inpt[..., permutation, :, :]

        if isinstance(inpt, features.Image):
            output = features.Image.new_like(inpt, output, color_space=features.ColorSpace.OTHER)
        elif isinstance(inpt, PIL.Image.Image):
            output = F.to_image_pil(output)

        return output

    def _transform(self, inpt: Any, params: Dict[str, Any]) -> Any:
        if params["brightness"]:
            inpt = F.adjust_brightness(
                inpt, brightness_factor=ColorJitter._generate_value(self.brightness[0], self.brightness[1])
            )
        if params["contrast1"] and params["contrast_before"]:
            inpt = F.adjust_contrast(
                inpt, contrast_factor=ColorJitter._generate_value(self.contrast[0], self.contrast[1])
            )
        if params["saturation"]:
            inpt = F.adjust_saturation(
                inpt, saturation_factor=ColorJitter._generate_value(self.saturation[0], self.saturation[1])
            )
        if params["hue"]:
            inpt = F.adjust_hue(inpt, hue_factor=ColorJitter._generate_value(self.hue[0], self.hue[1]))
        if params["contrast2"] and not params["contrast_before"]:
            inpt = F.adjust_contrast(
                inpt, contrast_factor=ColorJitter._generate_value(self.contrast[0], self.contrast[1])
            )
        if params["channel_permutation"] is not None:
            inpt = self._permute_channels(inpt, permutation=params["channel_permutation"])
        return inpt


class RandomEqualize(_RandomApplyTransform):
    def _transform(self, inpt: Any, params: Dict[str, Any]) -> Any:
        return F.equalize(inpt)


class RandomInvert(_RandomApplyTransform):
    def _transform(self, inpt: Any, params: Dict[str, Any]) -> Any:
        return F.invert(inpt)


class RandomPosterize(_RandomApplyTransform):
    def __init__(self, bits: int, p: float = 0.5) -> None:
        super().__init__(p=p)
        self.bits = bits

    def _transform(self, inpt: Any, params: Dict[str, Any]) -> Any:
        return F.posterize(inpt, bits=self.bits)


class RandomSolarize(_RandomApplyTransform):
    def __init__(self, threshold: float, p: float = 0.5) -> None:
        super().__init__(p=p)
        self.threshold = threshold

    def _transform(self, inpt: Any, params: Dict[str, Any]) -> Any:
        return F.solarize(inpt, threshold=self.threshold)


class RandomAutocontrast(_RandomApplyTransform):
    def _transform(self, inpt: Any, params: Dict[str, Any]) -> Any:
        return F.autocontrast(inpt)


class RandomAdjustSharpness(_RandomApplyTransform):
    def __init__(self, sharpness_factor: float, p: float = 0.5) -> None:
        super().__init__(p=p)
        self.sharpness_factor = sharpness_factor

    def _transform(self, inpt: Any, params: Dict[str, Any]) -> Any:
        return F.adjust_sharpness(inpt, sharpness_factor=self.sharpness_factor)
