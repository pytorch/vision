import math
from typing import Any, Dict, Tuple, Optional, Callable, List, cast, TypeVar

import PIL.Image
import torch
from torchvision.prototype import features
from torchvision.prototype.transforms import Transform, InterpolationMode, AutoAugmentPolicy, functional as F
from torchvision.prototype.utils._internal import apply_recursively

from ._utils import query_image

K = TypeVar("K")
V = TypeVar("V")


class _AutoAugmentBase(Transform):
    def __init__(
        self, *, interpolation: InterpolationMode = InterpolationMode.NEAREST, fill: Optional[List[float]] = None
    ) -> None:
        super().__init__()
        self.interpolation = interpolation
        self.fill = fill

    def _get_random_item(self, dct: Dict[K, V]) -> Tuple[K, V]:
        keys = tuple(dct.keys())
        key = keys[int(torch.randint(len(keys), ()))]
        return key, dct[key]

    def _apply_transform(self, sample: Any, transform_id: str, magnitude: float) -> Any:
        def dispatch(
            image_tensor_kernel: Callable,
            image_pil_kernel: Callable,
            input: Any,
            *args: Any,
            **kwargs: Any,
        ) -> Any:
            if isinstance(input, (features.BoundingBox, features.SegmentationMask)):
                raise TypeError(f"{type(input).__name__}'s are not supported by {type(self).__name__}()")
            elif isinstance(input, features.Image):
                output = image_tensor_kernel(input, *args, **kwargs)
                return features.Image.new_like(input, output)
            elif isinstance(input, torch.Tensor):
                return image_tensor_kernel(input, *args, **kwargs)
            elif isinstance(input, PIL.Image.Image):
                return image_pil_kernel(input, *args, **kwargs)
            else:
                return input

        image = query_image(sample)
        num_channels = F.get_image_num_channels(image)

        fill = self.fill
        if isinstance(fill, (int, float)):
            fill = [float(fill)] * num_channels
        elif fill is not None:
            fill = [float(f) for f in fill]

        interpolation = self.interpolation

        def transform(input: Any) -> Any:
            if type(input) in {features.BoundingBox, features.SegmentationMask}:
                raise TypeError(f"{type(input)} is not supported by {type(self).__name__}()")
            elif not (type(input) in {features.Image, torch.Tensor} or isinstance(input, PIL.Image.Image)):
                return input

            if transform_id == "Identity":
                return input
            elif transform_id == "ShearX":
                return dispatch(
                    F.affine_image_tensor,
                    F.affine_image_pil,
                    input,
                    angle=0.0,
                    translate=[0, 0],
                    scale=1.0,
                    shear=[math.degrees(magnitude), 0.0],
                    interpolation=interpolation,
                    fill=fill,
                )
            elif transform_id == "ShearY":
                return dispatch(
                    F.affine_image_tensor,
                    F.affine_image_pil,
                    input,
                    angle=0.0,
                    translate=[0, 0],
                    scale=1.0,
                    shear=[0.0, math.degrees(magnitude)],
                    interpolation=interpolation,
                    fill=fill,
                )
            elif transform_id == "TranslateX":
                return dispatch(
                    F.affine_image_tensor,
                    F.affine_image_pil,
                    input,
                    angle=0.0,
                    translate=[int(magnitude), 0],
                    scale=1.0,
                    shear=[0.0, 0.0],
                    interpolation=interpolation,
                    fill=fill,
                )
            elif transform_id == "TranslateY":
                return dispatch(
                    F.affine_image_tensor,
                    F.affine_image_pil,
                    input,
                    angle=0.0,
                    translate=[0, int(magnitude)],
                    scale=1.0,
                    shear=[0.0, 0.0],
                    interpolation=interpolation,
                    fill=fill,
                )
            elif transform_id == "Rotate":
                return dispatch(F.rotate_image_tensor, F.rotate_image_pil, input, angle=magnitude)
            elif transform_id == "Brightness":
                return dispatch(
                    F.adjust_brightness_image_tensor,
                    F.adjust_brightness_image_pil,
                    input,
                    brightness_factor=1.0 + magnitude,
                )
            elif transform_id == "Color":
                return dispatch(
                    F.adjust_saturation_image_tensor,
                    F.adjust_saturation_image_pil,
                    input,
                    saturation_factor=1.0 + magnitude,
                )
            elif transform_id == "Contrast":
                return dispatch(
                    F.adjust_contrast_image_tensor, F.adjust_contrast_image_pil, input, contrast_factor=1.0 + magnitude
                )
            elif transform_id == "Sharpness":
                return dispatch(
                    F.adjust_sharpness_image_tensor,
                    F.adjust_sharpness_image_pil,
                    input,
                    sharpness_factor=1.0 + magnitude,
                )
            elif transform_id == "Posterize":
                return dispatch(F.posterize_image_tensor, F.posterize_image_pil, input, bits=int(magnitude))
            elif transform_id == "Solarize":
                return dispatch(F.solarize_image_tensor, F.solarize_image_pil, input, threshold=magnitude)
            elif transform_id == "AutoContrast":
                return dispatch(F.autocontrast_image_tensor, F.autocontrast_image_pil, input)
            elif transform_id == "Equalize":
                return dispatch(F.equalize_image_tensor, F.equalize_image_pil, input)
            elif transform_id == "Invert":
                return dispatch(F.invert_image_tensor, F.invert_image_pil, input)
            else:
                raise ValueError(f"No transform available for {transform_id}")

        return apply_recursively(transform, sample)


class AutoAugment(_AutoAugmentBase):
    _AUGMENTATION_SPACE = {
        "ShearX": (lambda num_bins, image_size: torch.linspace(0.0, 0.3, num_bins), True),
        "ShearY": (lambda num_bins, image_size: torch.linspace(0.0, 0.3, num_bins), True),
        "TranslateX": (lambda num_bins, image_size: torch.linspace(0.0, 150.0 / 331.0 * image_size[0], num_bins), True),
        "TranslateY": (lambda num_bins, image_size: torch.linspace(0.0, 150.0 / 331.0 * image_size[1], num_bins), True),
        "Rotate": (lambda num_bins, image_size: torch.linspace(0.0, 30.0, num_bins), True),
        "Brightness": (lambda num_bins, image_size: torch.linspace(0.0, 0.9, num_bins), True),
        "Color": (lambda num_bins, image_size: torch.linspace(0.0, 0.9, num_bins), True),
        "Contrast": (lambda num_bins, image_size: torch.linspace(0.0, 0.9, num_bins), True),
        "Sharpness": (lambda num_bins, image_size: torch.linspace(0.0, 0.9, num_bins), True),
        "Posterize": (
            lambda num_bins, image_size: cast(torch.Tensor, 8 - (torch.arange(num_bins) / ((num_bins - 1) / 4)))
            .round()
            .int(),
            False,
        ),
        "Solarize": (lambda num_bins, image_size: torch.linspace(255.0, 0.0, num_bins), False),
        "AutoContrast": (lambda num_bins, image_size: None, False),
        "Equalize": (lambda num_bins, image_size: None, False),
        "Invert": (lambda num_bins, image_size: None, False),
    }

    def __init__(self, policy: AutoAugmentPolicy = AutoAugmentPolicy.IMAGENET, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.policy = policy
        self._policies = self._get_policies(policy)

    def _get_policies(
        self, policy: AutoAugmentPolicy
    ) -> List[Tuple[Tuple[str, float, Optional[int]], Tuple[str, float, Optional[int]]]]:
        if policy == AutoAugmentPolicy.IMAGENET:
            return [
                (("Posterize", 0.4, 8), ("Rotate", 0.6, 9)),
                (("Solarize", 0.6, 5), ("AutoContrast", 0.6, None)),
                (("Equalize", 0.8, None), ("Equalize", 0.6, None)),
                (("Posterize", 0.6, 7), ("Posterize", 0.6, 6)),
                (("Equalize", 0.4, None), ("Solarize", 0.2, 4)),
                (("Equalize", 0.4, None), ("Rotate", 0.8, 8)),
                (("Solarize", 0.6, 3), ("Equalize", 0.6, None)),
                (("Posterize", 0.8, 5), ("Equalize", 1.0, None)),
                (("Rotate", 0.2, 3), ("Solarize", 0.6, 8)),
                (("Equalize", 0.6, None), ("Posterize", 0.4, 6)),
                (("Rotate", 0.8, 8), ("Color", 0.4, 0)),
                (("Rotate", 0.4, 9), ("Equalize", 0.6, None)),
                (("Equalize", 0.0, None), ("Equalize", 0.8, None)),
                (("Invert", 0.6, None), ("Equalize", 1.0, None)),
                (("Color", 0.6, 4), ("Contrast", 1.0, 8)),
                (("Rotate", 0.8, 8), ("Color", 1.0, 2)),
                (("Color", 0.8, 8), ("Solarize", 0.8, 7)),
                (("Sharpness", 0.4, 7), ("Invert", 0.6, None)),
                (("ShearX", 0.6, 5), ("Equalize", 1.0, None)),
                (("Color", 0.4, 0), ("Equalize", 0.6, None)),
                (("Equalize", 0.4, None), ("Solarize", 0.2, 4)),
                (("Solarize", 0.6, 5), ("AutoContrast", 0.6, None)),
                (("Invert", 0.6, None), ("Equalize", 1.0, None)),
                (("Color", 0.6, 4), ("Contrast", 1.0, 8)),
                (("Equalize", 0.8, None), ("Equalize", 0.6, None)),
            ]
        elif policy == AutoAugmentPolicy.CIFAR10:
            return [
                (("Invert", 0.1, None), ("Contrast", 0.2, 6)),
                (("Rotate", 0.7, 2), ("TranslateX", 0.3, 9)),
                (("Sharpness", 0.8, 1), ("Sharpness", 0.9, 3)),
                (("ShearY", 0.5, 8), ("TranslateY", 0.7, 9)),
                (("AutoContrast", 0.5, None), ("Equalize", 0.9, None)),
                (("ShearY", 0.2, 7), ("Posterize", 0.3, 7)),
                (("Color", 0.4, 3), ("Brightness", 0.6, 7)),
                (("Sharpness", 0.3, 9), ("Brightness", 0.7, 9)),
                (("Equalize", 0.6, None), ("Equalize", 0.5, None)),
                (("Contrast", 0.6, 7), ("Sharpness", 0.6, 5)),
                (("Color", 0.7, 7), ("TranslateX", 0.5, 8)),
                (("Equalize", 0.3, None), ("AutoContrast", 0.4, None)),
                (("TranslateY", 0.4, 3), ("Sharpness", 0.2, 6)),
                (("Brightness", 0.9, 6), ("Color", 0.2, 8)),
                (("Solarize", 0.5, 2), ("Invert", 0.0, None)),
                (("Equalize", 0.2, None), ("AutoContrast", 0.6, None)),
                (("Equalize", 0.2, None), ("Equalize", 0.6, None)),
                (("Color", 0.9, 9), ("Equalize", 0.6, None)),
                (("AutoContrast", 0.8, None), ("Solarize", 0.2, 8)),
                (("Brightness", 0.1, 3), ("Color", 0.7, 0)),
                (("Solarize", 0.4, 5), ("AutoContrast", 0.9, None)),
                (("TranslateY", 0.9, 9), ("TranslateY", 0.7, 9)),
                (("AutoContrast", 0.9, None), ("Solarize", 0.8, 3)),
                (("Equalize", 0.8, None), ("Invert", 0.1, None)),
                (("TranslateY", 0.7, 9), ("AutoContrast", 0.9, None)),
            ]
        elif policy == AutoAugmentPolicy.SVHN:
            return [
                (("ShearX", 0.9, 4), ("Invert", 0.2, None)),
                (("ShearY", 0.9, 8), ("Invert", 0.7, None)),
                (("Equalize", 0.6, None), ("Solarize", 0.6, 6)),
                (("Invert", 0.9, None), ("Equalize", 0.6, None)),
                (("Equalize", 0.6, None), ("Rotate", 0.9, 3)),
                (("ShearX", 0.9, 4), ("AutoContrast", 0.8, None)),
                (("ShearY", 0.9, 8), ("Invert", 0.4, None)),
                (("ShearY", 0.9, 5), ("Solarize", 0.2, 6)),
                (("Invert", 0.9, None), ("AutoContrast", 0.8, None)),
                (("Equalize", 0.6, None), ("Rotate", 0.9, 3)),
                (("ShearX", 0.9, 4), ("Solarize", 0.3, 3)),
                (("ShearY", 0.8, 8), ("Invert", 0.7, None)),
                (("Equalize", 0.9, None), ("TranslateY", 0.6, 6)),
                (("Invert", 0.9, None), ("Equalize", 0.6, None)),
                (("Contrast", 0.3, 3), ("Rotate", 0.8, 4)),
                (("Invert", 0.8, None), ("TranslateY", 0.0, 2)),
                (("ShearY", 0.7, 6), ("Solarize", 0.4, 8)),
                (("Invert", 0.6, None), ("Rotate", 0.8, 4)),
                (("ShearY", 0.3, 7), ("TranslateX", 0.9, 3)),
                (("ShearX", 0.1, 6), ("Invert", 0.6, None)),
                (("Solarize", 0.7, 2), ("TranslateY", 0.6, 7)),
                (("ShearY", 0.8, 4), ("Invert", 0.8, None)),
                (("ShearX", 0.7, 9), ("TranslateY", 0.8, 3)),
                (("ShearY", 0.8, 5), ("AutoContrast", 0.7, None)),
                (("ShearX", 0.7, 2), ("Invert", 0.1, None)),
            ]
        else:
            raise ValueError(f"The provided policy {policy} is not recognized.")

    def forward(self, *inputs: Any) -> Any:
        sample = inputs if len(inputs) > 1 else inputs[0]

        image = query_image(sample)
        image_size = F.get_image_size(image)

        policy = self._policies[int(torch.randint(len(self._policies), ()))]

        for transform_id, probability, magnitude_idx in policy:
            if not torch.rand(()) <= probability:
                continue

            magnitudes_fn, signed = self._AUGMENTATION_SPACE[transform_id]

            magnitudes = magnitudes_fn(10, image_size)
            if magnitudes is not None:
                magnitude = float(magnitudes[magnitude_idx])
                if signed and torch.rand(()) <= 0.5:
                    magnitude *= -1
            else:
                magnitude = 0.0

            sample = self._apply_transform(sample, transform_id, magnitude)

        return sample


class RandAugment(_AutoAugmentBase):
    _AUGMENTATION_SPACE = {
        "Identity": (lambda num_bins, image_size: None, False),
        "ShearX": (lambda num_bins, image_size: torch.linspace(0.0, 0.3, num_bins), True),
        "ShearY": (lambda num_bins, image_size: torch.linspace(0.0, 0.3, num_bins), True),
        "TranslateX": (lambda num_bins, image_size: torch.linspace(0.0, 150.0 / 331.0 * image_size[0], num_bins), True),
        "TranslateY": (lambda num_bins, image_size: torch.linspace(0.0, 150.0 / 331.0 * image_size[1], num_bins), True),
        "Rotate": (lambda num_bins, image_size: torch.linspace(0.0, 30.0, num_bins), True),
        "Brightness": (lambda num_bins, image_size: torch.linspace(0.0, 0.9, num_bins), True),
        "Color": (lambda num_bins, image_size: torch.linspace(0.0, 0.9, num_bins), True),
        "Contrast": (lambda num_bins, image_size: torch.linspace(0.0, 0.9, num_bins), True),
        "Sharpness": (lambda num_bins, image_size: torch.linspace(0.0, 0.9, num_bins), True),
        "Posterize": (
            lambda num_bins, image_size: cast(torch.Tensor, 8 - (torch.arange(num_bins) / ((num_bins - 1) / 4)))
            .round()
            .int(),
            False,
        ),
        "Solarize": (lambda num_bins, image_size: torch.linspace(255.0, 0.0, num_bins), False),
        "AutoContrast": (lambda num_bins, image_size: None, False),
        "Equalize": (lambda num_bins, image_size: None, False),
    }

    def __init__(self, *, num_ops: int = 2, magnitude: int = 9, num_magnitude_bins: int = 31, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.num_ops = num_ops
        self.magnitude = magnitude
        self.num_magnitude_bins = num_magnitude_bins

    def forward(self, *inputs: Any) -> Any:
        sample = inputs if len(inputs) > 1 else inputs[0]

        image = query_image(sample)
        image_size = F.get_image_size(image)

        for _ in range(self.num_ops):
            transform_id, (magnitudes_fn, signed) = self._get_random_item(self._AUGMENTATION_SPACE)

            magnitudes = magnitudes_fn(self.num_magnitude_bins, image_size)
            if magnitudes is not None:
                magnitude = float(magnitudes[int(torch.randint(self.num_magnitude_bins, ()))])
                if signed and torch.rand(()) <= 0.5:
                    magnitude *= -1
            else:
                magnitude = 0.0

            sample = self._apply_transform(sample, transform_id, magnitude)

        return sample


class TrivialAugmentWide(_AutoAugmentBase):
    _AUGMENTATION_SPACE = {
        "Identity": (lambda num_bins, image_size: None, False),
        "ShearX": (lambda num_bins, image_size: torch.linspace(0.0, 0.99, num_bins), True),
        "ShearY": (lambda num_bins, image_size: torch.linspace(0.0, 0.99, num_bins), True),
        "TranslateX": (lambda num_bins, image_size: torch.linspace(0.0, 32.0, num_bins), True),
        "TranslateY": (lambda num_bins, image_size: torch.linspace(0.0, 32.0, num_bins), True),
        "Rotate": (lambda num_bins, image_size: torch.linspace(0.0, 135.0, num_bins), True),
        "Brightness": (lambda num_bins, image_size: torch.linspace(0.0, 0.99, num_bins), True),
        "Color": (lambda num_bins, image_size: torch.linspace(0.0, 0.99, num_bins), True),
        "Contrast": (lambda num_bins, image_size: torch.linspace(0.0, 0.99, num_bins), True),
        "Sharpness": (lambda num_bins, image_size: torch.linspace(0.0, 0.99, num_bins), True),
        "Posterize": (
            lambda num_bins, image_size: cast(torch.Tensor, 8 - (torch.arange(num_bins) / ((num_bins - 1) / 6)))
            .round()
            .int(),
            False,
        ),
        "Solarize": (lambda num_bins, image_size: torch.linspace(255.0, 0.0, num_bins), False),
        "AutoContrast": (lambda num_bins, image_size: None, False),
        "Equalize": (lambda num_bins, image_size: None, False),
    }

    def __init__(self, *, num_magnitude_bins: int = 31, **kwargs: Any):
        super().__init__(**kwargs)
        self.num_magnitude_bins = num_magnitude_bins

    def forward(self, *inputs: Any) -> Any:
        sample = inputs if len(inputs) > 1 else inputs[0]

        image = query_image(sample)
        image_size = F.get_image_size(image)

        transform_id, (magnitudes_fn, signed) = self._get_random_item(self._AUGMENTATION_SPACE)

        magnitudes = magnitudes_fn(self.num_magnitude_bins, image_size)
        if magnitudes is not None:
            magnitude = float(magnitudes[int(torch.randint(self.num_magnitude_bins, ()))])
            if signed and torch.rand(()) <= 0.5:
                magnitude *= -1
        else:
            magnitude = 0.0

        return self._apply_transform(sample, transform_id, magnitude)
