import functools
import math
from typing import Any, Dict, Tuple, Optional, Callable, List, cast, TypeVar, Union

import PIL.Image
import torch
from torchvision.prototype import features
from torchvision.prototype.transforms import Transform, InterpolationMode, AutoAugmentPolicy, functional as F
from torchvision.prototype.utils._internal import apply_recursively
from torchvision.transforms.functional import pil_to_tensor, to_pil_image

from ._utils import query_images, get_image_dimensions

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

    def _query_image(self, sample: Any) -> Union[PIL.Image.Image, torch.Tensor, features.Image]:
        images = list(query_images(sample))
        if len(images) > 1:
            raise TypeError(
                f"Auto augment transformations are only properly defined for a single image, but found {len(images)}."
            )
        return images[0]

    def _parse_fill(self, sample: Any) -> Optional[List[float]]:
        fill = self.fill

        if fill is None:
            return fill

        image = self._query_image(sample)

        if not isinstance(image, torch.Tensor):
            return fill

        if isinstance(fill, (int, float)):
            num_channels, *_ = get_image_dimensions(image)
            fill = [float(fill)] * num_channels
        else:
            fill = [float(f) for f in fill]

        return fill

    def _dispatch(
        self,
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

    def _apply_transform_to_item(
        self,
        item: Any,
        transform_id: str,
        magnitude: float,
        interpolation: InterpolationMode,
        fill: Optional[List[float]],
    ) -> Any:
        if transform_id == "Identity":
            return item
        elif transform_id == "ShearX":
            return self._dispatch(
                F.affine_image_tensor,
                F.affine_image_pil,
                item,
                angle=0.0,
                translate=[0, 0],
                scale=1.0,
                shear=[math.degrees(magnitude), 0.0],
                interpolation=interpolation,
                fill=fill,
            )
        elif transform_id == "ShearY":
            return self._dispatch(
                F.affine_image_tensor,
                F.affine_image_pil,
                item,
                angle=0.0,
                translate=[0, 0],
                scale=1.0,
                shear=[0.0, math.degrees(magnitude)],
                interpolation=interpolation,
                fill=fill,
            )
        elif transform_id == "TranslateX":
            return self._dispatch(
                F.affine_image_tensor,
                F.affine_image_pil,
                item,
                angle=0.0,
                translate=[int(magnitude), 0],
                scale=1.0,
                shear=[0.0, 0.0],
                interpolation=interpolation,
                fill=fill,
            )
        elif transform_id == "TranslateY":
            return self._dispatch(
                F.affine_image_tensor,
                F.affine_image_pil,
                item,
                angle=0.0,
                translate=[0, int(magnitude)],
                scale=1.0,
                shear=[0.0, 0.0],
                interpolation=interpolation,
                fill=fill,
            )
        elif transform_id == "Rotate":
            return self._dispatch(F.rotate_image_tensor, F.rotate_image_pil, item, angle=magnitude)
        elif transform_id == "Brightness":
            return self._dispatch(
                F.adjust_brightness_image_tensor,
                F.adjust_brightness_image_pil,
                item,
                brightness_factor=1.0 + magnitude,
            )
        elif transform_id == "Color":
            return self._dispatch(
                F.adjust_saturation_image_tensor,
                F.adjust_saturation_image_pil,
                item,
                saturation_factor=1.0 + magnitude,
            )
        elif transform_id == "Contrast":
            return self._dispatch(
                F.adjust_contrast_image_tensor, F.adjust_contrast_image_pil, item, contrast_factor=1.0 + magnitude
            )
        elif transform_id == "Sharpness":
            return self._dispatch(
                F.adjust_sharpness_image_tensor,
                F.adjust_sharpness_image_pil,
                item,
                sharpness_factor=1.0 + magnitude,
            )
        elif transform_id == "Posterize":
            return self._dispatch(F.posterize_image_tensor, F.posterize_image_pil, item, bits=int(magnitude))
        elif transform_id == "Solarize":
            return self._dispatch(F.solarize_image_tensor, F.solarize_image_pil, item, threshold=magnitude)
        elif transform_id == "AutoContrast":
            return self._dispatch(F.autocontrast_image_tensor, F.autocontrast_image_pil, item)
        elif transform_id == "Equalize":
            return self._dispatch(F.equalize_image_tensor, F.equalize_image_pil, item)
        elif transform_id == "Invert":
            return self._dispatch(F.invert_image_tensor, F.invert_image_pil, item)
        else:
            raise ValueError(f"No transform available for {transform_id}")

    def _apply_transform_to_sample(self, sample: Any, transform_id: str, magnitude: float) -> Any:
        return apply_recursively(
            functools.partial(
                self._apply_transform_to_item,
                transform_id=transform_id,
                magnitude=magnitude,
                interpolation=self.interpolation,
                fill=self._parse_fill(sample),
            ),
            sample,
        )


class AutoAugment(_AutoAugmentBase):
    _AUGMENTATION_SPACE = {
        "ShearX": (lambda num_bins, image_size: torch.linspace(0.0, 0.3, num_bins), True),
        "ShearY": (lambda num_bins, image_size: torch.linspace(0.0, 0.3, num_bins), True),
        "TranslateX": (lambda num_bins, image_size: torch.linspace(0.0, 150.0 / 331.0 * image_size[1], num_bins), True),
        "TranslateY": (lambda num_bins, image_size: torch.linspace(0.0, 150.0 / 331.0 * image_size[0], num_bins), True),
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

        image = self._query_image(sample)
        _, height, width = get_image_dimensions(image)

        policy = self._policies[int(torch.randint(len(self._policies), ()))]

        for transform_id, probability, magnitude_idx in policy:
            if not torch.rand(()) <= probability:
                continue

            magnitudes_fn, signed = self._AUGMENTATION_SPACE[transform_id]

            magnitudes = magnitudes_fn(10, (height, width))
            if magnitudes is not None:
                magnitude = float(magnitudes[magnitude_idx])
                if signed and torch.rand(()) <= 0.5:
                    magnitude *= -1
            else:
                magnitude = 0.0

            sample = self._apply_transform_to_sample(sample, transform_id, magnitude)

        return sample


class RandAugment(_AutoAugmentBase):
    _AUGMENTATION_SPACE = {
        "Identity": (lambda num_bins, image_size: None, False),
        "ShearX": (lambda num_bins, image_size: torch.linspace(0.0, 0.3, num_bins), True),
        "ShearY": (lambda num_bins, image_size: torch.linspace(0.0, 0.3, num_bins), True),
        "TranslateX": (lambda num_bins, image_size: torch.linspace(0.0, 150.0 / 331.0 * image_size[1], num_bins), True),
        "TranslateY": (lambda num_bins, image_size: torch.linspace(0.0, 150.0 / 331.0 * image_size[0], num_bins), True),
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

        image = self._query_image(sample)
        _, height, width = get_image_dimensions(image)

        for _ in range(self.num_ops):
            transform_id, (magnitudes_fn, signed) = self._get_random_item(self._AUGMENTATION_SPACE)

            magnitudes = magnitudes_fn(self.num_magnitude_bins, (height, width))
            if magnitudes is not None:
                magnitude = float(magnitudes[int(torch.randint(self.num_magnitude_bins, ()))])
                if signed and torch.rand(()) <= 0.5:
                    magnitude *= -1
            else:
                magnitude = 0.0

            sample = self._apply_transform_to_sample(sample, transform_id, magnitude)

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

        image = self._query_image(sample)
        _, height, width = get_image_dimensions(image)

        transform_id, (magnitudes_fn, signed) = self._get_random_item(self._AUGMENTATION_SPACE)

        magnitudes = magnitudes_fn(self.num_magnitude_bins, (height, width))
        if magnitudes is not None:
            magnitude = float(magnitudes[int(torch.randint(self.num_magnitude_bins, ()))])
            if signed and torch.rand(()) <= 0.5:
                magnitude *= -1
        else:
            magnitude = 0.0

        return self._apply_transform_to_sample(sample, transform_id, magnitude)


class AugMix(_AutoAugmentBase):
    _PARTIAL_AUGMENTATION_SPACE = {
        "ShearX": (lambda num_bins, image_size: torch.linspace(0.0, 0.3, num_bins), True),
        "ShearY": (lambda num_bins, image_size: torch.linspace(0.0, 0.3, num_bins), True),
        "TranslateX": (lambda num_bins, image_size: torch.linspace(0.0, image_size[1] / 3.0, num_bins), True),
        "TranslateY": (lambda num_bins, image_size: torch.linspace(0.0, image_size[0] / 3.0, num_bins), True),
        "Rotate": (lambda num_bins, image_size: torch.linspace(0.0, 30.0, num_bins), True),
        "Posterize": (
            lambda num_bins, image_size: cast(torch.Tensor, 4 - (torch.arange(num_bins) / ((num_bins - 1) / 4)))
            .round()
            .int(),
            False,
        ),
        "Solarize": (lambda num_bins, image_size: torch.linspace(255.0, 0.0, num_bins), False),
        "AutoContrast": (lambda num_bins, image_size: None, False),
        "Equalize": (lambda num_bins, image_size: None, False),
    }
    _AUGMENTATION_SPACE: Dict[str, Tuple[Callable[[int, Tuple[int, int]], Optional[torch.Tensor]], bool]] = {
        **_PARTIAL_AUGMENTATION_SPACE,
        "Brightness": (lambda num_bins, image_size: torch.linspace(0.0, 0.9, num_bins), True),
        "Color": (lambda num_bins, image_size: torch.linspace(0.0, 0.9, num_bins), True),
        "Contrast": (lambda num_bins, image_size: torch.linspace(0.0, 0.9, num_bins), True),
        "Sharpness": (lambda num_bins, image_size: torch.linspace(0.0, 0.9, num_bins), True),
    }

    def __init__(
        self,
        severity: int = 3,
        mixture_width: int = 3,
        chain_depth: int = -1,
        alpha: float = 1.0,
        all_ops: bool = True,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self._PARAMETER_MAX = 10
        if not (1 <= severity <= self._PARAMETER_MAX):
            raise ValueError(f"The severity must be between [1, {self._PARAMETER_MAX}]. Got {severity} instead.")
        self.severity = severity
        self.mixture_width = mixture_width
        self.chain_depth = chain_depth
        self.alpha = alpha
        self.all_ops = all_ops

    def _sample_dirichlet(self, params: torch.Tensor) -> torch.Tensor:
        # Must be on a separate method so that we can overwrite it in tests.
        return torch._sample_dirichlet(params)

    def _apply_augmix(self, input: Any) -> Any:
        if isinstance(input, (features.BoundingBox, features.SegmentationMask)):
            raise TypeError(f"{type(input).__name__}'s are not supported by {type(self).__name__}()")
        elif isinstance(input, torch.Tensor):
            image = input
        elif isinstance(input, PIL.Image.Image):
            image = pil_to_tensor(input)
        else:
            return input

        augmentation_space = self._AUGMENTATION_SPACE if self.all_ops else self._PARTIAL_AUGMENTATION_SPACE

        _, height, width = get_image_dimensions(image)
        fill = self._parse_fill(image)

        orig_dims = list(image.shape)
        batch = image.view([1] * max(4 - image.ndim, 0) + orig_dims)
        batch_dims = [batch.size(0)] + [1] * (batch.ndim - 1)

        # Sample the beta weights for combining the original and augmented image. To get Beta, we use a Dirichlet
        # with 2 parameters. The 1st column stores the weights of the original and the 2nd the ones of augmented image.
        m = self._sample_dirichlet(
            torch.tensor([self.alpha, self.alpha], device=batch.device).expand(batch_dims[0], -1)
        )

        # Sample the mixing weights and combine them with the ones sampled from Beta for the augmented images.
        combined_weights = self._sample_dirichlet(
            torch.tensor([self.alpha] * self.mixture_width, device=batch.device).expand(batch_dims[0], -1)
        ) * m[:, 1].view([batch_dims[0], -1])

        mix = m[:, 0].view(batch_dims) * batch
        for i in range(self.mixture_width):
            aug = batch
            depth = self.chain_depth if self.chain_depth > 0 else int(torch.randint(low=1, high=4, size=(1,)).item())
            for _ in range(depth):
                transform_id, (magnitudes_fn, signed) = self._get_random_item(augmentation_space)

                magnitudes = magnitudes_fn(self._PARAMETER_MAX, (height, width))
                if magnitudes is not None:
                    magnitude = float(magnitudes[int(torch.randint(self.severity, ()))])
                    if signed and torch.rand(()) <= 0.5:
                        magnitude *= -1
                else:
                    magnitude = 0.0

                aug = self._apply_transform_to_item(
                    image, transform_id, magnitude, interpolation=self.interpolation, fill=fill
                )
            mix.add_(combined_weights[:, i].view(batch_dims) * aug)
        mix = mix.view(orig_dims).to(dtype=image.dtype)

        if isinstance(input, features.Image):
            return features.Image.new_like(input, mix)
        elif isinstance(input, torch.Tensor):
            return mix
        else:  # isinstance(input, PIL.Image.Image):
            return to_pil_image(mix)

    def forward(self, *inputs: Any) -> Any:
        sample = inputs if len(inputs) > 1 else inputs[0]
        self._query_image(sample)
        return apply_recursively(self._apply_augmix, sample)
