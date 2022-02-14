import dataclasses
import math
from typing import Any, Dict, Tuple, Optional, Callable, List, cast, Iterator

import PIL.Image
import torch
from torchvision.prototype import features
from torchvision.prototype.transforms import Transform, InterpolationMode, AutoAugmentPolicy, kernels as K
from torchvision.prototype.utils._internal import apply_recursively
from torchvision.transforms import AutoAugment as _AutoAugment, functional as _F

from .utils import Query


@dataclasses.dataclass
class AutoAugmentDispatcher:
    kernel: Callable
    legacy_kernel: Callable
    magnitude_fn: Optional[Callable[[float], Dict[str, Any]]] = None
    extra_kwargs: Dict[str, Any] = dataclasses.field(default_factory=dict)
    takes_interpolation_kwargs: bool = False

    def __call__(
        self, input: Any, *, magnitude: float, interpolation: InterpolationMode, fill: Optional[List[float]]
    ) -> Any:
        kwargs = self.extra_kwargs.copy()
        if self.magnitude_fn is not None:
            kwargs.update(self.magnitude_fn(magnitude))
        if self.takes_interpolation_kwargs:
            kwargs.update(dict(interpolation=interpolation, fill=fill))

        kernel = self.kernel if type(input) is features.Image else self.legacy_kernel
        return kernel(input, **kwargs)


class _AutoAugmentBase(Transform):
    def __init__(
        self, *, interpolation: InterpolationMode = InterpolationMode.NEAREST, fill: Optional[List[float]] = None
    ) -> None:
        super().__init__()
        self.interpolation = interpolation
        self.fill = fill

    _DISPATCHER_MAP = {
        "ShearX": AutoAugmentDispatcher(
            K.affine_image,
            _F.affine,
            magnitude_fn=lambda magnitude: dict(shear=[math.degrees(magnitude), 0]),
            extra_kwargs=dict(angle=0.0, translate=[0, 0], scale=1.0),
            takes_interpolation_kwargs=True,
        ),
        "ShearY": AutoAugmentDispatcher(
            K.affine_image,
            _F.affine,
            magnitude_fn=lambda magnitude: dict(shear=[0, math.degrees(magnitude)]),
            extra_kwargs=dict(angle=0.0, translate=[0, 0], scale=1.0),
            takes_interpolation_kwargs=True,
        ),
        "TranslateX": AutoAugmentDispatcher(
            K.affine_image,
            _F.affine,
            magnitude_fn=lambda magnitude: dict(translate=[int(magnitude), 0]),
            extra_kwargs=dict(angle=0.0, scale=1.0, shear=[0.0, 0.0]),
            takes_interpolation_kwargs=True,
        ),
        "TranslateY": AutoAugmentDispatcher(
            K.affine_image,
            _F.affine,
            magnitude_fn=lambda magnitude: dict(translate=[0, int(magnitude)]),
            extra_kwargs=dict(angle=0.0, scale=1.0, shear=[0.0, 0.0]),
            takes_interpolation_kwargs=True,
        ),
        "Rotate": AutoAugmentDispatcher(
            K.rotate_image, _F.rotate, magnitude_fn=lambda magnitude: dict(angle=magnitude)
        ),
        "Brightness": AutoAugmentDispatcher(
            K.adjust_brightness_image,
            _F.adjust_brightness,
            magnitude_fn=lambda magnitude: dict(brightness_factor=1.0 + magnitude),
        ),
        "Color": AutoAugmentDispatcher(
            K.adjust_saturation_image,
            _F.adjust_saturation,
            magnitude_fn=lambda magnitude: dict(saturation_factor=1.0 + magnitude),
        ),
        "Contrast": AutoAugmentDispatcher(
            K.adjust_contrast_image,
            _F.adjust_contrast,
            magnitude_fn=lambda magnitude: dict(contrast_factor=1.0 + magnitude),
        ),
        "Sharpness": AutoAugmentDispatcher(
            K.adjust_sharpness_image,
            _F.adjust_sharpness,
            magnitude_fn=lambda magnitude: dict(sharpness_factor=1.0 + magnitude),
        ),
        "Posterize": AutoAugmentDispatcher(
            K.posterize_image, _F.posterize, magnitude_fn=lambda magnitude: dict(bits=int(magnitude))
        ),
        "Solarize": AutoAugmentDispatcher(
            K.solarize_image, _F.solarize, magnitude_fn=lambda magnitude: dict(threshold=magnitude)
        ),
        "AutoContrast": AutoAugmentDispatcher(K.autocontrast_image, _F.autocontrast),
        "Equalize": AutoAugmentDispatcher(K.equalize_image, _F.equalize),
        "Invert": AutoAugmentDispatcher(K.invert_image, _F.invert),
    }

    def get_params(self, sample: Any) -> Dict[str, Any]:
        image = Query(sample).image_for_size_and_channels_extraction()

        fill = self.fill
        if isinstance(fill, (int, float)):
            fill = [float(fill)] * image.num_channels
        elif fill is not None:
            fill = [float(f) for f in fill]

        return dict(fill=fill)

    def get_transforms_meta(self, image_size: Tuple[int, int]) -> Iterator[Tuple[str, float]]:
        raise NotImplementedError

    def forward(self, *inputs: Any, params: Optional[Dict[str, Any]] = None) -> Any:
        sample = inputs if len(inputs) > 1 else inputs[0]
        params = params or self.get_params(sample)

        for transform_id, magnitude in self.get_transforms_meta(Query(sample).image_size()):
            dispatcher = self._DISPATCHER_MAP[transform_id]

            def transform(input: Any) -> Any:
                if type(input) in {torch.Tensor, features.Image} or isinstance(input, PIL.Image.Image):
                    return dispatcher(  # type: ignore[arg-type]
                        input,
                        magnitude=magnitude,
                        interpolation=self.interpolation,
                        **params,
                    )
                else:
                    return input

            sample = apply_recursively(transform, sample)

        return sample

    def _randbool(self, p: float = 0.5) -> bool:
        """Randomly returns either ``True`` or ``False``.

        Args:
            p: Probability to return ``True``. Defaults to ``0.5``.
        """
        return float(torch.rand(())) <= p


@dataclasses.dataclass
class AugmentationMeta:
    dispatcher_id: str
    magnitudes_fn: Callable[[int, Tuple[int, int]], Optional[torch.Tensor]]
    signed: bool


class AutoAugment(_AutoAugmentBase):
    _LEGACY_CLS = _AutoAugment
    _AUGMENTATION_SPACE = (
        AugmentationMeta("ShearX", lambda num_bins, image_size: torch.linspace(0.0, 0.3, num_bins), True),
        AugmentationMeta("ShearY", lambda num_bins, image_size: torch.linspace(0.0, 0.3, num_bins), True),
        AugmentationMeta(
            "TranslateX",
            lambda num_bins, image_size: torch.linspace(0.0, 150.0 / 331.0 * image_size[1], num_bins),
            True,
        ),
        AugmentationMeta(
            "TranslateY",
            lambda num_bins, image_size: torch.linspace(0.0, 150.0 / 331.0 * image_size[0], num_bins),
            True,
        ),
        AugmentationMeta("Rotate", lambda num_bins, image_size: torch.linspace(0.0, 30.0, num_bins), True),
        AugmentationMeta("Brightness", lambda num_bins, image_size: torch.linspace(0.0, 0.9, num_bins), True),
        AugmentationMeta("Color", lambda num_bins, image_size: torch.linspace(0.0, 0.9, num_bins), True),
        AugmentationMeta("Contrast", lambda num_bins, image_size: torch.linspace(0.0, 0.9, num_bins), True),
        AugmentationMeta("Sharpness", lambda num_bins, image_size: torch.linspace(0.0, 0.9, num_bins), True),
        AugmentationMeta(
            "Posterize",
            lambda num_bins, image_size: cast(torch.Tensor, 8 - (torch.arange(num_bins) / ((num_bins - 1) / 4)))
            .round()
            .int(),
            False,
        ),
        AugmentationMeta("Solarize", lambda num_bins, image_size: torch.linspace(255.0, 0.0, num_bins), False),
        AugmentationMeta("AutoContrast", lambda num_bins, image_size: None, False),
        AugmentationMeta("Equalize", lambda num_bins, image_size: None, False),
        AugmentationMeta("Invert", lambda num_bins, image_size: None, False),
    )
    _AUGMENTATION_SPACE = {
        augmentation_meta.dispatcher_id: augmentation_meta for augmentation_meta in _AUGMENTATION_SPACE
    }

    def __init__(self, policy: AutoAugmentPolicy = AutoAugmentPolicy.IMAGENET, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.policy = policy
        self._policies = self._LEGACY_CLS._get_policies(None, policy)  # type: ignore[arg-type]

    def get_transforms_meta(self, image_size: Tuple[int, int]) -> Iterator[Tuple[str, float]]:
        policy = self._policies[int(torch.randint(len(self._policies), ()))]

        for dispatcher_id, probability, magnitude_idx in policy:
            if not self._randbool(probability):
                continue

            augmentation_meta = self._AUGMENTATION_SPACE[dispatcher_id]

            magnitudes = augmentation_meta.magnitudes_fn(10, image_size)
            if magnitudes is not None:
                magnitude = float(magnitudes[magnitude_idx])
                if augmentation_meta.signed and self._randbool():
                    magnitude *= -1
            else:
                magnitude = 0.0

            yield augmentation_meta.dispatcher_id, magnitude


class RandAugment(_AutoAugmentBase):
    _AUGMENTATION_SPACE = (
        AugmentationMeta("Identity", lambda num_bins, image_size: None, False),
        AugmentationMeta("ShearX", lambda num_bins, image_size: torch.linspace(0.0, 0.3, num_bins), True),
        AugmentationMeta("ShearY", lambda num_bins, image_size: torch.linspace(0.0, 0.3, num_bins), True),
        AugmentationMeta(
            "TranslateX",
            lambda num_bins, image_size: torch.linspace(0.0, 150.0 / 331.0 * image_size[1], num_bins),
            True,
        ),
        AugmentationMeta(
            "TranslateY",
            lambda num_bins, image_size: torch.linspace(0.0, 150.0 / 331.0 * image_size[0], num_bins),
            True,
        ),
        AugmentationMeta("Rotate", lambda num_bins, image_size: torch.linspace(0.0, 30.0, num_bins), True),
        AugmentationMeta("Brightness", lambda num_bins, image_size: torch.linspace(0.0, 0.9, num_bins), True),
        AugmentationMeta("Color", lambda num_bins, image_size: torch.linspace(0.0, 0.9, num_bins), True),
        AugmentationMeta("Contrast", lambda num_bins, image_size: torch.linspace(0.0, 0.9, num_bins), True),
        AugmentationMeta("Sharpness", lambda num_bins, image_size: torch.linspace(0.0, 0.9, num_bins), True),
        AugmentationMeta(
            "Posterize",
            lambda num_bins, image_size: cast(torch.Tensor, 8 - (torch.arange(num_bins) / ((num_bins - 1) / 4)))
            .round()
            .int(),
            False,
        ),
        AugmentationMeta("Solarize", lambda num_bins, image_size: torch.linspace(255.0, 0.0, num_bins), False),
        AugmentationMeta("AutoContrast", lambda num_bins, image_size: None, False),
        AugmentationMeta("Equalize", lambda num_bins, image_size: None, False),
    )

    def __init__(self, *, num_ops: int = 2, magnitude: int = 9, num_magnitude_bins: int = 31, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.num_ops = num_ops
        self.magnitude = magnitude
        self.num_magnitude_bins = num_magnitude_bins

    def get_transforms_meta(self, image_size: Tuple[int, int]) -> Iterator[Tuple[str, float]]:
        for _ in range(self.num_ops):
            augmentation_meta = self._AUGMENTATION_SPACE[int(torch.randint(len(self._AUGMENTATION_SPACE), ()))]
            if augmentation_meta.dispatcher_id == "Identity":
                continue

            magnitudes = augmentation_meta.magnitudes_fn(self.num_magnitude_bins, image_size)
            if magnitudes is not None:
                magnitude = float(magnitudes[int(torch.randint(self.num_magnitude_bins, ()))])
                if augmentation_meta.signed and self._randbool():
                    magnitude *= -1
            else:
                magnitude = 0.0

            yield augmentation_meta.dispatcher_id, magnitude


class TrivialAugmentWide(_AutoAugmentBase):
    _AUGMENTATION_SPACE = (
        AugmentationMeta("Identity", lambda num_bins, image_size: None, False),
        AugmentationMeta("ShearX", lambda num_bins, image_size: torch.linspace(0.0, 0.99, num_bins), True),
        AugmentationMeta("ShearY", lambda num_bins, image_size: torch.linspace(0.0, 0.99, num_bins), True),
        AugmentationMeta("TranslateX", lambda num_bins, image_size: torch.linspace(0.0, 32.0, num_bins), True),
        AugmentationMeta("TranslateY", lambda num_bins, image_size: torch.linspace(0.0, 32.0, num_bins), True),
        AugmentationMeta("Rotate", lambda num_bins, image_size: torch.linspace(0.0, 135.0, num_bins), True),
        AugmentationMeta("Brightness", lambda num_bins, image_size: torch.linspace(0.0, 0.99, num_bins), True),
        AugmentationMeta("Color", lambda num_bins, image_size: torch.linspace(0.0, 0.99, num_bins), True),
        AugmentationMeta("Contrast", lambda num_bins, image_size: torch.linspace(0.0, 0.99, num_bins), True),
        AugmentationMeta("Sharpness", lambda num_bins, image_size: torch.linspace(0.0, 0.99, num_bins), True),
        AugmentationMeta(
            "Posterize",
            lambda num_bins, image_size: cast(torch.Tensor, 8 - (torch.arange(num_bins) / ((num_bins - 1) / 6)))
            .round()
            .int(),
            False,
        ),
        AugmentationMeta("Solarize", lambda num_bins, image_size: torch.linspace(255.0, 0.0, num_bins), False),
        AugmentationMeta("AutoContrast", lambda num_bins, image_size: None, False),
        AugmentationMeta("Equalize", lambda num_bins, image_size: None, False),
    )

    def __init__(self, *, num_magnitude_bins: int = 31, **kwargs: Any):
        super().__init__(**kwargs)
        self.num_magnitude_bins = num_magnitude_bins

    def get_transforms_meta(self, image_size: Tuple[int, int]) -> Iterator[Tuple[str, float]]:
        augmentation_meta = self._AUGMENTATION_SPACE[int(torch.randint(len(self._AUGMENTATION_SPACE), ()))]

        if augmentation_meta.dispatcher_id == "Identity":
            return

        magnitudes = augmentation_meta.magnitudes_fn(self.num_magnitude_bins, image_size)
        if magnitudes is not None:
            magnitude = float(magnitudes[int(torch.randint(self.num_magnitude_bins, ()))])
            if augmentation_meta.signed and self._randbool():
                magnitude *= -1
        else:
            magnitude = 0.0

        yield augmentation_meta.dispatcher_id, magnitude
