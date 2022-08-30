import itertools

import pytest
import torch.testing
from test_prototype_transforms_functional import make_images
from torchvision import transforms as legacy_transforms
from torchvision.prototype import features, transforms as prototype_transforms
from torchvision.prototype.transforms.functional import to_image_pil, to_image_tensor

DEFAULT_MAKE_IMAGES_KWARGS = dict(color_spaces=[features.ColorSpace.RGB], extra_dims=[(4,)])


class ArgsKwargs:
    def __init__(self, *args, **kwargs):
        self.args = args
        self.kwargs = kwargs

    def __iter__(self):
        yield self.args
        yield self.kwargs

    def __str__(self):
        return ", ".join(
            itertools.chain(
                [repr(arg) for arg in self.args],
                [f"{param}={repr(kwarg)}" for param, kwarg in self.kwargs.items()],
            )
        )


class ConsistencyConfig:
    def __init__(
        self, prototype_cls, legacy_cls, transform_args_kwargs=None, make_images_kwargs=None, supports_pil=True
    ):
        self.prototype_cls = prototype_cls
        self.legacy_cls = legacy_cls
        self.transform_args_kwargs = transform_args_kwargs or [((), dict())]
        self.make_images_kwargs = make_images_kwargs or DEFAULT_MAKE_IMAGES_KWARGS
        self.supports_pil = supports_pil

    def parametrization(self):
        return [
            pytest.param(
                self.prototype_cls,
                self.legacy_cls,
                args_kwargs,
                self.make_images_kwargs,
                self.supports_pil,
                id=f"{self.prototype_cls.__name__}({args_kwargs})",
            )
            for args_kwargs in self.transform_args_kwargs
        ]


CONSISTENCY_CONFIGS = [
    ConsistencyConfig(
        prototype_transforms.Normalize,
        legacy_transforms.Normalize,
        [
            ArgsKwargs(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ],
        supports_pil=False,
        make_images_kwargs=dict(DEFAULT_MAKE_IMAGES_KWARGS, dtypes=[torch.float]),
    ),
    ConsistencyConfig(
        prototype_transforms.Resize,
        legacy_transforms.Resize,
        [
            ArgsKwargs(32),
            ArgsKwargs((32, 29)),
            ArgsKwargs((31, 28), interpolation=prototype_transforms.InterpolationMode.BICUBIC),
        ],
    ),
    ConsistencyConfig(
        prototype_transforms.CenterCrop,
        legacy_transforms.CenterCrop,
        [
            ArgsKwargs(18),
            ArgsKwargs((18, 13)),
        ],
    ),
]


@pytest.mark.parametrize(
    ("prototype_transform_cls", "legacy_transform_cls", "args_kwargs", "make_images_kwargs", "supports_pil"),
    itertools.chain.from_iterable(config.parametrization() for config in CONSISTENCY_CONFIGS),
)
def test_consistency(prototype_transform_cls, legacy_transform_cls, args_kwargs, make_images_kwargs, supports_pil):
    args, kwargs = args_kwargs

    try:
        legacy = legacy_transform_cls(*args, **kwargs)
    except Exception as exc:
        raise pytest.UsageError(
            f"Initializing the legacy transform failed with the error above. "
            f"Please correct the `ArgsKwargs({args_kwargs})` in the `ConsistencyConfig`."
        ) from exc

    try:
        prototype = prototype_transform_cls(*args, **kwargs)
    except Exception as exc:
        raise AssertionError(
            "Initializing the prototype transform failed with the error above. "
            "This means there is a consistency bug in the constructor."
        ) from exc

    for image in make_images(**make_images_kwargs):
        image_tensor = torch.Tensor(image)
        image_pil = to_image_pil(image) if image.ndim == 3 and supports_pil else None

        try:
            output_legacy_tensor = legacy(image_tensor)
        except Exception as exc:
            raise pytest.UsageError(
                f"Transforming a tensor image with shape {tuple(image.shape)} failed with the error above. "
                "This means that you need to specify the parameters passed to `make_images` through the "
                "`make_images_kwargs` of the `ConsistencyConfig`."
            ) from exc

        try:
            output_prototype_tensor = prototype(image_tensor)
        except Exception as exc:
            raise AssertionError(
                f"Transforming a tensor image with shape {tuple(image.shape)} failed with the error above. "
                f"This means there is a consistency bug either in `_get_params` "
                f"or in the `is_simple_tensor` path in `_transform`."
            ) from exc

        torch.testing.assert_close(
            output_prototype_tensor,
            output_legacy_tensor,
            atol=0,
            rtol=0,
            msg=lambda msg: f"Tensor image consistency check failed with: \n\n{msg}",
        )

        try:
            output_prototype_image = prototype(image)
        except Exception as exc:
            raise AssertionError(
                f"Transforming a feature image with shape {tuple(image.shape)} failed with the error above. "
                f"This means there is a consistency bug either in `_get_params` "
                f"or in the `features.Image` path in `_transform`."
            ) from exc

        torch.testing.assert_close(
            torch.Tensor(output_prototype_image),
            output_prototype_tensor,
            atol=0,
            rtol=0,
            msg=lambda msg: f"Output for feature and tensor images is not equal: \n\n{msg}",
        )

        if image_pil is not None:
            torch.testing.assert_close(
                to_image_tensor(prototype(image_pil)),
                to_image_tensor(legacy(image_pil)),
                atol=0,
                rtol=0,
                msg=lambda msg: f"PIL image consistency check failed with: \n\n{msg}",
            )
