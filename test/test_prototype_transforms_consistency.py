import enum
import inspect

import numpy as np
import PIL.Image
import pytest

import torch
from prototype_common_utils import ArgsKwargs, assert_equal, make_images
from torchvision import transforms as legacy_transforms
from torchvision._utils import sequence_to_str
from torchvision.prototype import features, transforms as prototype_transforms
from torchvision.prototype.transforms.functional import to_image_pil


DEFAULT_MAKE_IMAGES_KWARGS = dict(color_spaces=[features.ColorSpace.RGB], extra_dims=[(4,)])


class ConsistencyConfig:
    def __init__(
        self,
        prototype_cls,
        legacy_cls,
        # If no args_kwargs is passed, only the signature will be checked
        args_kwargs=(),
        make_images_kwargs=None,
        supports_pil=True,
        removed_params=(),
    ):
        self.prototype_cls = prototype_cls
        self.legacy_cls = legacy_cls
        self.args_kwargs = args_kwargs
        self.make_images_kwargs = make_images_kwargs or DEFAULT_MAKE_IMAGES_KWARGS
        self.supports_pil = supports_pil
        self.removed_params = removed_params


# These are here since both the prototype and legacy transform need to be constructed with the same random parameters
LINEAR_TRANSFORMATION_MEAN = torch.rand(36)
LINEAR_TRANSFORMATION_MATRIX = torch.rand([LINEAR_TRANSFORMATION_MEAN.numel()] * 2)

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
            ArgsKwargs((31, 28), interpolation=prototype_transforms.InterpolationMode.NEAREST),
            ArgsKwargs((33, 26), interpolation=prototype_transforms.InterpolationMode.BICUBIC),
            # FIXME: these are currently failing, since the new transform only supports the enum. The int input is
            #  already deprecated and scheduled to be removed in 0.15. Should we support ints on the prototype
            #  transform? I guess it depends if we roll out before 0.15 or not.
            # ArgsKwargs((30, 27), interpolation=0),
            # ArgsKwargs((35, 29), interpolation=2),
            # ArgsKwargs((34, 25), interpolation=3),
            ArgsKwargs(31, max_size=32),
            ArgsKwargs(30, max_size=100),
            ArgsKwargs((29, 32), antialias=False),
            ArgsKwargs((28, 31), antialias=True),
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
    ConsistencyConfig(
        prototype_transforms.FiveCrop,
        legacy_transforms.FiveCrop,
        [
            ArgsKwargs(18),
            ArgsKwargs((18, 13)),
        ],
        make_images_kwargs=dict(DEFAULT_MAKE_IMAGES_KWARGS, sizes=[(20, 19)]),
    ),
    ConsistencyConfig(
        prototype_transforms.TenCrop,
        legacy_transforms.TenCrop,
        [
            ArgsKwargs(18),
            ArgsKwargs((18, 13)),
            ArgsKwargs(18, vertical_flip=True),
        ],
        make_images_kwargs=dict(DEFAULT_MAKE_IMAGES_KWARGS, sizes=[(20, 19)]),
    ),
    ConsistencyConfig(
        prototype_transforms.Pad,
        legacy_transforms.Pad,
        [
            ArgsKwargs(3),
            ArgsKwargs([3]),
            ArgsKwargs([2, 3]),
            ArgsKwargs([3, 2, 1, 4]),
            ArgsKwargs(5, fill=1, padding_mode="constant"),
            ArgsKwargs(5, padding_mode="edge"),
            ArgsKwargs(5, padding_mode="reflect"),
            ArgsKwargs(5, padding_mode="symmetric"),
        ],
    ),
    ConsistencyConfig(
        prototype_transforms.LinearTransformation,
        legacy_transforms.LinearTransformation,
        [
            ArgsKwargs(LINEAR_TRANSFORMATION_MATRIX, LINEAR_TRANSFORMATION_MEAN),
        ],
        # Make sure that the product of the height, width and number of channels matches the number of elements in
        # `LINEAR_TRANSFORMATION_MEAN`. For example 2 * 6 * 3 == 4 * 3 * 3 == 36.
        make_images_kwargs=dict(
            DEFAULT_MAKE_IMAGES_KWARGS, sizes=[(2, 6), (4, 3)], color_spaces=[features.ColorSpace.RGB]
        ),
        supports_pil=False,
    ),
    ConsistencyConfig(
        prototype_transforms.Grayscale,
        legacy_transforms.Grayscale,
        [
            ArgsKwargs(num_output_channels=1),
            ArgsKwargs(num_output_channels=3),
        ],
        make_images_kwargs=dict(
            DEFAULT_MAKE_IMAGES_KWARGS, color_spaces=[features.ColorSpace.RGB, features.ColorSpace.GRAY]
        ),
    ),
    ConsistencyConfig(
        prototype_transforms.ConvertImageDtype,
        legacy_transforms.ConvertImageDtype,
        [
            ArgsKwargs(torch.float16),
            ArgsKwargs(torch.bfloat16),
            ArgsKwargs(torch.float32),
            ArgsKwargs(torch.float64),
            ArgsKwargs(torch.uint8),
        ],
        supports_pil=False,
    ),
    ConsistencyConfig(
        prototype_transforms.ToPILImage,
        legacy_transforms.ToPILImage,
        [ArgsKwargs()],
        make_images_kwargs=dict(
            color_spaces=[
                features.ColorSpace.GRAY,
                features.ColorSpace.GRAY_ALPHA,
                features.ColorSpace.RGB,
                features.ColorSpace.RGB_ALPHA,
            ],
            extra_dims=[()],
        ),
        supports_pil=False,
    ),
    ConsistencyConfig(
        prototype_transforms.Lambda,
        legacy_transforms.Lambda,
        [
            ArgsKwargs(lambda image: image / 2),
        ],
        # Technically, this also supports PIL, but it is overkill to write a function here that supports tensor and PIL
        # images given that the transform does nothing but call it anyway.
        supports_pil=False,
    ),
    ConsistencyConfig(
        prototype_transforms.RandomHorizontalFlip,
        legacy_transforms.RandomHorizontalFlip,
        [
            ArgsKwargs(p=0),
            ArgsKwargs(p=1),
        ],
    ),
    ConsistencyConfig(
        prototype_transforms.RandomVerticalFlip,
        legacy_transforms.RandomVerticalFlip,
        [
            ArgsKwargs(p=0),
            ArgsKwargs(p=1),
        ],
    ),
    ConsistencyConfig(
        prototype_transforms.RandomEqualize,
        legacy_transforms.RandomEqualize,
        [
            ArgsKwargs(p=0),
            ArgsKwargs(p=1),
        ],
        make_images_kwargs=dict(DEFAULT_MAKE_IMAGES_KWARGS, dtypes=[torch.uint8]),
    ),
    ConsistencyConfig(
        prototype_transforms.RandomInvert,
        legacy_transforms.RandomInvert,
        [
            ArgsKwargs(p=0),
            ArgsKwargs(p=1),
        ],
    ),
    ConsistencyConfig(
        prototype_transforms.RandomPosterize,
        legacy_transforms.RandomPosterize,
        [
            ArgsKwargs(p=0, bits=5),
            ArgsKwargs(p=1, bits=1),
            ArgsKwargs(p=1, bits=3),
        ],
        make_images_kwargs=dict(DEFAULT_MAKE_IMAGES_KWARGS, dtypes=[torch.uint8]),
    ),
    ConsistencyConfig(
        prototype_transforms.RandomSolarize,
        legacy_transforms.RandomSolarize,
        [
            ArgsKwargs(p=0, threshold=0.5),
            ArgsKwargs(p=1, threshold=0.3),
            ArgsKwargs(p=1, threshold=0.99),
        ],
    ),
    ConsistencyConfig(
        prototype_transforms.RandomAutocontrast,
        legacy_transforms.RandomAutocontrast,
        [
            ArgsKwargs(p=0),
            ArgsKwargs(p=1),
        ],
    ),
    ConsistencyConfig(
        prototype_transforms.RandomAdjustSharpness,
        legacy_transforms.RandomAdjustSharpness,
        [
            ArgsKwargs(p=0, sharpness_factor=0.5),
            ArgsKwargs(p=1, sharpness_factor=0.3),
            ArgsKwargs(p=1, sharpness_factor=0.99),
        ],
    ),
    ConsistencyConfig(
        prototype_transforms.RandomGrayscale,
        legacy_transforms.RandomGrayscale,
        [
            ArgsKwargs(p=0),
            ArgsKwargs(p=1),
        ],
    ),
    ConsistencyConfig(
        prototype_transforms.RandomResizedCrop,
        legacy_transforms.RandomResizedCrop,
        [
            ArgsKwargs(16),
            ArgsKwargs(17, scale=(0.3, 0.7)),
            ArgsKwargs(25, ratio=(0.5, 1.5)),
            ArgsKwargs((31, 28), interpolation=prototype_transforms.InterpolationMode.NEAREST),
            ArgsKwargs((33, 26), interpolation=prototype_transforms.InterpolationMode.BICUBIC),
            ArgsKwargs((29, 32), antialias=False),
            ArgsKwargs((28, 31), antialias=True),
        ],
    ),
    ConsistencyConfig(
        prototype_transforms.RandomErasing,
        legacy_transforms.RandomErasing,
        [
            ArgsKwargs(p=0),
            ArgsKwargs(p=1),
            ArgsKwargs(p=1, scale=(0.3, 0.7)),
            ArgsKwargs(p=1, ratio=(0.5, 1.5)),
            ArgsKwargs(p=1, value=1),
            ArgsKwargs(p=1, value=(1, 2, 3)),
            ArgsKwargs(p=1, value="random"),
        ],
        supports_pil=False,
    ),
    ConsistencyConfig(
        prototype_transforms.ColorJitter,
        legacy_transforms.ColorJitter,
        [
            ArgsKwargs(),
            ArgsKwargs(brightness=0.1),
            ArgsKwargs(brightness=(0.2, 0.3)),
            ArgsKwargs(contrast=0.4),
            ArgsKwargs(contrast=(0.5, 0.6)),
            ArgsKwargs(saturation=0.7),
            ArgsKwargs(saturation=(0.8, 0.9)),
            ArgsKwargs(hue=0.3),
            ArgsKwargs(hue=(-0.1, 0.2)),
            ArgsKwargs(brightness=0.1, contrast=0.4, saturation=0.7, hue=0.3),
        ],
    ),
    ConsistencyConfig(
        prototype_transforms.ElasticTransform,
        legacy_transforms.ElasticTransform,
        [
            ArgsKwargs(),
            ArgsKwargs(alpha=20.0),
            ArgsKwargs(alpha=(15.3, 27.2)),
            ArgsKwargs(sigma=3.0),
            ArgsKwargs(sigma=(2.5, 3.9)),
            ArgsKwargs(interpolation=prototype_transforms.InterpolationMode.NEAREST),
            ArgsKwargs(interpolation=prototype_transforms.InterpolationMode.BICUBIC),
            ArgsKwargs(fill=1),
        ],
        # ElasticTransform needs larger images to avoid the needed internal padding being larger than the actual image
        make_images_kwargs=dict(DEFAULT_MAKE_IMAGES_KWARGS, sizes=[(163, 163), (72, 333), (313, 95)]),
    ),
    ConsistencyConfig(
        prototype_transforms.GaussianBlur,
        legacy_transforms.GaussianBlur,
        [
            ArgsKwargs(kernel_size=3),
            ArgsKwargs(kernel_size=(1, 5)),
            ArgsKwargs(kernel_size=3, sigma=0.7),
            ArgsKwargs(kernel_size=5, sigma=(0.3, 1.4)),
        ],
    ),
    ConsistencyConfig(
        prototype_transforms.RandomAffine,
        legacy_transforms.RandomAffine,
        [
            ArgsKwargs(degrees=30.0),
            ArgsKwargs(degrees=(-20.0, 10.0)),
            ArgsKwargs(degrees=0.0, translate=(0.4, 0.6)),
            ArgsKwargs(degrees=0.0, scale=(0.3, 0.8)),
            ArgsKwargs(degrees=0.0, shear=13),
            ArgsKwargs(degrees=0.0, shear=(8, 17)),
            ArgsKwargs(degrees=0.0, shear=(4, 5, 4, 13)),
            ArgsKwargs(degrees=(-20.0, 10.0), translate=(0.4, 0.6), scale=(0.3, 0.8), shear=(4, 5, 4, 13)),
            ArgsKwargs(degrees=30.0, interpolation=prototype_transforms.InterpolationMode.NEAREST),
            ArgsKwargs(degrees=30.0, fill=1),
            ArgsKwargs(degrees=30.0, fill=(2, 3, 4)),
            ArgsKwargs(degrees=30.0, center=(0, 0)),
        ],
        removed_params=["fillcolor", "resample"],
    ),
    ConsistencyConfig(
        prototype_transforms.RandomCrop,
        legacy_transforms.RandomCrop,
        [
            ArgsKwargs(12),
            ArgsKwargs((15, 17)),
            ArgsKwargs(11, padding=1),
            ArgsKwargs((8, 13), padding=(2, 3)),
            ArgsKwargs((14, 9), padding=(0, 2, 1, 0)),
            ArgsKwargs(36, pad_if_needed=True),
            ArgsKwargs((7, 8), fill=1),
            ArgsKwargs(5, fill=(1, 2, 3)),
            ArgsKwargs(12),
            ArgsKwargs(15, padding=2, padding_mode="edge"),
            ArgsKwargs(17, padding=(1, 0), padding_mode="reflect"),
            ArgsKwargs(8, padding=(3, 0, 0, 1), padding_mode="symmetric"),
        ],
        make_images_kwargs=dict(DEFAULT_MAKE_IMAGES_KWARGS, sizes=[(26, 26), (18, 33), (29, 22)]),
    ),
    ConsistencyConfig(
        prototype_transforms.RandomPerspective,
        legacy_transforms.RandomPerspective,
        [
            ArgsKwargs(p=0),
            ArgsKwargs(p=1),
            ArgsKwargs(p=1, distortion_scale=0.3),
            ArgsKwargs(p=1, distortion_scale=0.2, interpolation=prototype_transforms.InterpolationMode.NEAREST),
            ArgsKwargs(p=1, distortion_scale=0.1, fill=1),
            ArgsKwargs(p=1, distortion_scale=0.4, fill=(1, 2, 3)),
        ],
    ),
    ConsistencyConfig(
        prototype_transforms.RandomRotation,
        legacy_transforms.RandomRotation,
        [
            ArgsKwargs(degrees=30.0),
            ArgsKwargs(degrees=(-20.0, 10.0)),
            ArgsKwargs(degrees=30.0, interpolation=prototype_transforms.InterpolationMode.BILINEAR),
            ArgsKwargs(degrees=30.0, expand=True),
            ArgsKwargs(degrees=30.0, center=(0, 0)),
            ArgsKwargs(degrees=30.0, fill=1),
            ArgsKwargs(degrees=30.0, fill=(1, 2, 3)),
        ],
        removed_params=["resample"],
    ),
    ConsistencyConfig(
        prototype_transforms.PILToTensor,
        legacy_transforms.PILToTensor,
    ),
    ConsistencyConfig(
        prototype_transforms.ToTensor,
        legacy_transforms.ToTensor,
    ),
    ConsistencyConfig(
        prototype_transforms.Compose,
        legacy_transforms.Compose,
    ),
    ConsistencyConfig(
        prototype_transforms.RandomApply,
        legacy_transforms.RandomApply,
    ),
    ConsistencyConfig(
        prototype_transforms.RandomChoice,
        legacy_transforms.RandomChoice,
    ),
    ConsistencyConfig(
        prototype_transforms.RandomOrder,
        legacy_transforms.RandomOrder,
    ),
    ConsistencyConfig(
        prototype_transforms.AugMix,
        legacy_transforms.AugMix,
    ),
    ConsistencyConfig(
        prototype_transforms.AutoAugment,
        legacy_transforms.AutoAugment,
    ),
    ConsistencyConfig(
        prototype_transforms.RandAugment,
        legacy_transforms.RandAugment,
    ),
    ConsistencyConfig(
        prototype_transforms.TrivialAugmentWide,
        legacy_transforms.TrivialAugmentWide,
    ),
]


def test_automatic_coverage():
    available = {
        name
        for name, obj in legacy_transforms.__dict__.items()
        if not name.startswith("_") and isinstance(obj, type) and not issubclass(obj, enum.Enum)
    }

    checked = {config.legacy_cls.__name__ for config in CONSISTENCY_CONFIGS}

    missing = available - checked
    if missing:
        raise AssertionError(
            f"The prototype transformations {sequence_to_str(sorted(missing), separate_last='and ')} "
            f"are not checked for consistency although a legacy counterpart exists."
        )


@pytest.mark.parametrize("config", CONSISTENCY_CONFIGS, ids=lambda config: config.legacy_cls.__name__)
def test_signature_consistency(config):
    legacy_params = dict(inspect.signature(config.legacy_cls).parameters)
    prototype_params = dict(inspect.signature(config.prototype_cls).parameters)

    for param in config.removed_params:
        legacy_params.pop(param, None)

    missing = legacy_params.keys() - prototype_params.keys()
    if missing:
        raise AssertionError(
            f"The prototype transform does not support the parameters "
            f"{sequence_to_str(sorted(missing), separate_last='and ')}, but the legacy transform does. "
            f"If that is intentional, e.g. pending deprecation, please add the parameters to the `removed_params` on "
            f"the `ConsistencyConfig`."
        )

    extra = prototype_params.keys() - legacy_params.keys()
    extra_without_default = {
        param
        for param in extra
        if prototype_params[param].default is inspect.Parameter.empty
        and prototype_params[param].kind not in {inspect.Parameter.VAR_POSITIONAL, inspect.Parameter.VAR_KEYWORD}
    }
    if extra_without_default:
        raise AssertionError(
            f"The prototype transform requires the parameters "
            f"{sequence_to_str(sorted(extra_without_default), separate_last='and ')}, but the legacy transform does "
            f"not. Please add a default value."
        )

    legacy_kinds = {name: param.kind for name, param in legacy_params.items()}
    prototype_kinds = {name: prototype_params[name].kind for name in legacy_kinds.keys()}
    assert prototype_kinds == legacy_kinds


def check_call_consistency(prototype_transform, legacy_transform, images=None, supports_pil=True):
    if images is None:
        images = make_images(**DEFAULT_MAKE_IMAGES_KWARGS)

    for image in images:
        image_repr = f"[{tuple(image.shape)}, {str(image.dtype).rsplit('.')[-1]}]"

        image_tensor = torch.Tensor(image)

        try:
            torch.manual_seed(0)
            output_legacy_tensor = legacy_transform(image_tensor)
        except Exception as exc:
            raise pytest.UsageError(
                f"Transforming a tensor image {image_repr} failed in the legacy transform with the "
                f"error above. This means that you need to specify the parameters passed to `make_images` through the "
                "`make_images_kwargs` of the `ConsistencyConfig`."
            ) from exc

        try:
            torch.manual_seed(0)
            output_prototype_tensor = prototype_transform(image_tensor)
        except Exception as exc:
            raise AssertionError(
                f"Transforming a tensor image with shape {image_repr} failed in the prototype transform with "
                f"the error above. This means there is a consistency bug either in `_get_params` or in the "
                f"`is_simple_tensor` path in `_transform`."
            ) from exc

        assert_equal(
            output_prototype_tensor,
            output_legacy_tensor,
            msg=lambda msg: f"Tensor image consistency check failed with: \n\n{msg}",
        )

        try:
            torch.manual_seed(0)
            output_prototype_image = prototype_transform(image)
        except Exception as exc:
            raise AssertionError(
                f"Transforming a feature image with shape {image_repr} failed in the prototype transform with "
                f"the error above. This means there is a consistency bug either in `_get_params` or in the "
                f"`features.Image` path in `_transform`."
            ) from exc

        assert_equal(
            output_prototype_image,
            output_prototype_tensor,
            msg=lambda msg: f"Output for feature and tensor images is not equal: \n\n{msg}",
        )

        if image.ndim == 3 and supports_pil:
            image_pil = to_image_pil(image)

            try:
                torch.manual_seed(0)
                output_legacy_pil = legacy_transform(image_pil)
            except Exception as exc:
                raise pytest.UsageError(
                    f"Transforming a PIL image with shape {image_repr} failed in the legacy transform with the "
                    f"error above. If this transform does not support PIL images, set `supports_pil=False` on the "
                    "`ConsistencyConfig`. "
                ) from exc

            try:
                torch.manual_seed(0)
                output_prototype_pil = prototype_transform(image_pil)
            except Exception as exc:
                raise AssertionError(
                    f"Transforming a PIL image with shape {image_repr} failed in the prototype transform with "
                    f"the error above. This means there is a consistency bug either in `_get_params` or in the "
                    f"`PIL.Image.Image` path in `_transform`."
                ) from exc

            assert_equal(
                output_prototype_pil,
                output_legacy_pil,
                msg=lambda msg: f"PIL image consistency check failed with: \n\n{msg}",
            )


@pytest.mark.parametrize(
    ("config", "args_kwargs"),
    [
        pytest.param(config, args_kwargs, id=f"{config.legacy_cls.__name__}({args_kwargs})")
        for config in CONSISTENCY_CONFIGS
        for args_kwargs in config.args_kwargs
    ],
)
def test_call_consistency(config, args_kwargs):
    args, kwargs = args_kwargs

    try:
        legacy_transform = config.legacy_cls(*args, **kwargs)
    except Exception as exc:
        raise pytest.UsageError(
            f"Initializing the legacy transform failed with the error above. "
            f"Please correct the `ArgsKwargs({args_kwargs})` in the `ConsistencyConfig`."
        ) from exc

    try:
        prototype_transform = config.prototype_cls(*args, **kwargs)
    except Exception as exc:
        raise AssertionError(
            "Initializing the prototype transform failed with the error above. "
            "This means there is a consistency bug in the constructor."
        ) from exc

    check_call_consistency(
        prototype_transform,
        legacy_transform,
        images=make_images(**config.make_images_kwargs),
        supports_pil=config.supports_pil,
    )


class TestContainerTransforms:
    """
    Since we are testing containers here, we also need some transforms to wrap. Thus, testing a container transform for
    consistency automatically tests the wrapped transforms consistency.

    Instead of complicated mocking or creating custom transforms just for these tests, here we use deterministic ones
    that were already tested for consistency above.
    """

    def test_compose(self):
        prototype_transform = prototype_transforms.Compose(
            [
                prototype_transforms.Resize(256),
                prototype_transforms.CenterCrop(224),
            ]
        )
        legacy_transform = legacy_transforms.Compose(
            [
                legacy_transforms.Resize(256),
                legacy_transforms.CenterCrop(224),
            ]
        )

        check_call_consistency(prototype_transform, legacy_transform)

    @pytest.mark.parametrize("p", [0, 0.1, 0.5, 0.9, 1])
    def test_random_apply(self, p):
        prototype_transform = prototype_transforms.RandomApply(
            [
                prototype_transforms.Resize(256),
                legacy_transforms.CenterCrop(224),
            ],
            p=p,
        )
        legacy_transform = legacy_transforms.RandomApply(
            [
                legacy_transforms.Resize(256),
                legacy_transforms.CenterCrop(224),
            ],
            p=p,
        )

        check_call_consistency(prototype_transform, legacy_transform)

    # We can't test other values for `p` since the random parameter generation is different
    @pytest.mark.parametrize("p", [(0, 1), (1, 0)])
    def test_random_choice(self, p):
        prototype_transform = prototype_transforms.RandomChoice(
            [
                prototype_transforms.Resize(256),
                legacy_transforms.CenterCrop(224),
            ],
            p=p,
        )
        legacy_transform = legacy_transforms.RandomChoice(
            [
                legacy_transforms.Resize(256),
                legacy_transforms.CenterCrop(224),
            ],
            p=p,
        )

        check_call_consistency(prototype_transform, legacy_transform)


class TestToTensorTransforms:
    def test_pil_to_tensor(self):
        prototype_transform = prototype_transforms.PILToTensor()
        legacy_transform = legacy_transforms.PILToTensor()

        for image in make_images(extra_dims=[()]):
            image_pil = to_image_pil(image)

            assert_equal(prototype_transform(image_pil), legacy_transform(image_pil))

    def test_to_tensor(self):
        prototype_transform = prototype_transforms.ToTensor()
        legacy_transform = legacy_transforms.ToTensor()

        for image in make_images(extra_dims=[()]):
            image_pil = to_image_pil(image)
            image_numpy = np.array(image_pil)

            assert_equal(prototype_transform(image_pil), legacy_transform(image_pil))
            assert_equal(prototype_transform(image_numpy), legacy_transform(image_numpy))


class TestAATransforms:
    @pytest.mark.parametrize(
        "inpt",
        [
            torch.randint(0, 256, size=(1, 3, 256, 256), dtype=torch.uint8),
            PIL.Image.new("RGB", (256, 256), 123),
            features.Image(torch.randint(0, 256, size=(1, 3, 256, 256), dtype=torch.uint8)),
        ],
    )
    @pytest.mark.parametrize(
        "interpolation",
        [prototype_transforms.InterpolationMode.NEAREST, prototype_transforms.InterpolationMode.BILINEAR],
    )
    def test_randaug(self, inpt, interpolation, mocker):
        t_ref = legacy_transforms.RandAugment(interpolation=interpolation, num_ops=1)
        t = prototype_transforms.RandAugment(interpolation=interpolation, num_ops=1)

        le = len(t._AUGMENTATION_SPACE)
        keys = list(t._AUGMENTATION_SPACE.keys())
        randint_values = []
        for i in range(le):
            # Stable API, op_index random call
            randint_values.append(i)
            # Stable API, if signed there is another random call
            if t._AUGMENTATION_SPACE[keys[i]][1]:
                randint_values.append(0)
            # New API, _get_random_item
            randint_values.append(i)
        randint_values = iter(randint_values)

        mocker.patch("torch.randint", side_effect=lambda *arg, **kwargs: torch.tensor(next(randint_values)))
        mocker.patch("torch.rand", return_value=1.0)

        for i in range(le):
            expected_output = t_ref(inpt)
            output = t(inpt)

            assert_equal(expected_output, output)

    @pytest.mark.parametrize(
        "inpt",
        [
            torch.randint(0, 256, size=(1, 3, 256, 256), dtype=torch.uint8),
            PIL.Image.new("RGB", (256, 256), 123),
            features.Image(torch.randint(0, 256, size=(1, 3, 256, 256), dtype=torch.uint8)),
        ],
    )
    @pytest.mark.parametrize(
        "interpolation",
        [prototype_transforms.InterpolationMode.NEAREST, prototype_transforms.InterpolationMode.BILINEAR],
    )
    def test_trivial_aug(self, inpt, interpolation, mocker):
        t_ref = legacy_transforms.TrivialAugmentWide(interpolation=interpolation)
        t = prototype_transforms.TrivialAugmentWide(interpolation=interpolation)

        le = len(t._AUGMENTATION_SPACE)
        keys = list(t._AUGMENTATION_SPACE.keys())
        randint_values = []
        for i in range(le):
            # Stable API, op_index random call
            randint_values.append(i)
            key = keys[i]
            # Stable API, random magnitude
            aug_op = t._AUGMENTATION_SPACE[key]
            magnitudes = aug_op[0](2, 0, 0)
            if magnitudes is not None:
                randint_values.append(5)
            # Stable API, if signed there is another random call
            if aug_op[1]:
                randint_values.append(0)
            # New API, _get_random_item
            randint_values.append(i)
            # New API, random magnitude
            if magnitudes is not None:
                randint_values.append(5)

        randint_values = iter(randint_values)

        mocker.patch("torch.randint", side_effect=lambda *arg, **kwargs: torch.tensor(next(randint_values)))
        mocker.patch("torch.rand", return_value=1.0)

        for _ in range(le):
            expected_output = t_ref(inpt)
            output = t(inpt)

            assert_equal(expected_output, output)

    @pytest.mark.parametrize(
        "inpt",
        [
            torch.randint(0, 256, size=(1, 3, 256, 256), dtype=torch.uint8),
            PIL.Image.new("RGB", (256, 256), 123),
            features.Image(torch.randint(0, 256, size=(1, 3, 256, 256), dtype=torch.uint8)),
        ],
    )
    @pytest.mark.parametrize(
        "interpolation",
        [prototype_transforms.InterpolationMode.NEAREST, prototype_transforms.InterpolationMode.BILINEAR],
    )
    def test_augmix(self, inpt, interpolation, mocker):
        t_ref = legacy_transforms.AugMix(interpolation=interpolation, mixture_width=1, chain_depth=1)
        t_ref._sample_dirichlet = lambda t: t.softmax(dim=-1)
        t = prototype_transforms.AugMix(interpolation=interpolation, mixture_width=1, chain_depth=1)
        t._sample_dirichlet = lambda t: t.softmax(dim=-1)

        le = len(t._AUGMENTATION_SPACE)
        keys = list(t._AUGMENTATION_SPACE.keys())
        randint_values = []
        for i in range(le):
            # Stable API, op_index random call
            randint_values.append(i)
            key = keys[i]
            # Stable API, random magnitude
            aug_op = t._AUGMENTATION_SPACE[key]
            magnitudes = aug_op[0](2, 0, 0)
            if magnitudes is not None:
                randint_values.append(5)
            # Stable API, if signed there is another random call
            if aug_op[1]:
                randint_values.append(0)
            # New API, _get_random_item
            randint_values.append(i)
            # New API, random magnitude
            if magnitudes is not None:
                randint_values.append(5)

        randint_values = iter(randint_values)

        mocker.patch("torch.randint", side_effect=lambda *arg, **kwargs: torch.tensor(next(randint_values)))
        mocker.patch("torch.rand", return_value=1.0)

        expected_output = t_ref(inpt)
        output = t(inpt)

        assert_equal(expected_output, output)

    @pytest.mark.parametrize(
        "inpt",
        [
            torch.randint(0, 256, size=(1, 3, 256, 256), dtype=torch.uint8),
            PIL.Image.new("RGB", (256, 256), 123),
            features.Image(torch.randint(0, 256, size=(1, 3, 256, 256), dtype=torch.uint8)),
        ],
    )
    @pytest.mark.parametrize(
        "interpolation",
        [prototype_transforms.InterpolationMode.NEAREST, prototype_transforms.InterpolationMode.BILINEAR],
    )
    def test_aa(self, inpt, interpolation):
        aa_policy = legacy_transforms.AutoAugmentPolicy("imagenet")
        t_ref = legacy_transforms.AutoAugment(aa_policy, interpolation=interpolation)
        t = prototype_transforms.AutoAugment(aa_policy, interpolation=interpolation)

        torch.manual_seed(12)
        expected_output = t_ref(inpt)

        torch.manual_seed(12)
        output = t(inpt)

        assert_equal(expected_output, output)
