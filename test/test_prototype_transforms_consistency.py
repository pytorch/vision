import enum
import inspect

import numpy as np
import PIL.Image
import pytest

import torch
from prototype_common_utils import ArgsKwargs, assert_equal, make_bounding_box, make_images
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


class TestRefDetTransforms:
    def make_datapoints(self, with_mask=True):
        size = (600, 800)

        pil_image = PIL.Image.new("RGB", size[::-1], 123)
        target = {
            "boxes": make_bounding_box(image_size=size, format="XYXY", extra_dims=(22,), dtype=torch.float),
            "labels": features.Label(torch.randint(0, 81, size=(22,))),
        }
        if with_mask:
            target["masks"] = features.SegmentationMask(torch.randint(0, 2, size=(22, *size), dtype=torch.long))

        yield (pil_image, target)

        tensor_image = torch.randint(0, 256, size=(3, *size), dtype=torch.uint8)
        target = {
            "boxes": make_bounding_box(image_size=size, format="XYXY", extra_dims=(22,), dtype=torch.float),
            "labels": features.Label(torch.randint(0, 81, size=(22,))),
        }
        if with_mask:
            target["masks"] = features.SegmentationMask(torch.randint(0, 2, size=(22, *size), dtype=torch.long))

        yield (tensor_image, target)

        feature_image = features.Image(torch.randint(0, 256, size=(3, *size), dtype=torch.uint8))
        target = {
            "boxes": make_bounding_box(image_size=size, format="XYXY", extra_dims=(22,), dtype=torch.float),
            "labels": features.Label(torch.randint(0, 81, size=(22,))),
        }
        if with_mask:
            target["masks"] = features.SegmentationMask(torch.randint(0, 2, size=(22, *size), dtype=torch.long))

        yield (feature_image, target)

    def _test_transform(self, t_ref, t, data_kwargs={}):
        for dp in self.make_datapoints(**data_kwargs):

            # We should use prototype transform first as reference transform performs inplace target update
            torch.manual_seed(12)
            output = t(dp)

            torch.manual_seed(12)
            expected_output = t_ref(*dp)

            assert_equal(expected_output, output)

    def test_randomhorizontalflip(self):
        t_ref = RandomHorizontalFlip(p=1.0)
        t = prototype_transforms.RandomHorizontalFlip(p=1.0)
        self._test_transform(t_ref, t)

    def test_randomioucrop(self):
        t_ref = RandomIoUCrop()
        t = prototype_transforms.RandomIoUCrop()
        self._test_transform(t_ref, t, {"with_mask": False})

    def test_randomzoomout(self):
        t_ref = RandomZoomOut()
        t = prototype_transforms.RandomZoomOut()
        self._test_transform(t_ref, t, {"with_mask": False})

    def test_scalejitter(self):
        t_ref = ScaleJitter((1024, 1024))
        t = prototype_transforms.ScaleJitter((1024, 1024))
        self._test_transform(t_ref, t)

    def test_fixedsizecrop(self):
        t_ref = FixedSizeCrop(size=(1024, 1024), fill=0)
        t = prototype_transforms.FixedSizeCrop(size=(1024, 1024), fill=0)
        self._test_transform(t_ref, t)

    def test_randomshortestsize(self):
        t_ref = RandomShortestSize(min_size=(480, 512, 544, 576, 608, 640, 672, 704, 736, 768, 800), max_size=1333)
        t = prototype_transforms.RandomShortestSize(
            min_size=(480, 512, 544, 576, 608, 640, 672, 704, 736, 768, 800), max_size=1333
        )
        self._test_transform(t_ref, t)


# -----
# Dumped reference detection transforms here for consistency checks
# torchvision/references/detection/transforms.py
# -----
from typing import Dict, List, Optional, Tuple, Union

import torch
import torchvision
from torch import nn, Tensor
from torchvision import ops
from torchvision.transforms import functional as F, InterpolationMode, transforms as T


def _flip_coco_person_keypoints(kps, width):
    flip_inds = [0, 2, 1, 4, 3, 6, 5, 8, 7, 10, 9, 12, 11, 14, 13, 16, 15]
    flipped_data = kps[:, flip_inds]
    flipped_data[..., 0] = width - flipped_data[..., 0]
    # Maintain COCO convention that if visibility == 0, then x, y = 0
    inds = flipped_data[..., 2] == 0
    flipped_data[inds] = 0
    return flipped_data


class Compose:
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, image, target):
        for t in self.transforms:
            image, target = t(image, target)
        return image, target


class RandomHorizontalFlip(T.RandomHorizontalFlip):
    def forward(
        self, image: Tensor, target: Optional[Dict[str, Tensor]] = None
    ) -> Tuple[Tensor, Optional[Dict[str, Tensor]]]:
        if torch.rand(1) < self.p:
            image = F.hflip(image)
            if target is not None:
                _, _, width = F.get_dimensions(image)
                target["boxes"][:, [0, 2]] = width - target["boxes"][:, [2, 0]]
                if "masks" in target:
                    target["masks"] = target["masks"].flip(-1)
                if "keypoints" in target:
                    keypoints = target["keypoints"]
                    keypoints = _flip_coco_person_keypoints(keypoints, width)
                    target["keypoints"] = keypoints
        return image, target


class PILToTensor(nn.Module):
    def forward(
        self, image: Tensor, target: Optional[Dict[str, Tensor]] = None
    ) -> Tuple[Tensor, Optional[Dict[str, Tensor]]]:
        image = F.pil_to_tensor(image)
        return image, target


class ConvertImageDtype(nn.Module):
    def __init__(self, dtype: torch.dtype) -> None:
        super().__init__()
        self.dtype = dtype

    def forward(
        self, image: Tensor, target: Optional[Dict[str, Tensor]] = None
    ) -> Tuple[Tensor, Optional[Dict[str, Tensor]]]:
        image = F.convert_image_dtype(image, self.dtype)
        return image, target


class RandomIoUCrop(nn.Module):
    def __init__(
        self,
        min_scale: float = 0.3,
        max_scale: float = 1.0,
        min_aspect_ratio: float = 0.5,
        max_aspect_ratio: float = 2.0,
        sampler_options: Optional[List[float]] = None,
        trials: int = 40,
    ):
        super().__init__()
        # Configuration similar to https://github.com/weiliu89/caffe/blob/ssd/examples/ssd/ssd_coco.py#L89-L174
        self.min_scale = min_scale
        self.max_scale = max_scale
        self.min_aspect_ratio = min_aspect_ratio
        self.max_aspect_ratio = max_aspect_ratio
        if sampler_options is None:
            sampler_options = [0.0, 0.1, 0.3, 0.5, 0.7, 0.9, 1.0]
        self.options = sampler_options
        self.trials = trials

    def forward(
        self, image: Tensor, target: Optional[Dict[str, Tensor]] = None
    ) -> Tuple[Tensor, Optional[Dict[str, Tensor]]]:
        if target is None:
            raise ValueError("The targets can't be None for this transform.")

        if isinstance(image, torch.Tensor):
            if image.ndimension() not in {2, 3}:
                raise ValueError(f"image should be 2/3 dimensional. Got {image.ndimension()} dimensions.")
            elif image.ndimension() == 2:
                image = image.unsqueeze(0)

        _, orig_h, orig_w = F.get_dimensions(image)

        while True:
            # sample an option
            idx = int(torch.randint(low=0, high=len(self.options), size=(1,)))
            min_jaccard_overlap = self.options[idx]
            if min_jaccard_overlap >= 1.0:  # a value larger than 1 encodes the leave as-is option
                return image, target

            for _ in range(self.trials):
                # check the aspect ratio limitations
                r = self.min_scale + (self.max_scale - self.min_scale) * torch.rand(2)
                new_w = int(orig_w * r[0])
                new_h = int(orig_h * r[1])
                aspect_ratio = new_w / new_h
                if not (self.min_aspect_ratio <= aspect_ratio <= self.max_aspect_ratio):
                    continue

                # check for 0 area crops
                r = torch.rand(2)
                left = int((orig_w - new_w) * r[0])
                top = int((orig_h - new_h) * r[1])
                right = left + new_w
                bottom = top + new_h
                if left == right or top == bottom:
                    continue

                # check for any valid boxes with centers within the crop area
                cx = 0.5 * (target["boxes"][:, 0] + target["boxes"][:, 2])
                cy = 0.5 * (target["boxes"][:, 1] + target["boxes"][:, 3])
                is_within_crop_area = (left < cx) & (cx < right) & (top < cy) & (cy < bottom)
                if not is_within_crop_area.any():
                    continue

                # check at least 1 box with jaccard limitations
                boxes = target["boxes"][is_within_crop_area]
                ious = torchvision.ops.boxes.box_iou(
                    boxes, torch.tensor([[left, top, right, bottom]], dtype=boxes.dtype, device=boxes.device)
                )
                if ious.max() < min_jaccard_overlap:
                    continue

                # keep only valid boxes and perform cropping
                target["boxes"] = boxes
                target["labels"] = target["labels"][is_within_crop_area]
                target["boxes"][:, 0::2] -= left
                target["boxes"][:, 1::2] -= top
                target["boxes"][:, 0::2].clamp_(min=0, max=new_w)
                target["boxes"][:, 1::2].clamp_(min=0, max=new_h)
                image = F.crop(image, top, left, new_h, new_w)

                return image, target


class RandomZoomOut(nn.Module):
    def __init__(
        self, fill: Optional[List[float]] = None, side_range: Tuple[float, float] = (1.0, 4.0), p: float = 0.5
    ):
        super().__init__()
        if fill is None:
            fill = [0.0, 0.0, 0.0]
        self.fill = fill
        self.side_range = side_range
        if side_range[0] < 1.0 or side_range[0] > side_range[1]:
            raise ValueError(f"Invalid canvas side range provided {side_range}.")
        self.p = p

    @torch.jit.unused
    def _get_fill_value(self, is_pil):
        # type: (bool) -> int
        # We fake the type to make it work on JIT
        return tuple(int(x) for x in self.fill) if is_pil else 0

    def forward(
        self, image: Tensor, target: Optional[Dict[str, Tensor]] = None
    ) -> Tuple[Tensor, Optional[Dict[str, Tensor]]]:
        if isinstance(image, torch.Tensor):
            if image.ndimension() not in {2, 3}:
                raise ValueError(f"image should be 2/3 dimensional. Got {image.ndimension()} dimensions.")
            elif image.ndimension() == 2:
                image = image.unsqueeze(0)

        if torch.rand(1) >= self.p:
            return image, target

        _, orig_h, orig_w = F.get_dimensions(image)

        r = self.side_range[0] + torch.rand(1) * (self.side_range[1] - self.side_range[0])
        canvas_width = int(orig_w * r)
        canvas_height = int(orig_h * r)

        r = torch.rand(2)
        left = int((canvas_width - orig_w) * r[0])
        top = int((canvas_height - orig_h) * r[1])
        right = canvas_width - (left + orig_w)
        bottom = canvas_height - (top + orig_h)

        if torch.jit.is_scripting():
            fill = 0
        else:
            fill = self._get_fill_value(F._is_pil_image(image))

        image = F.pad(image, [left, top, right, bottom], fill=fill)
        if isinstance(image, torch.Tensor):
            # PyTorch's pad supports only integers on fill. So we need to overwrite the colour
            v = torch.tensor(self.fill, device=image.device, dtype=image.dtype).view(-1, 1, 1)
            image[..., :top, :] = image[..., :, :left] = image[..., (top + orig_h) :, :] = image[
                ..., :, (left + orig_w) :
            ] = v

        if target is not None:
            target["boxes"][:, 0::2] += left
            target["boxes"][:, 1::2] += top

        return image, target


class RandomPhotometricDistort(nn.Module):
    def __init__(
        self,
        contrast: Tuple[float, float] = (0.5, 1.5),
        saturation: Tuple[float, float] = (0.5, 1.5),
        hue: Tuple[float, float] = (-0.05, 0.05),
        brightness: Tuple[float, float] = (0.875, 1.125),
        p: float = 0.5,
    ):
        super().__init__()
        self._brightness = T.ColorJitter(brightness=brightness)
        self._contrast = T.ColorJitter(contrast=contrast)
        self._hue = T.ColorJitter(hue=hue)
        self._saturation = T.ColorJitter(saturation=saturation)
        self.p = p

    def forward(
        self, image: Tensor, target: Optional[Dict[str, Tensor]] = None
    ) -> Tuple[Tensor, Optional[Dict[str, Tensor]]]:
        if isinstance(image, torch.Tensor):
            if image.ndimension() not in {2, 3}:
                raise ValueError(f"image should be 2/3 dimensional. Got {image.ndimension()} dimensions.")
            elif image.ndimension() == 2:
                image = image.unsqueeze(0)

        r = torch.rand(7)

        if r[0] < self.p:
            image = self._brightness(image)

        contrast_before = r[1] < 0.5
        if contrast_before:
            if r[2] < self.p:
                image = self._contrast(image)

        if r[3] < self.p:
            image = self._saturation(image)

        if r[4] < self.p:
            image = self._hue(image)

        if not contrast_before:
            if r[5] < self.p:
                image = self._contrast(image)

        if r[6] < self.p:
            channels, _, _ = F.get_dimensions(image)
            permutation = torch.randperm(channels)

            is_pil = F._is_pil_image(image)
            if is_pil:
                image = F.pil_to_tensor(image)
                image = F.convert_image_dtype(image)
            image = image[..., permutation, :, :]
            if is_pil:
                image = F.to_pil_image(image)

        return image, target


class ScaleJitter(nn.Module):
    """Randomly resizes the image and its bounding boxes  within the specified scale range.
    The class implements the Scale Jitter augmentation as described in the paper
    `"Simple Copy-Paste is a Strong Data Augmentation Method for Instance Segmentation" <https://arxiv.org/abs/2012.07177>`_.

    Args:
        target_size (tuple of ints): The target size for the transform provided in (height, weight) format.
        scale_range (tuple of ints): scaling factor interval, e.g (a, b), then scale is randomly sampled from the
            range a <= scale <= b.
        interpolation (InterpolationMode): Desired interpolation enum defined by
            :class:`torchvision.transforms.InterpolationMode`. Default is ``InterpolationMode.BILINEAR``.
    """

    def __init__(
        self,
        target_size: Tuple[int, int],
        scale_range: Tuple[float, float] = (0.1, 2.0),
        interpolation: InterpolationMode = InterpolationMode.BILINEAR,
    ):
        super().__init__()
        self.target_size = target_size
        self.scale_range = scale_range
        self.interpolation = interpolation

    def forward(
        self, image: Tensor, target: Optional[Dict[str, Tensor]] = None
    ) -> Tuple[Tensor, Optional[Dict[str, Tensor]]]:
        if isinstance(image, torch.Tensor):
            if image.ndimension() not in {2, 3}:
                raise ValueError(f"image should be 2/3 dimensional. Got {image.ndimension()} dimensions.")
            elif image.ndimension() == 2:
                image = image.unsqueeze(0)

        _, orig_height, orig_width = F.get_dimensions(image)

        scale = self.scale_range[0] + torch.rand(1) * (self.scale_range[1] - self.scale_range[0])
        r = min(self.target_size[1] / orig_height, self.target_size[0] / orig_width) * scale
        new_width = int(orig_width * r)
        new_height = int(orig_height * r)

        image = F.resize(image, [new_height, new_width], interpolation=self.interpolation)

        if target is not None:
            target["boxes"][:, 0::2] *= new_width / orig_width
            target["boxes"][:, 1::2] *= new_height / orig_height
            if "masks" in target:
                target["masks"] = F.resize(
                    target["masks"], [new_height, new_width], interpolation=InterpolationMode.NEAREST
                )

        return image, target


class FixedSizeCrop(nn.Module):
    def __init__(self, size, fill=0, padding_mode="constant"):
        super().__init__()
        size = tuple(T._setup_size(size, error_msg="Please provide only two dimensions (h, w) for size."))
        self.crop_height = size[0]
        self.crop_width = size[1]
        self.fill = fill  # TODO: Fill is currently respected only on PIL. Apply tensor patch.
        self.padding_mode = padding_mode

    def _pad(self, img, target, padding):
        # Taken from the functional_tensor.py pad
        if isinstance(padding, int):
            pad_left = pad_right = pad_top = pad_bottom = padding
        elif len(padding) == 1:
            pad_left = pad_right = pad_top = pad_bottom = padding[0]
        elif len(padding) == 2:
            pad_left = pad_right = padding[0]
            pad_top = pad_bottom = padding[1]
        else:
            pad_left = padding[0]
            pad_top = padding[1]
            pad_right = padding[2]
            pad_bottom = padding[3]

        padding = [pad_left, pad_top, pad_right, pad_bottom]
        img = F.pad(img, padding, self.fill, self.padding_mode)
        if target is not None:
            target["boxes"][:, 0::2] += pad_left
            target["boxes"][:, 1::2] += pad_top
            if "masks" in target:
                target["masks"] = F.pad(target["masks"], padding, 0, "constant")

        return img, target

    def _crop(self, img, target, top, left, height, width):
        img = F.crop(img, top, left, height, width)
        if target is not None:
            boxes = target["boxes"]
            boxes[:, 0::2] -= left
            boxes[:, 1::2] -= top
            boxes[:, 0::2].clamp_(min=0, max=width)
            boxes[:, 1::2].clamp_(min=0, max=height)

            is_valid = (boxes[:, 0] < boxes[:, 2]) & (boxes[:, 1] < boxes[:, 3])

            target["boxes"] = boxes[is_valid]
            target["labels"] = target["labels"][is_valid]
            if "masks" in target:
                target["masks"] = F.crop(target["masks"][is_valid], top, left, height, width)

        return img, target

    def forward(self, img, target=None):
        _, height, width = F.get_dimensions(img)
        new_height = min(height, self.crop_height)
        new_width = min(width, self.crop_width)

        if new_height != height or new_width != width:
            offset_height = max(height - self.crop_height, 0)
            offset_width = max(width - self.crop_width, 0)

            r = torch.rand(1)
            top = int(offset_height * r)
            left = int(offset_width * r)

            img, target = self._crop(img, target, top, left, new_height, new_width)

        pad_bottom = max(self.crop_height - new_height, 0)
        pad_right = max(self.crop_width - new_width, 0)
        if pad_bottom != 0 or pad_right != 0:
            img, target = self._pad(img, target, [0, 0, pad_right, pad_bottom])

        return img, target


class RandomShortestSize(nn.Module):
    def __init__(
        self,
        min_size: Union[List[int], Tuple[int], int],
        max_size: int,
        interpolation: InterpolationMode = InterpolationMode.BILINEAR,
    ):
        super().__init__()
        self.min_size = [min_size] if isinstance(min_size, int) else list(min_size)
        self.max_size = max_size
        self.interpolation = interpolation

    def forward(
        self, image: Tensor, target: Optional[Dict[str, Tensor]] = None
    ) -> Tuple[Tensor, Optional[Dict[str, Tensor]]]:
        _, orig_height, orig_width = F.get_dimensions(image)

        min_size = self.min_size[torch.randint(len(self.min_size), (1,)).item()]
        r = min(min_size / min(orig_height, orig_width), self.max_size / max(orig_height, orig_width))

        new_width = int(orig_width * r)
        new_height = int(orig_height * r)

        image = F.resize(image, [new_height, new_width], interpolation=self.interpolation)

        if target is not None:
            target["boxes"][:, 0::2] *= new_width / orig_width
            target["boxes"][:, 1::2] *= new_height / orig_height
            if "masks" in target:
                target["masks"] = F.resize(
                    target["masks"], [new_height, new_width], interpolation=InterpolationMode.NEAREST
                )

        return image, target


def _copy_paste(
    image: torch.Tensor,
    target: Dict[str, Tensor],
    paste_image: torch.Tensor,
    paste_target: Dict[str, Tensor],
    blending: bool = True,
    resize_interpolation: F.InterpolationMode = F.InterpolationMode.BILINEAR,
) -> Tuple[torch.Tensor, Dict[str, Tensor]]:

    # Random paste targets selection:
    num_masks = len(paste_target["masks"])

    if num_masks < 1:
        # Such degerante case with num_masks=0 can happen with LSJ
        # Let's just return (image, target)
        return image, target

    # We have to please torch script by explicitly specifying dtype as torch.long
    random_selection = torch.randint(0, num_masks, (num_masks,), device=paste_image.device)
    random_selection = torch.unique(random_selection).to(torch.long)

    paste_masks = paste_target["masks"][random_selection]
    paste_boxes = paste_target["boxes"][random_selection]
    paste_labels = paste_target["labels"][random_selection]

    masks = target["masks"]

    # We resize source and paste data if they have different sizes
    # This is something we introduced here as originally the algorithm works
    # on equal-sized data (for example, coming from LSJ data augmentations)
    size1 = image.shape[-2:]
    size2 = paste_image.shape[-2:]
    if size1 != size2:
        paste_image = F.resize(paste_image, size1, interpolation=resize_interpolation)
        paste_masks = F.resize(paste_masks, size1, interpolation=F.InterpolationMode.NEAREST)
        # resize bboxes:
        ratios = torch.tensor((size1[1] / size2[1], size1[0] / size2[0]), device=paste_boxes.device)
        paste_boxes = paste_boxes.view(-1, 2, 2).mul(ratios).view(paste_boxes.shape)

    paste_alpha_mask = paste_masks.sum(dim=0) > 0

    if blending:
        paste_alpha_mask = F.gaussian_blur(
            paste_alpha_mask.unsqueeze(0),
            kernel_size=(5, 5),
            sigma=[
                2.0,
            ],
        )

    # Copy-paste images:
    image = (image * (~paste_alpha_mask)) + (paste_image * paste_alpha_mask)

    # Copy-paste masks:
    masks = masks * (~paste_alpha_mask)
    non_all_zero_masks = masks.sum((-1, -2)) > 0
    masks = masks[non_all_zero_masks]

    # Do a shallow copy of the target dict
    out_target = {k: v for k, v in target.items()}

    out_target["masks"] = torch.cat([masks, paste_masks])

    # Copy-paste boxes and labels
    boxes = ops.masks_to_boxes(masks)
    out_target["boxes"] = torch.cat([boxes, paste_boxes])

    labels = target["labels"][non_all_zero_masks]
    out_target["labels"] = torch.cat([labels, paste_labels])

    # Update additional optional keys: area and iscrowd if exist
    if "area" in target:
        out_target["area"] = out_target["masks"].sum((-1, -2)).to(torch.float32)

    if "iscrowd" in target and "iscrowd" in paste_target:
        # target['iscrowd'] size can be differ from mask size (non_all_zero_masks)
        # For example, if previous transforms geometrically modifies masks/boxes/labels but
        # does not update "iscrowd"
        if len(target["iscrowd"]) == len(non_all_zero_masks):
            iscrowd = target["iscrowd"][non_all_zero_masks]
            paste_iscrowd = paste_target["iscrowd"][random_selection]
            out_target["iscrowd"] = torch.cat([iscrowd, paste_iscrowd])

    # Check for degenerated boxes and remove them
    boxes = out_target["boxes"]
    degenerate_boxes = boxes[:, 2:] <= boxes[:, :2]
    if degenerate_boxes.any():
        valid_targets = ~degenerate_boxes.any(dim=1)

        out_target["boxes"] = boxes[valid_targets]
        out_target["masks"] = out_target["masks"][valid_targets]
        out_target["labels"] = out_target["labels"][valid_targets]

        if "area" in out_target:
            out_target["area"] = out_target["area"][valid_targets]
        if "iscrowd" in out_target and len(out_target["iscrowd"]) == len(valid_targets):
            out_target["iscrowd"] = out_target["iscrowd"][valid_targets]

    return image, out_target


class SimpleCopyPaste(torch.nn.Module):
    def __init__(self, blending=True, resize_interpolation=F.InterpolationMode.BILINEAR):
        super().__init__()
        self.resize_interpolation = resize_interpolation
        self.blending = blending

    def forward(
        self, images: List[torch.Tensor], targets: List[Dict[str, Tensor]]
    ) -> Tuple[List[torch.Tensor], List[Dict[str, Tensor]]]:
        torch._assert(
            isinstance(images, (list, tuple)) and all([isinstance(v, torch.Tensor) for v in images]),
            "images should be a list of tensors",
        )
        torch._assert(
            isinstance(targets, (list, tuple)) and len(images) == len(targets),
            "targets should be a list of the same size as images",
        )
        for target in targets:
            # Can not check for instance type dict with inside torch.jit.script
            # torch._assert(isinstance(target, dict), "targets item should be a dict")
            for k in ["masks", "boxes", "labels"]:
                torch._assert(k in target, f"Key {k} should be present in targets")
                torch._assert(isinstance(target[k], torch.Tensor), f"Value for the key {k} should be a tensor")

        # images = [t1, t2, ..., tN]
        # Let's define paste_images as shifted list of input images
        # paste_images = [t2, t3, ..., tN, t1]
        # FYI: in TF they mix data on the dataset level
        images_rolled = images[-1:] + images[:-1]
        targets_rolled = targets[-1:] + targets[:-1]

        output_images: List[torch.Tensor] = []
        output_targets: List[Dict[str, Tensor]] = []

        for image, target, paste_image, paste_target in zip(images, targets, images_rolled, targets_rolled):
            output_image, output_data = _copy_paste(
                image,
                target,
                paste_image,
                paste_target,
                blending=self.blending,
                resize_interpolation=self.resize_interpolation,
            )
            output_images.append(output_image)
            output_targets.append(output_data)

        return output_images, output_targets

    def __repr__(self) -> str:
        s = f"{self.__class__.__name__}(blending={self.blending}, resize_interpolation={self.resize_interpolation})"
        return s
