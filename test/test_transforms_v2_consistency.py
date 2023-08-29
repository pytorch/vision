import importlib.machinery
import importlib.util
import inspect
import random
import re
from pathlib import Path

import numpy as np
import PIL.Image
import pytest

import torch
import torchvision.transforms.v2 as v2_transforms
from common_utils import assert_close, assert_equal, set_rng_seed
from torch import nn
from torchvision import transforms as legacy_transforms, tv_tensors
from torchvision._utils import sequence_to_str

from torchvision.transforms import functional as legacy_F
from torchvision.transforms.v2 import functional as prototype_F
from torchvision.transforms.v2._utils import _get_fill, query_size
from torchvision.transforms.v2.functional import to_pil_image
from transforms_v2_legacy_utils import (
    ArgsKwargs,
    make_bounding_boxes,
    make_detection_mask,
    make_image,
    make_images,
    make_segmentation_mask,
)

DEFAULT_MAKE_IMAGES_KWARGS = dict(color_spaces=["RGB"], extra_dims=[(4,)])


@pytest.fixture(autouse=True)
def fix_rng_seed():
    set_rng_seed(0)
    yield


class NotScriptableArgsKwargs(ArgsKwargs):
    """
    This class is used to mark parameters that render the transform non-scriptable. They still work in eager mode and
    thus will be tested there, but will be skipped by the JIT tests.
    """

    pass


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
        closeness_kwargs=None,
    ):
        self.prototype_cls = prototype_cls
        self.legacy_cls = legacy_cls
        self.args_kwargs = args_kwargs
        self.make_images_kwargs = make_images_kwargs or DEFAULT_MAKE_IMAGES_KWARGS
        self.supports_pil = supports_pil
        self.removed_params = removed_params
        self.closeness_kwargs = closeness_kwargs or dict(rtol=0, atol=0)


# These are here since both the prototype and legacy transform need to be constructed with the same random parameters
LINEAR_TRANSFORMATION_MEAN = torch.rand(36)
LINEAR_TRANSFORMATION_MATRIX = torch.rand([LINEAR_TRANSFORMATION_MEAN.numel()] * 2)

CONSISTENCY_CONFIGS = [
    ConsistencyConfig(
        v2_transforms.Normalize,
        legacy_transforms.Normalize,
        [
            ArgsKwargs(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ],
        supports_pil=False,
        make_images_kwargs=dict(DEFAULT_MAKE_IMAGES_KWARGS, dtypes=[torch.float]),
    ),
    ConsistencyConfig(
        v2_transforms.CenterCrop,
        legacy_transforms.CenterCrop,
        [
            ArgsKwargs(18),
            ArgsKwargs((18, 13)),
        ],
    ),
    ConsistencyConfig(
        v2_transforms.FiveCrop,
        legacy_transforms.FiveCrop,
        [
            ArgsKwargs(18),
            ArgsKwargs((18, 13)),
        ],
        make_images_kwargs=dict(DEFAULT_MAKE_IMAGES_KWARGS, sizes=[(20, 19)]),
    ),
    ConsistencyConfig(
        v2_transforms.TenCrop,
        legacy_transforms.TenCrop,
        [
            ArgsKwargs(18),
            ArgsKwargs((18, 13)),
            ArgsKwargs(18, vertical_flip=True),
        ],
        make_images_kwargs=dict(DEFAULT_MAKE_IMAGES_KWARGS, sizes=[(20, 19)]),
    ),
    ConsistencyConfig(
        v2_transforms.Pad,
        legacy_transforms.Pad,
        [
            NotScriptableArgsKwargs(3),
            ArgsKwargs([3]),
            ArgsKwargs([2, 3]),
            ArgsKwargs([3, 2, 1, 4]),
            NotScriptableArgsKwargs(5, fill=1, padding_mode="constant"),
            ArgsKwargs([5], fill=1, padding_mode="constant"),
            NotScriptableArgsKwargs(5, padding_mode="edge"),
            NotScriptableArgsKwargs(5, padding_mode="reflect"),
            NotScriptableArgsKwargs(5, padding_mode="symmetric"),
        ],
    ),
    *[
        ConsistencyConfig(
            v2_transforms.LinearTransformation,
            legacy_transforms.LinearTransformation,
            [
                ArgsKwargs(LINEAR_TRANSFORMATION_MATRIX.to(matrix_dtype), LINEAR_TRANSFORMATION_MEAN.to(matrix_dtype)),
            ],
            # Make sure that the product of the height, width and number of channels matches the number of elements in
            # `LINEAR_TRANSFORMATION_MEAN`. For example 2 * 6 * 3 == 4 * 3 * 3 == 36.
            make_images_kwargs=dict(
                DEFAULT_MAKE_IMAGES_KWARGS, sizes=[(2, 6), (4, 3)], color_spaces=["RGB"], dtypes=[image_dtype]
            ),
            supports_pil=False,
        )
        for matrix_dtype, image_dtype in [
            (torch.float32, torch.float32),
            (torch.float64, torch.float64),
            (torch.float32, torch.uint8),
            (torch.float64, torch.float32),
            (torch.float32, torch.float64),
        ]
    ],
    ConsistencyConfig(
        v2_transforms.Grayscale,
        legacy_transforms.Grayscale,
        [
            ArgsKwargs(num_output_channels=1),
            ArgsKwargs(num_output_channels=3),
        ],
        make_images_kwargs=dict(DEFAULT_MAKE_IMAGES_KWARGS, color_spaces=["RGB", "GRAY"]),
        # Use default tolerances of `torch.testing.assert_close`
        closeness_kwargs=dict(rtol=None, atol=None),
    ),
    ConsistencyConfig(
        v2_transforms.ToPILImage,
        legacy_transforms.ToPILImage,
        [NotScriptableArgsKwargs()],
        make_images_kwargs=dict(
            color_spaces=[
                "GRAY",
                "GRAY_ALPHA",
                "RGB",
                "RGBA",
            ],
            extra_dims=[()],
        ),
        supports_pil=False,
    ),
    ConsistencyConfig(
        v2_transforms.Lambda,
        legacy_transforms.Lambda,
        [
            NotScriptableArgsKwargs(lambda image: image / 2),
        ],
        # Technically, this also supports PIL, but it is overkill to write a function here that supports tensor and PIL
        # images given that the transform does nothing but call it anyway.
        supports_pil=False,
    ),
    ConsistencyConfig(
        v2_transforms.RandomEqualize,
        legacy_transforms.RandomEqualize,
        [
            ArgsKwargs(p=0),
            ArgsKwargs(p=1),
        ],
        make_images_kwargs=dict(DEFAULT_MAKE_IMAGES_KWARGS, dtypes=[torch.uint8]),
    ),
    ConsistencyConfig(
        v2_transforms.RandomInvert,
        legacy_transforms.RandomInvert,
        [
            ArgsKwargs(p=0),
            ArgsKwargs(p=1),
        ],
    ),
    ConsistencyConfig(
        v2_transforms.RandomPosterize,
        legacy_transforms.RandomPosterize,
        [
            ArgsKwargs(p=0, bits=5),
            ArgsKwargs(p=1, bits=1),
            ArgsKwargs(p=1, bits=3),
        ],
        make_images_kwargs=dict(DEFAULT_MAKE_IMAGES_KWARGS, dtypes=[torch.uint8]),
    ),
    ConsistencyConfig(
        v2_transforms.RandomSolarize,
        legacy_transforms.RandomSolarize,
        [
            ArgsKwargs(p=0, threshold=0.5),
            ArgsKwargs(p=1, threshold=0.3),
            ArgsKwargs(p=1, threshold=0.99),
        ],
    ),
    *[
        ConsistencyConfig(
            v2_transforms.RandomAutocontrast,
            legacy_transforms.RandomAutocontrast,
            [
                ArgsKwargs(p=0),
                ArgsKwargs(p=1),
            ],
            make_images_kwargs=dict(DEFAULT_MAKE_IMAGES_KWARGS, dtypes=[dt]),
            closeness_kwargs=ckw,
        )
        for dt, ckw in [(torch.uint8, dict(atol=1, rtol=0)), (torch.float32, dict(rtol=None, atol=None))]
    ],
    ConsistencyConfig(
        v2_transforms.RandomAdjustSharpness,
        legacy_transforms.RandomAdjustSharpness,
        [
            ArgsKwargs(p=0, sharpness_factor=0.5),
            ArgsKwargs(p=1, sharpness_factor=0.2),
            ArgsKwargs(p=1, sharpness_factor=0.99),
        ],
        closeness_kwargs={"atol": 1e-6, "rtol": 1e-6},
    ),
    ConsistencyConfig(
        v2_transforms.RandomGrayscale,
        legacy_transforms.RandomGrayscale,
        [
            ArgsKwargs(p=0),
            ArgsKwargs(p=1),
        ],
        make_images_kwargs=dict(DEFAULT_MAKE_IMAGES_KWARGS, color_spaces=["RGB", "GRAY"]),
        # Use default tolerances of `torch.testing.assert_close`
        closeness_kwargs=dict(rtol=None, atol=None),
    ),
    ConsistencyConfig(
        v2_transforms.RandomResizedCrop,
        legacy_transforms.RandomResizedCrop,
        [
            ArgsKwargs(16),
            ArgsKwargs(17, scale=(0.3, 0.7)),
            ArgsKwargs(25, ratio=(0.5, 1.5)),
            ArgsKwargs((31, 28), interpolation=v2_transforms.InterpolationMode.NEAREST),
            ArgsKwargs((31, 28), interpolation=PIL.Image.NEAREST),
            ArgsKwargs((29, 32), antialias=False),
            ArgsKwargs((28, 31), antialias=True),
        ],
        # atol=1 due to Resize v2 is using native uint8 interpolate path for bilinear and nearest modes
        closeness_kwargs=dict(rtol=0, atol=1),
    ),
    ConsistencyConfig(
        v2_transforms.RandomResizedCrop,
        legacy_transforms.RandomResizedCrop,
        [
            ArgsKwargs((33, 26), interpolation=v2_transforms.InterpolationMode.BICUBIC, antialias=True),
            ArgsKwargs((33, 26), interpolation=PIL.Image.BICUBIC, antialias=True),
        ],
        closeness_kwargs=dict(rtol=0, atol=21),
    ),
    ConsistencyConfig(
        v2_transforms.RandomErasing,
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
        v2_transforms.ColorJitter,
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
            ArgsKwargs(brightness=0.1, contrast=0.4, saturation=0.5, hue=0.3),
        ],
        closeness_kwargs={"atol": 1e-5, "rtol": 1e-5},
    ),
    ConsistencyConfig(
        v2_transforms.GaussianBlur,
        legacy_transforms.GaussianBlur,
        [
            ArgsKwargs(kernel_size=3),
            ArgsKwargs(kernel_size=(1, 5)),
            ArgsKwargs(kernel_size=3, sigma=0.7),
            ArgsKwargs(kernel_size=5, sigma=(0.3, 1.4)),
        ],
        closeness_kwargs={"rtol": 1e-5, "atol": 1e-5},
    ),
    ConsistencyConfig(
        v2_transforms.RandomCrop,
        legacy_transforms.RandomCrop,
        [
            ArgsKwargs(12),
            ArgsKwargs((15, 17)),
            NotScriptableArgsKwargs(11, padding=1),
            ArgsKwargs(11, padding=[1]),
            ArgsKwargs((8, 13), padding=(2, 3)),
            ArgsKwargs((14, 9), padding=(0, 2, 1, 0)),
            ArgsKwargs(36, pad_if_needed=True),
            ArgsKwargs((7, 8), fill=1),
            NotScriptableArgsKwargs(5, fill=(1, 2, 3)),
            ArgsKwargs(12),
            NotScriptableArgsKwargs(15, padding=2, padding_mode="edge"),
            ArgsKwargs(17, padding=(1, 0), padding_mode="reflect"),
            ArgsKwargs(8, padding=(3, 0, 0, 1), padding_mode="symmetric"),
        ],
        make_images_kwargs=dict(DEFAULT_MAKE_IMAGES_KWARGS, sizes=[(26, 26), (18, 33), (29, 22)]),
    ),
    ConsistencyConfig(
        v2_transforms.RandomPerspective,
        legacy_transforms.RandomPerspective,
        [
            ArgsKwargs(p=0),
            ArgsKwargs(p=1),
            ArgsKwargs(p=1, distortion_scale=0.3),
            ArgsKwargs(p=1, distortion_scale=0.2, interpolation=v2_transforms.InterpolationMode.NEAREST),
            ArgsKwargs(p=1, distortion_scale=0.2, interpolation=PIL.Image.NEAREST),
            ArgsKwargs(p=1, distortion_scale=0.1, fill=1),
            ArgsKwargs(p=1, distortion_scale=0.4, fill=(1, 2, 3)),
        ],
        closeness_kwargs={"atol": None, "rtol": None},
    ),
    ConsistencyConfig(
        v2_transforms.PILToTensor,
        legacy_transforms.PILToTensor,
    ),
    ConsistencyConfig(
        v2_transforms.ToTensor,
        legacy_transforms.ToTensor,
    ),
    ConsistencyConfig(
        v2_transforms.Compose,
        legacy_transforms.Compose,
    ),
    ConsistencyConfig(
        v2_transforms.RandomApply,
        legacy_transforms.RandomApply,
    ),
    ConsistencyConfig(
        v2_transforms.RandomChoice,
        legacy_transforms.RandomChoice,
    ),
    ConsistencyConfig(
        v2_transforms.RandomOrder,
        legacy_transforms.RandomOrder,
    ),
    ConsistencyConfig(
        v2_transforms.AugMix,
        legacy_transforms.AugMix,
    ),
    ConsistencyConfig(
        v2_transforms.AutoAugment,
        legacy_transforms.AutoAugment,
    ),
    ConsistencyConfig(
        v2_transforms.RandAugment,
        legacy_transforms.RandAugment,
    ),
    ConsistencyConfig(
        v2_transforms.TrivialAugmentWide,
        legacy_transforms.TrivialAugmentWide,
    ),
]


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

    legacy_signature = list(legacy_params.keys())
    # Since we made sure that we don't have any extra parameters without default above, we clamp the prototype signature
    # to the same number of parameters as the legacy one
    prototype_signature = list(prototype_params.keys())[: len(legacy_signature)]

    assert prototype_signature == legacy_signature


def check_call_consistency(
    prototype_transform, legacy_transform, images=None, supports_pil=True, closeness_kwargs=None
):
    if images is None:
        images = make_images(**DEFAULT_MAKE_IMAGES_KWARGS)

    closeness_kwargs = closeness_kwargs or dict()

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
                f"`is_pure_tensor` path in `_transform`."
            ) from exc

        assert_close(
            output_prototype_tensor,
            output_legacy_tensor,
            msg=lambda msg: f"Tensor image consistency check failed with: \n\n{msg}",
            **closeness_kwargs,
        )

        try:
            torch.manual_seed(0)
            output_prototype_image = prototype_transform(image)
        except Exception as exc:
            raise AssertionError(
                f"Transforming a image tv_tensor with shape {image_repr} failed in the prototype transform with "
                f"the error above. This means there is a consistency bug either in `_get_params` or in the "
                f"`tv_tensors.Image` path in `_transform`."
            ) from exc

        assert_close(
            output_prototype_image,
            output_prototype_tensor,
            msg=lambda msg: f"Output for tv_tensor and tensor images is not equal: \n\n{msg}",
            **closeness_kwargs,
        )

        if image.ndim == 3 and supports_pil:
            image_pil = to_pil_image(image)

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

            assert_close(
                output_prototype_pil,
                output_legacy_pil,
                msg=lambda msg: f"PIL image consistency check failed with: \n\n{msg}",
                **closeness_kwargs,
            )


@pytest.mark.parametrize(
    ("config", "args_kwargs"),
    [
        pytest.param(
            config, args_kwargs, id=f"{config.legacy_cls.__name__}-{idx:0{len(str(len(config.args_kwargs)))}d}"
        )
        for config in CONSISTENCY_CONFIGS
        for idx, args_kwargs in enumerate(config.args_kwargs)
    ],
)
@pytest.mark.filterwarnings("ignore")
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
        closeness_kwargs=config.closeness_kwargs,
    )


get_params_parametrization = pytest.mark.parametrize(
    ("config", "get_params_args_kwargs"),
    [
        pytest.param(
            next(config for config in CONSISTENCY_CONFIGS if config.prototype_cls is transform_cls),
            get_params_args_kwargs,
            id=transform_cls.__name__,
        )
        for transform_cls, get_params_args_kwargs in [
            (v2_transforms.RandomResizedCrop, ArgsKwargs(make_image(), scale=[0.3, 0.7], ratio=[0.5, 1.5])),
            (v2_transforms.RandomErasing, ArgsKwargs(make_image(), scale=(0.3, 0.7), ratio=(0.5, 1.5))),
            (v2_transforms.ColorJitter, ArgsKwargs(brightness=None, contrast=None, saturation=None, hue=None)),
            (v2_transforms.GaussianBlur, ArgsKwargs(0.3, 1.4)),
            (v2_transforms.RandomCrop, ArgsKwargs(make_image(size=(61, 47)), output_size=(19, 25))),
            (v2_transforms.RandomPerspective, ArgsKwargs(23, 17, 0.5)),
            (v2_transforms.AutoAugment, ArgsKwargs(5)),
        ]
    ],
)


@get_params_parametrization
def test_get_params_alias(config, get_params_args_kwargs):
    assert config.prototype_cls.get_params is config.legacy_cls.get_params

    if not config.args_kwargs:
        return
    args, kwargs = config.args_kwargs[0]
    legacy_transform = config.legacy_cls(*args, **kwargs)
    prototype_transform = config.prototype_cls(*args, **kwargs)

    assert prototype_transform.get_params is legacy_transform.get_params


@get_params_parametrization
def test_get_params_jit(config, get_params_args_kwargs):
    get_params_args, get_params_kwargs = get_params_args_kwargs

    torch.jit.script(config.prototype_cls.get_params)(*get_params_args, **get_params_kwargs)

    if not config.args_kwargs:
        return
    args, kwargs = config.args_kwargs[0]
    transform = config.prototype_cls(*args, **kwargs)

    torch.jit.script(transform.get_params)(*get_params_args, **get_params_kwargs)


@pytest.mark.parametrize(
    ("config", "args_kwargs"),
    [
        pytest.param(
            config, args_kwargs, id=f"{config.legacy_cls.__name__}-{idx:0{len(str(len(config.args_kwargs)))}d}"
        )
        for config in CONSISTENCY_CONFIGS
        for idx, args_kwargs in enumerate(config.args_kwargs)
        if not isinstance(args_kwargs, NotScriptableArgsKwargs)
    ],
)
def test_jit_consistency(config, args_kwargs):
    args, kwargs = args_kwargs

    prototype_transform_eager = config.prototype_cls(*args, **kwargs)
    legacy_transform_eager = config.legacy_cls(*args, **kwargs)

    legacy_transform_scripted = torch.jit.script(legacy_transform_eager)
    prototype_transform_scripted = torch.jit.script(prototype_transform_eager)

    for image in make_images(**config.make_images_kwargs):
        image = image.as_subclass(torch.Tensor)

        torch.manual_seed(0)
        output_legacy_scripted = legacy_transform_scripted(image)

        torch.manual_seed(0)
        output_prototype_scripted = prototype_transform_scripted(image)

        assert_close(output_prototype_scripted, output_legacy_scripted, **config.closeness_kwargs)


class TestContainerTransforms:
    """
    Since we are testing containers here, we also need some transforms to wrap. Thus, testing a container transform for
    consistency automatically tests the wrapped transforms consistency.

    Instead of complicated mocking or creating custom transforms just for these tests, here we use deterministic ones
    that were already tested for consistency above.
    """

    def test_compose(self):
        prototype_transform = v2_transforms.Compose(
            [
                v2_transforms.Resize(256),
                v2_transforms.CenterCrop(224),
            ]
        )
        legacy_transform = legacy_transforms.Compose(
            [
                legacy_transforms.Resize(256),
                legacy_transforms.CenterCrop(224),
            ]
        )

        # atol=1 due to Resize v2 is using native uint8 interpolate path for bilinear and nearest modes
        check_call_consistency(prototype_transform, legacy_transform, closeness_kwargs=dict(rtol=0, atol=1))

    @pytest.mark.parametrize("p", [0, 0.1, 0.5, 0.9, 1])
    @pytest.mark.parametrize("sequence_type", [list, nn.ModuleList])
    def test_random_apply(self, p, sequence_type):
        prototype_transform = v2_transforms.RandomApply(
            sequence_type(
                [
                    v2_transforms.Resize(256),
                    v2_transforms.CenterCrop(224),
                ]
            ),
            p=p,
        )
        legacy_transform = legacy_transforms.RandomApply(
            sequence_type(
                [
                    legacy_transforms.Resize(256),
                    legacy_transforms.CenterCrop(224),
                ]
            ),
            p=p,
        )

        # atol=1 due to Resize v2 is using native uint8 interpolate path for bilinear and nearest modes
        check_call_consistency(prototype_transform, legacy_transform, closeness_kwargs=dict(rtol=0, atol=1))

        if sequence_type is nn.ModuleList:
            # quick and dirty test that it is jit-scriptable
            scripted = torch.jit.script(prototype_transform)
            scripted(torch.rand(1, 3, 300, 300))

    # We can't test other values for `p` since the random parameter generation is different
    @pytest.mark.parametrize("probabilities", [(0, 1), (1, 0)])
    def test_random_choice(self, probabilities):
        prototype_transform = v2_transforms.RandomChoice(
            [
                v2_transforms.Resize(256),
                legacy_transforms.CenterCrop(224),
            ],
            p=probabilities,
        )
        legacy_transform = legacy_transforms.RandomChoice(
            [
                legacy_transforms.Resize(256),
                legacy_transforms.CenterCrop(224),
            ],
            p=probabilities,
        )

        # atol=1 due to Resize v2 is using native uint8 interpolate path for bilinear and nearest modes
        check_call_consistency(prototype_transform, legacy_transform, closeness_kwargs=dict(rtol=0, atol=1))


class TestToTensorTransforms:
    def test_pil_to_tensor(self):
        prototype_transform = v2_transforms.PILToTensor()
        legacy_transform = legacy_transforms.PILToTensor()

        for image in make_images(extra_dims=[()]):
            image_pil = to_pil_image(image)

            assert_equal(prototype_transform(image_pil), legacy_transform(image_pil))

    def test_to_tensor(self):
        with pytest.warns(UserWarning, match=re.escape("The transform `ToTensor()` is deprecated")):
            prototype_transform = v2_transforms.ToTensor()
        legacy_transform = legacy_transforms.ToTensor()

        for image in make_images(extra_dims=[()]):
            image_pil = to_pil_image(image)
            image_numpy = np.array(image_pil)

            assert_equal(prototype_transform(image_pil), legacy_transform(image_pil))
            assert_equal(prototype_transform(image_numpy), legacy_transform(image_numpy))


class TestAATransforms:
    @pytest.mark.parametrize(
        "inpt",
        [
            torch.randint(0, 256, size=(1, 3, 256, 256), dtype=torch.uint8),
            PIL.Image.new("RGB", (256, 256), 123),
            tv_tensors.Image(torch.randint(0, 256, size=(1, 3, 256, 256), dtype=torch.uint8)),
        ],
    )
    @pytest.mark.parametrize(
        "interpolation",
        [
            v2_transforms.InterpolationMode.NEAREST,
            v2_transforms.InterpolationMode.BILINEAR,
            PIL.Image.NEAREST,
        ],
    )
    def test_randaug(self, inpt, interpolation, mocker):
        t_ref = legacy_transforms.RandAugment(interpolation=interpolation, num_ops=1)
        t = v2_transforms.RandAugment(interpolation=interpolation, num_ops=1)

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

            assert_close(expected_output, output, atol=1, rtol=0.1)

    @pytest.mark.parametrize(
        "interpolation",
        [
            v2_transforms.InterpolationMode.NEAREST,
            v2_transforms.InterpolationMode.BILINEAR,
        ],
    )
    def test_randaug_jit(self, interpolation):
        inpt = torch.randint(0, 256, size=(1, 3, 256, 256), dtype=torch.uint8)
        t_ref = legacy_transforms.RandAugment(interpolation=interpolation, num_ops=1)
        t = v2_transforms.RandAugment(interpolation=interpolation, num_ops=1)

        tt_ref = torch.jit.script(t_ref)
        tt = torch.jit.script(t)

        torch.manual_seed(12)
        expected_output = tt_ref(inpt)

        torch.manual_seed(12)
        scripted_output = tt(inpt)

        assert_equal(scripted_output, expected_output)

    @pytest.mark.parametrize(
        "inpt",
        [
            torch.randint(0, 256, size=(1, 3, 256, 256), dtype=torch.uint8),
            PIL.Image.new("RGB", (256, 256), 123),
            tv_tensors.Image(torch.randint(0, 256, size=(1, 3, 256, 256), dtype=torch.uint8)),
        ],
    )
    @pytest.mark.parametrize(
        "interpolation",
        [
            v2_transforms.InterpolationMode.NEAREST,
            v2_transforms.InterpolationMode.BILINEAR,
            PIL.Image.NEAREST,
        ],
    )
    def test_trivial_aug(self, inpt, interpolation, mocker):
        t_ref = legacy_transforms.TrivialAugmentWide(interpolation=interpolation)
        t = v2_transforms.TrivialAugmentWide(interpolation=interpolation)

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

            assert_close(expected_output, output, atol=1, rtol=0.1)

    @pytest.mark.parametrize(
        "interpolation",
        [
            v2_transforms.InterpolationMode.NEAREST,
            v2_transforms.InterpolationMode.BILINEAR,
        ],
    )
    def test_trivial_aug_jit(self, interpolation):
        inpt = torch.randint(0, 256, size=(1, 3, 256, 256), dtype=torch.uint8)
        t_ref = legacy_transforms.TrivialAugmentWide(interpolation=interpolation)
        t = v2_transforms.TrivialAugmentWide(interpolation=interpolation)

        tt_ref = torch.jit.script(t_ref)
        tt = torch.jit.script(t)

        torch.manual_seed(12)
        expected_output = tt_ref(inpt)

        torch.manual_seed(12)
        scripted_output = tt(inpt)

        assert_equal(scripted_output, expected_output)

    @pytest.mark.parametrize(
        "inpt",
        [
            torch.randint(0, 256, size=(1, 3, 256, 256), dtype=torch.uint8),
            PIL.Image.new("RGB", (256, 256), 123),
            tv_tensors.Image(torch.randint(0, 256, size=(1, 3, 256, 256), dtype=torch.uint8)),
        ],
    )
    @pytest.mark.parametrize(
        "interpolation",
        [
            v2_transforms.InterpolationMode.NEAREST,
            v2_transforms.InterpolationMode.BILINEAR,
            PIL.Image.NEAREST,
        ],
    )
    def test_augmix(self, inpt, interpolation, mocker):
        t_ref = legacy_transforms.AugMix(interpolation=interpolation, mixture_width=1, chain_depth=1)
        t_ref._sample_dirichlet = lambda t: t.softmax(dim=-1)
        t = v2_transforms.AugMix(interpolation=interpolation, mixture_width=1, chain_depth=1)
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
        "interpolation",
        [
            v2_transforms.InterpolationMode.NEAREST,
            v2_transforms.InterpolationMode.BILINEAR,
        ],
    )
    def test_augmix_jit(self, interpolation):
        inpt = torch.randint(0, 256, size=(1, 3, 256, 256), dtype=torch.uint8)

        t_ref = legacy_transforms.AugMix(interpolation=interpolation, mixture_width=1, chain_depth=1)
        t = v2_transforms.AugMix(interpolation=interpolation, mixture_width=1, chain_depth=1)

        tt_ref = torch.jit.script(t_ref)
        tt = torch.jit.script(t)

        torch.manual_seed(12)
        expected_output = tt_ref(inpt)

        torch.manual_seed(12)
        scripted_output = tt(inpt)

        assert_equal(scripted_output, expected_output)

    @pytest.mark.parametrize(
        "inpt",
        [
            torch.randint(0, 256, size=(1, 3, 256, 256), dtype=torch.uint8),
            PIL.Image.new("RGB", (256, 256), 123),
            tv_tensors.Image(torch.randint(0, 256, size=(1, 3, 256, 256), dtype=torch.uint8)),
        ],
    )
    @pytest.mark.parametrize(
        "interpolation",
        [
            v2_transforms.InterpolationMode.NEAREST,
            v2_transforms.InterpolationMode.BILINEAR,
            PIL.Image.NEAREST,
        ],
    )
    def test_aa(self, inpt, interpolation):
        aa_policy = legacy_transforms.AutoAugmentPolicy("imagenet")
        t_ref = legacy_transforms.AutoAugment(aa_policy, interpolation=interpolation)
        t = v2_transforms.AutoAugment(aa_policy, interpolation=interpolation)

        torch.manual_seed(12)
        expected_output = t_ref(inpt)

        torch.manual_seed(12)
        output = t(inpt)

        assert_equal(expected_output, output)

    @pytest.mark.parametrize(
        "interpolation",
        [
            v2_transforms.InterpolationMode.NEAREST,
            v2_transforms.InterpolationMode.BILINEAR,
        ],
    )
    def test_aa_jit(self, interpolation):
        inpt = torch.randint(0, 256, size=(1, 3, 256, 256), dtype=torch.uint8)
        aa_policy = legacy_transforms.AutoAugmentPolicy("imagenet")
        t_ref = legacy_transforms.AutoAugment(aa_policy, interpolation=interpolation)
        t = v2_transforms.AutoAugment(aa_policy, interpolation=interpolation)

        tt_ref = torch.jit.script(t_ref)
        tt = torch.jit.script(t)

        torch.manual_seed(12)
        expected_output = tt_ref(inpt)

        torch.manual_seed(12)
        scripted_output = tt(inpt)

        assert_equal(scripted_output, expected_output)


def import_transforms_from_references(reference):
    HERE = Path(__file__).parent
    PROJECT_ROOT = HERE.parent

    loader = importlib.machinery.SourceFileLoader(
        "transforms", str(PROJECT_ROOT / "references" / reference / "transforms.py")
    )
    spec = importlib.util.spec_from_loader("transforms", loader)
    module = importlib.util.module_from_spec(spec)
    loader.exec_module(module)
    return module


det_transforms = import_transforms_from_references("detection")


class TestRefDetTransforms:
    def make_tv_tensors(self, with_mask=True):
        size = (600, 800)
        num_objects = 22

        def make_label(extra_dims, categories):
            return torch.randint(categories, extra_dims, dtype=torch.int64)

        pil_image = to_pil_image(make_image(size=size, color_space="RGB"))
        target = {
            "boxes": make_bounding_boxes(canvas_size=size, format="XYXY", batch_dims=(num_objects,), dtype=torch.float),
            "labels": make_label(extra_dims=(num_objects,), categories=80),
        }
        if with_mask:
            target["masks"] = make_detection_mask(size=size, num_objects=num_objects, dtype=torch.long)

        yield (pil_image, target)

        tensor_image = torch.Tensor(make_image(size=size, color_space="RGB", dtype=torch.float32))
        target = {
            "boxes": make_bounding_boxes(canvas_size=size, format="XYXY", batch_dims=(num_objects,), dtype=torch.float),
            "labels": make_label(extra_dims=(num_objects,), categories=80),
        }
        if with_mask:
            target["masks"] = make_detection_mask(size=size, num_objects=num_objects, dtype=torch.long)

        yield (tensor_image, target)

        tv_tensor_image = make_image(size=size, color_space="RGB", dtype=torch.float32)
        target = {
            "boxes": make_bounding_boxes(canvas_size=size, format="XYXY", batch_dims=(num_objects,), dtype=torch.float),
            "labels": make_label(extra_dims=(num_objects,), categories=80),
        }
        if with_mask:
            target["masks"] = make_detection_mask(size=size, num_objects=num_objects, dtype=torch.long)

        yield (tv_tensor_image, target)

    @pytest.mark.parametrize(
        "t_ref, t, data_kwargs",
        [
            (det_transforms.RandomHorizontalFlip(p=1.0), v2_transforms.RandomHorizontalFlip(p=1.0), {}),
            (
                det_transforms.RandomIoUCrop(),
                v2_transforms.Compose(
                    [
                        v2_transforms.RandomIoUCrop(),
                        v2_transforms.SanitizeBoundingBoxes(labels_getter=lambda sample: sample[1]["labels"]),
                    ]
                ),
                {"with_mask": False},
            ),
            (det_transforms.RandomZoomOut(), v2_transforms.RandomZoomOut(), {"with_mask": False}),
            (det_transforms.ScaleJitter((1024, 1024)), v2_transforms.ScaleJitter((1024, 1024), antialias=True), {}),
            (
                det_transforms.RandomShortestSize(
                    min_size=(480, 512, 544, 576, 608, 640, 672, 704, 736, 768, 800), max_size=1333
                ),
                v2_transforms.RandomShortestSize(
                    min_size=(480, 512, 544, 576, 608, 640, 672, 704, 736, 768, 800), max_size=1333
                ),
                {},
            ),
        ],
    )
    def test_transform(self, t_ref, t, data_kwargs):
        for dp in self.make_tv_tensors(**data_kwargs):

            # We should use prototype transform first as reference transform performs inplace target update
            torch.manual_seed(12)
            output = t(dp)

            torch.manual_seed(12)
            expected_output = t_ref(*dp)

            assert_equal(expected_output, output)


seg_transforms = import_transforms_from_references("segmentation")


# We need this transform for two reasons:
# 1. transforms.RandomCrop uses a different scheme to pad images and masks of insufficient size than its name
#    counterpart in the detection references. Thus, we cannot use it with `pad_if_needed=True`
# 2. transforms.Pad only supports a fixed padding, but the segmentation datasets don't have a fixed image size.
class PadIfSmaller(v2_transforms.Transform):
    def __init__(self, size, fill=0):
        super().__init__()
        self.size = size
        self.fill = v2_transforms._geometry._setup_fill_arg(fill)

    def _get_params(self, sample):
        height, width = query_size(sample)
        padding = [0, 0, max(self.size - width, 0), max(self.size - height, 0)]
        needs_padding = any(padding)
        return dict(padding=padding, needs_padding=needs_padding)

    def _transform(self, inpt, params):
        if not params["needs_padding"]:
            return inpt

        fill = _get_fill(self.fill, type(inpt))
        return prototype_F.pad(inpt, padding=params["padding"], fill=fill)


class TestRefSegTransforms:
    def make_tv_tensors(self, supports_pil=True, image_dtype=torch.uint8):
        size = (256, 460)
        num_categories = 21

        conv_fns = []
        if supports_pil:
            conv_fns.append(to_pil_image)
        conv_fns.extend([torch.Tensor, lambda x: x])

        for conv_fn in conv_fns:
            tv_tensor_image = make_image(size=size, color_space="RGB", dtype=image_dtype)
            tv_tensor_mask = make_segmentation_mask(size=size, num_categories=num_categories, dtype=torch.uint8)

            dp = (conv_fn(tv_tensor_image), tv_tensor_mask)
            dp_ref = (
                to_pil_image(tv_tensor_image) if supports_pil else tv_tensor_image.as_subclass(torch.Tensor),
                to_pil_image(tv_tensor_mask),
            )

            yield dp, dp_ref

    def set_seed(self, seed=12):
        torch.manual_seed(seed)
        random.seed(seed)

    def check(self, t, t_ref, data_kwargs=None):
        for dp, dp_ref in self.make_tv_tensors(**data_kwargs or dict()):

            self.set_seed()
            actual = actual_image, actual_mask = t(dp)

            self.set_seed()
            expected_image, expected_mask = t_ref(*dp_ref)
            if isinstance(actual_image, torch.Tensor) and not isinstance(expected_image, torch.Tensor):
                expected_image = legacy_F.pil_to_tensor(expected_image)
            expected_mask = legacy_F.pil_to_tensor(expected_mask).squeeze(0)
            expected = (expected_image, expected_mask)

            assert_equal(actual, expected)

    @pytest.mark.parametrize(
        ("t_ref", "t", "data_kwargs"),
        [
            (
                seg_transforms.RandomHorizontalFlip(flip_prob=1.0),
                v2_transforms.RandomHorizontalFlip(p=1.0),
                dict(),
            ),
            (
                seg_transforms.RandomHorizontalFlip(flip_prob=0.0),
                v2_transforms.RandomHorizontalFlip(p=0.0),
                dict(),
            ),
            (
                seg_transforms.RandomCrop(size=480),
                v2_transforms.Compose(
                    [
                        PadIfSmaller(size=480, fill={tv_tensors.Mask: 255, "others": 0}),
                        v2_transforms.RandomCrop(size=480),
                    ]
                ),
                dict(),
            ),
            (
                seg_transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
                v2_transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
                dict(supports_pil=False, image_dtype=torch.float),
            ),
        ],
    )
    def test_common(self, t_ref, t, data_kwargs):
        self.check(t, t_ref, data_kwargs)


@pytest.mark.parametrize(
    ("legacy_dispatcher", "name_only_params"),
    [
        (legacy_F.get_dimensions, {}),
        (legacy_F.get_image_size, {}),
        (legacy_F.get_image_num_channels, {}),
        (legacy_F.to_tensor, {}),
        (legacy_F.pil_to_tensor, {}),
        (legacy_F.convert_image_dtype, {}),
        (legacy_F.to_pil_image, {}),
        (legacy_F.normalize, {}),
        (legacy_F.resize, {"interpolation"}),
        (legacy_F.pad, {"padding", "fill"}),
        (legacy_F.crop, {}),
        (legacy_F.center_crop, {}),
        (legacy_F.resized_crop, {"interpolation"}),
        (legacy_F.hflip, {}),
        (legacy_F.perspective, {"startpoints", "endpoints", "fill", "interpolation"}),
        (legacy_F.vflip, {}),
        (legacy_F.five_crop, {}),
        (legacy_F.ten_crop, {}),
        (legacy_F.adjust_brightness, {}),
        (legacy_F.adjust_contrast, {}),
        (legacy_F.adjust_saturation, {}),
        (legacy_F.adjust_hue, {}),
        (legacy_F.adjust_gamma, {}),
        (legacy_F.rotate, {"center", "fill", "interpolation"}),
        (legacy_F.affine, {"angle", "translate", "center", "fill", "interpolation"}),
        (legacy_F.to_grayscale, {}),
        (legacy_F.rgb_to_grayscale, {}),
        (legacy_F.to_tensor, {}),
        (legacy_F.erase, {}),
        (legacy_F.gaussian_blur, {}),
        (legacy_F.invert, {}),
        (legacy_F.posterize, {}),
        (legacy_F.solarize, {}),
        (legacy_F.adjust_sharpness, {}),
        (legacy_F.autocontrast, {}),
        (legacy_F.equalize, {}),
        (legacy_F.elastic_transform, {"fill", "interpolation"}),
    ],
)
def test_dispatcher_signature_consistency(legacy_dispatcher, name_only_params):
    legacy_signature = inspect.signature(legacy_dispatcher)
    legacy_params = list(legacy_signature.parameters.values())[1:]

    try:
        prototype_dispatcher = getattr(prototype_F, legacy_dispatcher.__name__)
    except AttributeError:
        raise AssertionError(
            f"Legacy dispatcher `F.{legacy_dispatcher.__name__}` has no prototype equivalent"
        ) from None

    prototype_signature = inspect.signature(prototype_dispatcher)
    prototype_params = list(prototype_signature.parameters.values())[1:]

    # Some dispatchers got extra parameters. This makes sure they have a default argument and thus are BC. We don't
    # need to check if parameters were added in the middle rather than at the end, since that will be caught by the
    # regular check below.
    prototype_params, new_prototype_params = (
        prototype_params[: len(legacy_params)],
        prototype_params[len(legacy_params) :],
    )
    for param in new_prototype_params:
        assert param.default is not param.empty

    # Some annotations were changed mostly to supersets of what was there before. Plus, some legacy dispatchers had no
    # annotations. In these cases we simply drop the annotation and default argument from the comparison
    for prototype_param, legacy_param in zip(prototype_params, legacy_params):
        if legacy_param.name in name_only_params:
            prototype_param._annotation = prototype_param._default = inspect.Parameter.empty
            legacy_param._annotation = legacy_param._default = inspect.Parameter.empty
        elif legacy_param.annotation is inspect.Parameter.empty:
            prototype_param._annotation = inspect.Parameter.empty

    assert prototype_params == legacy_params
