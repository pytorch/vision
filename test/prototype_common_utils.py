"""This module is separated from common_utils.py to prevent the former to be dependent on torchvision.prototype"""

import collections.abc
import functools
import itertools

import PIL.Image
import pytest
import torch
import torch.testing
from torch.nn.functional import one_hot
from torch.testing._comparison import (
    assert_equal as _assert_equal,
    BooleanPair,
    NonePair,
    NumberPair,
    TensorLikePair,
    UnsupportedInputs,
)
from torchvision.prototype import features
from torchvision.transforms.functional_tensor import _max_value as get_max_value

__all__ = ["assert_close"]


# class ImagePair(TensorLikePair):
#     def _process_inputs(self, actual, expected, *, id, allow_subclasses):
#         return super()._process_inputs(
#             *[to_image_tensor(input) if isinstance(input, PIL.Image.Image) else input for input in [actual, expected]],
#             id=id,
#             allow_subclasses=allow_subclasses,
#         )


class PILImagePair(TensorLikePair):
    def __init__(
        self,
        actual,
        expected,
        *,
        agg_method=None,
        allowed_percentage_diff=None,
        **other_parameters,
    ):
        if not any(isinstance(input, PIL.Image.Image) for input in (actual, expected)):
            raise UnsupportedInputs()

        actual, expected = [
            features.Image(input) if isinstance(input, PIL.Image.Image) else input for input in (actual, expected)
        ]

        super().__init__(actual, expected, **other_parameters)
        self.agg_method = getattr(torch, agg_method) if isinstance(agg_method, str) else agg_method
        self.allowed_percentage_diff = allowed_percentage_diff

        # TODO: comment
        self.check_dtype = False
        self.check_device = False

    def _equalize_attributes(self, actual, expected):
        actual, expected = [input.to(torch.float64).div_(get_max_value(input.dtype)) for input in [actual, expected]]
        return super()._equalize_attributes(actual, expected)

    def compare(self) -> None:
        actual, expected = self.actual, self.expected

        self._compare_attributes(actual, expected)
        if all(isinstance(input, features.Image) for input in (actual, expected)):
            if actual.color_space != expected.color_space:
                self._make_error_meta(AssertionError, "color space mismatch")

        actual, expected = self._equalize_attributes(actual, expected)
        abs_diff = torch.abs(actual - expected)

        if self.allowed_percentage_diff is not None:
            percentage_diff = (abs_diff != 0).to(torch.float).mean()
            if percentage_diff > self.allowed_percentage_diff:
                self._make_error_meta(AssertionError, "percentage mismatch")

        if self.agg_method is None:
            super()._compare_values(actual, expected)
        else:
            err = self.agg_method(abs_diff)
            if err > self.atol:
                self._make_error_meta(AssertionError, "aggregated mismatch")


def assert_close(
    actual,
    expected,
    *,
    allow_subclasses=True,
    rtol=None,
    atol=None,
    equal_nan=False,
    check_device=True,
    check_dtype=True,
    check_layout=True,
    check_stride=False,
    msg=None,
    **kwargs,
):
    """Superset of :func:`torch.testing.assert_close` with support for PIL vs. tensor image comparison"""
    __tracebackhide__ = True

    _assert_equal(
        actual,
        expected,
        pair_types=(
            NonePair,
            BooleanPair,
            NumberPair,
            PILImagePair,
            TensorLikePair,
        ),
        allow_subclasses=allow_subclasses,
        rtol=rtol,
        atol=atol,
        equal_nan=equal_nan,
        check_device=check_device,
        check_dtype=check_dtype,
        check_layout=check_layout,
        check_stride=check_stride,
        msg=msg,
        **kwargs,
    )


assert_equal = functools.partial(assert_close, rtol=0, atol=0)


class ArgsKwargs:
    def __init__(self, *args, **kwargs):
        self.args = args
        self.kwargs = kwargs

    def __iter__(self):
        yield self.args
        yield self.kwargs

    def __str__(self):
        def short_repr(obj, max=20):
            repr_ = repr(obj)
            if len(repr_) <= max:
                return repr_

            return f"{repr_[:max//2]}...{repr_[-(max//2-3):]}"

        return ", ".join(
            itertools.chain(
                [short_repr(arg) for arg in self.args],
                [f"{param}={short_repr(kwarg)}" for param, kwarg in self.kwargs.items()],
            )
        )


DEFAULT_SQUARE_IMAGE_SIZE = (16, 16)
DEFAULT_LANDSCAPE_IMAGE_SIZE = (7, 33)
DEFAULT_PORTRAIT_IMAGE_SIZE = (31, 9)
DEFAULT_IMAGE_SIZES = (DEFAULT_LANDSCAPE_IMAGE_SIZE, DEFAULT_PORTRAIT_IMAGE_SIZE, DEFAULT_SQUARE_IMAGE_SIZE)

DEFAULT_EXTRA_DIMS = ((), (0,), (4,), (2, 3), (5, 0), (0, 5))


def make_image(
    size=None,
    *,
    color_space=features.ColorSpace.RGB,
    extra_dims=(),
    device="cpu",
    dtype=torch.float32,
    constant_alpha=True,
):
    size = size or torch.randint(16, 33, (2,)).tolist()

    try:
        num_channels = {
            features.ColorSpace.GRAY: 1,
            features.ColorSpace.GRAY_ALPHA: 2,
            features.ColorSpace.RGB: 3,
            features.ColorSpace.RGB_ALPHA: 4,
        }[color_space]
    except KeyError as error:
        raise pytest.UsageError(f"Can't determine the number of channels for color space {color_space}") from error

    max_value = get_max_value(dtype)
    data = torch.testing.make_tensor(
        *extra_dims, num_channels, *size, low=0, high=max_value, dtype=dtype, device=device
    )
    if color_space in {features.ColorSpace.GRAY_ALPHA, features.ColorSpace.RGB_ALPHA} and constant_alpha:
        data[..., -1, :, :] = max_value
    return features.Image(data, color_space=color_space)


def make_images(
    *,
    sizes=DEFAULT_IMAGE_SIZES,
    color_spaces=(
        features.ColorSpace.GRAY,
        features.ColorSpace.GRAY_ALPHA,
        features.ColorSpace.RGB,
        features.ColorSpace.RGB_ALPHA,
    ),
    extra_dims=DEFAULT_EXTRA_DIMS,
    device="cpu",
    dtypes=(torch.float32, torch.uint8),
    constant_alpha=True,
):
    for size, color_space, dtype in itertools.product(sizes, color_spaces, dtypes):
        yield make_image(size, color_space=color_space, device=device, dtype=dtype, constant_alpha=constant_alpha)

    for color_space, dtype, extra_dims_ in itertools.product(color_spaces, dtypes, extra_dims):
        yield make_image(
            size=sizes[0],
            color_space=color_space,
            extra_dims=extra_dims_,
            device=device,
            dtype=dtype,
            constant_alpha=constant_alpha,
        )


def randint_with_tensor_bounds(arg1, arg2=None, **kwargs):
    low, high = torch.broadcast_tensors(
        *[torch.as_tensor(arg) for arg in ((0, arg1) if arg2 is None else (arg1, arg2))]
    )
    return torch.stack(
        [
            torch.randint(low_scalar, high_scalar, (), **kwargs)
            for low_scalar, high_scalar in zip(low.flatten().tolist(), high.flatten().tolist())
        ]
    ).reshape(low.shape)


def make_bounding_box(
    *, extra_dims=(), format, image_size=DEFAULT_LANDSCAPE_IMAGE_SIZE, device="cpu", dtype=torch.float32
):
    if isinstance(format, str):
        format = features.BoundingBoxFormat[format]

    if any(dim == 0 for dim in extra_dims):
        return features.BoundingBox(torch.empty(*extra_dims, 4), format=format, image_size=image_size)

    height, width = image_size

    if format == features.BoundingBoxFormat.XYXY:
        x1 = torch.randint(0, width // 2, extra_dims)
        y1 = torch.randint(0, height // 2, extra_dims)
        x2 = randint_with_tensor_bounds(x1 + 1, width - x1) + x1
        y2 = randint_with_tensor_bounds(y1 + 1, height - y1) + y1
        parts = (x1, y1, x2, y2)
    elif format == features.BoundingBoxFormat.XYWH:
        x = torch.randint(0, width // 2, extra_dims)
        y = torch.randint(0, height // 2, extra_dims)
        w = randint_with_tensor_bounds(1, width - x)
        h = randint_with_tensor_bounds(1, height - y)
        parts = (x, y, w, h)
    elif format == features.BoundingBoxFormat.CXCYWH:
        cx = torch.randint(1, width - 1, ())
        cy = torch.randint(1, height - 1, ())
        w = randint_with_tensor_bounds(1, torch.minimum(cx, width - cx) + 1)
        h = randint_with_tensor_bounds(1, torch.minimum(cy, height - cy) + 1)
        parts = (cx, cy, w, h)
    else:
        raise pytest.UsageError(f"Can't make bounding box in format {format}")

    return features.BoundingBox(
        torch.stack(parts, dim=-1).to(dtype=dtype, device=device), format=format, image_size=image_size
    )


def make_bounding_boxes(
    *,
    extra_dims=DEFAULT_EXTRA_DIMS,
    formats=(features.BoundingBoxFormat.XYXY, features.BoundingBoxFormat.XYWH, features.BoundingBoxFormat.CXCYWH),
    image_size=(32, 32),
    device="cpu",
    dtypes=(torch.float32, torch.int64),
):
    for extra_dims_, format in itertools.product(extra_dims, formats):
        yield make_bounding_box(extra_dims=extra_dims_, format=format, image_size=image_size, device=device)

    for format, dtype in itertools.product(formats, dtypes):
        yield make_bounding_box(format=format, image_size=image_size, device=device, dtype=dtype)


def make_label(*, extra_dims=(), categories=None, device="cpu", dtype=torch.int64):
    if categories is None:
        categories = int(torch.randint(1, 11, ()))
    if isinstance(categories, int):
        num_categories = categories
        categories = [f"category{idx}" for idx in range(num_categories)]
    elif isinstance(categories, collections.abc.Sequence) and all(isinstance(category, str) for category in categories):
        num_categories = len(categories)
    else:
        raise pytest.UsageError("FIXME")

    # The idiom `make_tensor(..., dtype=torch.int64).to(dtype)` is intentional to only get integer values, regardless of
    # the requested dtype, e.g. 0 or 0.0 rather than 0 or 0.123
    data = torch.testing.make_tensor(extra_dims, low=0, high=num_categories, dtype=torch.int64, device=device).to(dtype)
    return features.Label(data, categories=categories)


def make_one_hot_label(*, categories=None, extra_dims=(), device="cpu", dtype=torch.int64):
    if categories == 0:
        data = torch.empty(*extra_dims, 0, dtype=dtype, device=device)
        categories = None
    else:
        # The idiom `make_label(..., dtype=torch.int64); ...; one_hot(...).to(dtype)` is intentional since `one_hot`
        # only supports int64
        label = make_label(extra_dims=extra_dims, categories=categories, device=device, dtype=torch.int64)
        categories = label.categories
        data = one_hot(label, num_classes=len(label.categories)).to(dtype)
    return features.OneHotLabel(data, categories=categories)


def make_one_hot_labels(
    *,
    categories=(1, 0, None),
    extra_dims=DEFAULT_EXTRA_DIMS,
    device="cpu",
    dtypes=(torch.int64, torch.float32),
):
    for categories_, extra_dims_ in itertools.product(categories, extra_dims):
        yield make_one_hot_label(categories=categories_, extra_dims=extra_dims_, device=device)

    for categories_, dtype in itertools.product(categories, dtypes):
        yield make_one_hot_label(categories=categories_, device=device, dtype=dtype)


def make_segmentation_mask(size=None, *, num_objects=None, extra_dims=(), device="cpu", dtype=torch.uint8):
    size = size if size is not None else torch.randint(16, 33, (2,)).tolist()
    num_objects = num_objects if num_objects is not None else int(torch.randint(1, 11, ()))
    data = torch.testing.make_tensor(*extra_dims, num_objects, *size, low=0, high=2, dtype=dtype, device=device)
    return features.SegmentationMask(data)


def make_segmentation_masks(
    sizes=DEFAULT_IMAGE_SIZES,
    num_objects=(1, 0, None),
    extra_dims=DEFAULT_EXTRA_DIMS,
    device="cpu",
    dtypes=(torch.uint8, torch.bool),
):
    for size, num_objects_, extra_dims_ in itertools.product(sizes, num_objects, extra_dims):
        yield make_segmentation_mask(size=size, num_objects=num_objects_, extra_dims=extra_dims_, device=device)

    for num_objects_, dtype in itertools.product(num_objects, dtypes):
        yield make_segmentation_mask(num_objects=num_objects_, device=device, dtype=dtype)
