"""This module is separated from common_utils.py to prevent the former to be dependent on torchvision.prototype"""

import collections.abc
import enum
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
from torchvision.prototype.transforms.functional import convert_image_dtype, to_image_tensor
from torchvision.transforms.functional_tensor import _max_value as get_max_value

__all__ = [
    "assert_close",
    "assert_equal",
    "ArgsKwargs",
    "make_image_loaders",
    "make_image",
    "make_images",
    "make_bounding_box_loaders",
    "make_bounding_box",
    "make_bounding_boxes",
    "make_segmentation_mask_loaders",
    "make_segmentation_mask",
    "make_segmentation_masks",
]


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

        # This parameter is ignored to enable checking PIL images to tensor images no on the CPU
        other_parameters["check_device"] = False

        super().__init__(actual, expected, **other_parameters)
        self.agg_method = getattr(torch, agg_method) if isinstance(agg_method, str) else agg_method
        self.allowed_percentage_diff = allowed_percentage_diff

    def _process_inputs(self, actual, expected, *, id, allow_subclasses):
        actual, expected = [
            to_image_tensor(input) if not isinstance(input, torch.Tensor) else input for input in [actual, expected]
        ]
        return super()._process_inputs(actual, expected, id=id, allow_subclasses=allow_subclasses)

    def _equalize_attributes(self, actual, expected):
        if actual.dtype != expected.dtype:
            dtype = torch.promote_types(actual.dtype, expected.dtype)
            actual = convert_image_dtype(actual, dtype)
            expected = convert_image_dtype(expected, dtype)

        return super()._equalize_attributes(actual, expected)

    def compare(self) -> None:
        actual, expected = self.actual, self.expected

        self._compare_attributes(actual, expected)

        actual, expected = self._equalize_attributes(actual, expected)
        abs_diff = torch.abs(actual - expected)

        if self.allowed_percentage_diff is not None:
            percentage_diff = (abs_diff != 0).to(torch.float).mean()
            if percentage_diff > self.allowed_percentage_diff:
                self._make_error_meta(AssertionError, "percentage mismatch")

        if self.agg_method is None:
            super()._compare_values(actual, expected)
        else:
            err = self.agg_method(abs_diff.to(torch.float64))
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

    def load(self, device="cpu"):
        args = tuple(arg.load(device) if isinstance(arg, TensorLoader) else arg for arg in self.args)
        kwargs = {
            keyword: arg.load(device) if isinstance(arg, TensorLoader) else arg for keyword, arg in self.kwargs.items()
        }
        return args, kwargs

    def __repr__(self):
        def better_repr(obj):
            if isinstance(obj, enum.Enum):
                return str(obj)
            else:
                return repr(obj)

        return ", ".join(
            itertools.chain(
                [better_repr(arg) for arg in self.args],
                [f"{param}={better_repr(kwarg)}" for param, kwarg in self.kwargs.items()],
            )
        )


DEFAULT_SQUARE_IMAGE_SIZE = (16, 16)
DEFAULT_LANDSCAPE_IMAGE_SIZE = (7, 33)
DEFAULT_PORTRAIT_IMAGE_SIZE = (31, 9)
DEFAULT_IMAGE_SIZES = (DEFAULT_LANDSCAPE_IMAGE_SIZE, DEFAULT_PORTRAIT_IMAGE_SIZE, DEFAULT_SQUARE_IMAGE_SIZE)

DEFAULT_EXTRA_DIMS = ((), (0,), (4,), (2, 3), (5, 0), (0, 5))


def from_loader(loader_fn):
    def wrapper(*args, **kwargs):
        loader = loader_fn(*args, **kwargs)
        return loader.load(kwargs.get("device", "cpu"))

    return wrapper


def from_loaders(loaders_fn):
    def wrapper(*args, **kwargs):
        loaders = loaders_fn(*args, **kwargs)
        for loader in loaders:
            yield loader.load(kwargs.get("device", "cpu"))

    return wrapper


class TensorLoader:
    def __init__(self, fn, *, shape, dtype):
        self.fn = fn
        self.shape = shape
        self.dtype = dtype

    def load(self, device):
        return self.fn(self.shape, self.dtype, device)

    _TYPE_NAME = "torch.Tensor"

    def _extra_repr(self):
        return []

    def __repr__(self):
        extra = ", ".join(
            [
                str(tuple(self.shape)),
                str(self.dtype).replace("torch.", ""),
                *[str(extra) for extra in self._extra_repr()],
            ]
        )
        return f"{self._TYPE_NAME}[{extra}]"


class ImageLoader(TensorLoader):
    def __init__(self, *args, color_space, **kwargs):
        super().__init__(*args, **kwargs)
        self.image_size = self.shape[-2:]
        self.num_channels = self.shape[-3]
        self.color_space = color_space

    _TYPE_NAME = "features.Image"

    def _extra_repr(self):
        return [self.color_space]


class SegmentationMaskLoader(TensorLoader):
    _TYPE_NAME = "features.SegmentationMask"


def make_image_loader(
    size=None,
    *,
    color_space=features.ColorSpace.RGB,
    extra_dims=(),
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

    def fn(shape, dtype, device):
        max_value = get_max_value(dtype)
        data = torch.testing.make_tensor(shape, low=0, high=max_value, dtype=dtype, device=device)
        if color_space in {features.ColorSpace.GRAY_ALPHA, features.ColorSpace.RGB_ALPHA} and constant_alpha:
            data[..., -1, :, :] = max_value
        return features.Image(data, color_space=color_space)

    return ImageLoader(fn, shape=(*extra_dims, num_channels, *size), dtype=dtype, color_space=color_space)


make_image = from_loader(make_image_loader)


def make_image_loaders(
    *,
    sizes=DEFAULT_IMAGE_SIZES,
    color_spaces=(
        features.ColorSpace.GRAY,
        features.ColorSpace.GRAY_ALPHA,
        features.ColorSpace.RGB,
        features.ColorSpace.RGB_ALPHA,
    ),
    extra_dims=DEFAULT_EXTRA_DIMS,
    dtypes=(torch.float32, torch.uint8),
    constant_alpha=True,
):
    for size, color_space, dtype in itertools.product(sizes, color_spaces, dtypes):
        yield make_image_loader(size, color_space=color_space, dtype=dtype, constant_alpha=constant_alpha)

    for color_space, dtype, extra_dims_ in itertools.product(color_spaces, dtypes, extra_dims):
        yield make_image_loader(
            size=sizes[0],
            color_space=color_space,
            extra_dims=extra_dims_,
            dtype=dtype,
            constant_alpha=constant_alpha,
        )


make_images = from_loaders(make_image_loaders)


class BoundingBoxLoader(TensorLoader):
    def __init__(self, *args, format, image_size, **kwargs):
        super().__init__(*args, **kwargs)
        self.format = format
        self.image_size = image_size

    _TYPE_NAME = "features.BoundingBox"

    def _extra_repr(self):
        return [self.format, f"image_size={self.image_size}"]


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


def make_bounding_box_loader(*, extra_dims=(), format, image_size=DEFAULT_LANDSCAPE_IMAGE_SIZE, dtype=torch.float32):
    if isinstance(format, str):
        format = features.BoundingBoxFormat[format]
    if format not in {
        features.BoundingBoxFormat.XYXY,
        features.BoundingBoxFormat.XYWH,
        features.BoundingBoxFormat.CXCYWH,
    }:
        raise pytest.UsageError(f"Can't make bounding box in format {format}")

    def fn(shape, dtype, device):
        *extra_dims, num_coordinates = shape
        if num_coordinates != 4:
            raise pytest.UsageError()

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
        else:  # format == features.BoundingBoxFormat.CXCYWH:
            cx = torch.randint(1, width - 1, ())
            cy = torch.randint(1, height - 1, ())
            w = randint_with_tensor_bounds(1, torch.minimum(cx, width - cx) + 1)
            h = randint_with_tensor_bounds(1, torch.minimum(cy, height - cy) + 1)
            parts = (cx, cy, w, h)

        return features.BoundingBox(torch.stack(parts, dim=-1).to(dtype=dtype), format=format, image_size=image_size)

    return BoundingBoxLoader(fn, shape=(*extra_dims, 4), dtype=dtype, format=format, image_size=image_size)


make_bounding_box = from_loader(make_bounding_box_loader)


def make_bounding_box_loaders(
    *,
    extra_dims=DEFAULT_EXTRA_DIMS,
    formats=(features.BoundingBoxFormat.XYXY, features.BoundingBoxFormat.XYWH, features.BoundingBoxFormat.CXCYWH),
    image_size=(32, 32),
    dtypes=(torch.float32, torch.int64),
):
    for extra_dims_, format in itertools.product(extra_dims, formats):
        yield make_bounding_box_loader(extra_dims=extra_dims_, format=format, image_size=image_size)

    for format, dtype in itertools.product(formats, dtypes):
        yield make_bounding_box_loader(format=format, image_size=image_size, dtype=dtype)


make_bounding_boxes = from_loaders(make_bounding_box_loaders)


def make_label(*, extra_dims=(), categories=None, device="cpu", dtype=torch.int64):
    if categories is None:
        categories = int(torch.randint(1, 11, ()))
    if isinstance(categories, int):
        num_categories = categories
        categories = [f"category{idx}" for idx in range(num_categories)]
    elif isinstance(categories, collections.abc.Sequence) and all(isinstance(category, str) for category in categories):
        num_categories = len(categories)
    else:
        raise pytest.UsageError(
            f"`categories` can either be `None` (default), an integer, or a sequence of strings, "
            f"but got '{categories}' instead"
        )

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


def make_segmentation_mask_loader(size=None, *, num_objects=None, extra_dims=(), dtype=torch.uint8):
    size = size if size is not None else torch.randint(16, 33, (2,)).tolist()
    num_objects = num_objects if num_objects is not None else int(torch.randint(1, 11, ()))

    def fn(shape, dtype, device):
        data = torch.testing.make_tensor(shape, low=0, high=2, dtype=dtype, device=device)
        return features.SegmentationMask(data)

    return SegmentationMaskLoader(fn, shape=(*extra_dims, num_objects, *size), dtype=dtype)


make_segmentation_mask = from_loader(make_segmentation_mask_loader)


def make_segmentation_mask_loaders(
    sizes=DEFAULT_IMAGE_SIZES,
    num_objects=(1, 0, None),
    extra_dims=DEFAULT_EXTRA_DIMS,
    dtypes=(torch.uint8, torch.bool),
):
    for size, num_objects_, extra_dims_ in itertools.product(sizes, num_objects, extra_dims):
        yield make_segmentation_mask_loader(size=size, num_objects=num_objects_, extra_dims=extra_dims_)

    for num_objects_, dtype in itertools.product(num_objects, dtypes):
        yield make_segmentation_mask_loader(num_objects=num_objects_, dtype=dtype)


make_segmentation_masks = from_loaders(make_segmentation_mask_loaders)
