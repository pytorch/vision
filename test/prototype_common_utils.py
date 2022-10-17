"""This module is separated from common_utils.py to prevent the former to be dependent on torchvision.prototype"""

import collections.abc
import dataclasses
import functools
from typing import Callable, Optional, Sequence, Tuple, Union

import PIL.Image
import pytest
import torch
import torch.testing
from datasets_utils import combinations_grid
from torch.nn.functional import one_hot
from torch.testing._comparison import (
    assert_equal as _assert_equal,
    BooleanPair,
    ErrorMeta,
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
    "VALID_EXTRA_DIMS",
    "make_image_loaders",
    "make_image",
    "make_images",
    "make_bounding_box_loaders",
    "make_bounding_box",
    "make_bounding_boxes",
    "make_label",
    "make_one_hot_labels",
    "make_detection_mask_loaders",
    "make_detection_mask",
    "make_detection_masks",
    "make_segmentation_mask_loaders",
    "make_segmentation_mask",
    "make_segmentation_masks",
    "make_mask_loaders",
    "make_masks",
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
            to_image_tensor(input) if not isinstance(input, torch.Tensor) else features.Image(input)
            for input in [actual, expected]
        ]
        # This broadcast is needed, because `features.Mask`'s can have a 2D shape, but converting the equivalent PIL
        # image to a tensor adds a singleton leading dimension.
        # Although it looks like this belongs in `self._equalize_attributes`, it has to happen here.
        # `self._equalize_attributes` is called after `super()._compare_attributes` and that has an unconditional
        # shape check that will fail if we don't broadcast before.
        try:
            actual, expected = torch.broadcast_tensors(actual, expected)
        except RuntimeError:
            raise ErrorMeta(
                AssertionError,
                f"The image shapes are not broadcastable: {actual.shape} != {expected.shape}.",
                id=id,
            ) from None
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


DEFAULT_SQUARE_IMAGE_SIZE = 15
DEFAULT_LANDSCAPE_IMAGE_SIZE = (7, 33)
DEFAULT_PORTRAIT_IMAGE_SIZE = (31, 9)
DEFAULT_IMAGE_SIZES = (DEFAULT_LANDSCAPE_IMAGE_SIZE, DEFAULT_PORTRAIT_IMAGE_SIZE, DEFAULT_SQUARE_IMAGE_SIZE, "random")


def _parse_image_size(size, *, name="size"):
    if size == "random":
        return tuple(torch.randint(15, 33, (2,)).tolist())
    elif isinstance(size, int) and size > 0:
        return (size, size)
    elif (
        isinstance(size, collections.abc.Sequence)
        and len(size) == 2
        and all(isinstance(length, int) and length > 0 for length in size)
    ):
        return tuple(size)
    else:
        raise pytest.UsageError(
            f"'{name}' can either be `'random'`, a positive integer, or a sequence of two positive integers,"
            f"but got {size} instead."
        )


VALID_EXTRA_DIMS = ((), (4,), (2, 3))
DEGENERATE_BATCH_DIMS = ((0,), (5, 0), (0, 5))

DEFAULT_EXTRA_DIMS = (*VALID_EXTRA_DIMS, *DEGENERATE_BATCH_DIMS)


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


@dataclasses.dataclass
class TensorLoader:
    fn: Callable[[Sequence[int], torch.dtype, Union[str, torch.device]], torch.Tensor]
    shape: Sequence[int]
    dtype: torch.dtype

    def load(self, device):
        return self.fn(self.shape, self.dtype, device)


@dataclasses.dataclass
class ImageLoader(TensorLoader):
    color_space: features.ColorSpace
    image_size: Tuple[int, int] = dataclasses.field(init=False)
    num_channels: int = dataclasses.field(init=False)

    def __post_init__(self):
        self.image_size = self.shape[-2:]
        self.num_channels = self.shape[-3]


def make_image_loader(
    size="random",
    *,
    color_space=features.ColorSpace.RGB,
    extra_dims=(),
    dtype=torch.float32,
    constant_alpha=True,
):
    size = _parse_image_size(size)

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
    for params in combinations_grid(size=sizes, color_space=color_spaces, extra_dims=extra_dims, dtype=dtypes):
        yield make_image_loader(**params, constant_alpha=constant_alpha)


make_images = from_loaders(make_image_loaders)


@dataclasses.dataclass
class BoundingBoxLoader(TensorLoader):
    format: features.BoundingBoxFormat
    image_size: Tuple[int, int]


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


def make_bounding_box_loader(*, extra_dims=(), format, image_size="random", dtype=torch.float32):
    if isinstance(format, str):
        format = features.BoundingBoxFormat[format]
    if format not in {
        features.BoundingBoxFormat.XYXY,
        features.BoundingBoxFormat.XYWH,
        features.BoundingBoxFormat.CXCYWH,
    }:
        raise pytest.UsageError(f"Can't make bounding box in format {format}")

    image_size = _parse_image_size(image_size, name="image_size")

    def fn(shape, dtype, device):
        *extra_dims, num_coordinates = shape
        if num_coordinates != 4:
            raise pytest.UsageError()

        if any(dim == 0 for dim in extra_dims):
            return features.BoundingBox(
                torch.empty(*extra_dims, 4, dtype=dtype, device=device), format=format, image_size=image_size
            )

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

        return features.BoundingBox(
            torch.stack(parts, dim=-1).to(dtype=dtype, device=device), format=format, image_size=image_size
        )

    return BoundingBoxLoader(fn, shape=(*extra_dims, 4), dtype=dtype, format=format, image_size=image_size)


make_bounding_box = from_loader(make_bounding_box_loader)


def make_bounding_box_loaders(
    *,
    extra_dims=DEFAULT_EXTRA_DIMS,
    formats=tuple(features.BoundingBoxFormat),
    image_size="random",
    dtypes=(torch.float32, torch.int64),
):
    for params in combinations_grid(extra_dims=extra_dims, format=formats, dtype=dtypes):
        yield make_bounding_box_loader(**params, image_size=image_size)


make_bounding_boxes = from_loaders(make_bounding_box_loaders)


@dataclasses.dataclass
class LabelLoader(TensorLoader):
    categories: Optional[Sequence[str]]


def _parse_categories(categories):
    if categories is None:
        num_categories = int(torch.randint(1, 11, ()))
    elif isinstance(categories, int):
        num_categories = categories
        categories = [f"category{idx}" for idx in range(num_categories)]
    elif isinstance(categories, collections.abc.Sequence) and all(isinstance(category, str) for category in categories):
        categories = list(categories)
        num_categories = len(categories)
    else:
        raise pytest.UsageError(
            f"`categories` can either be `None` (default), an integer, or a sequence of strings, "
            f"but got '{categories}' instead."
        )
    return categories, num_categories


def make_label_loader(*, extra_dims=(), categories=None, dtype=torch.int64):
    categories, num_categories = _parse_categories(categories)

    def fn(shape, dtype, device):
        # The idiom `make_tensor(..., dtype=torch.int64).to(dtype)` is intentional to only get integer values,
        # regardless of the requested dtype, e.g. 0 or 0.0 rather than 0 or 0.123
        data = torch.testing.make_tensor(shape, low=0, high=num_categories, dtype=torch.int64, device=device).to(dtype)
        return features.Label(data, categories=categories)

    return LabelLoader(fn, shape=extra_dims, dtype=dtype, categories=categories)


make_label = from_loader(make_label_loader)


@dataclasses.dataclass
class OneHotLabelLoader(TensorLoader):
    categories: Optional[Sequence[str]]


def make_one_hot_label_loader(*, categories=None, extra_dims=(), dtype=torch.int64):
    categories, num_categories = _parse_categories(categories)

    def fn(shape, dtype, device):
        if num_categories == 0:
            data = torch.empty(shape, dtype=dtype, device=device)
        else:
            # The idiom `make_label_loader(..., dtype=torch.int64); ...; one_hot(...).to(dtype)` is intentional
            # since `one_hot` only supports int64
            label = make_label_loader(extra_dims=extra_dims, categories=num_categories, dtype=torch.int64).load(device)
            data = one_hot(label, num_classes=num_categories).to(dtype)
        return features.OneHotLabel(data, categories=categories)

    return OneHotLabelLoader(fn, shape=(*extra_dims, num_categories), dtype=dtype, categories=categories)


def make_one_hot_label_loaders(
    *,
    categories=(1, 0, None),
    extra_dims=DEFAULT_EXTRA_DIMS,
    dtypes=(torch.int64, torch.float32),
):
    for params in combinations_grid(categories=categories, extra_dims=extra_dims, dtype=dtypes):
        yield make_one_hot_label_loader(**params)


make_one_hot_labels = from_loaders(make_one_hot_label_loaders)


class MaskLoader(TensorLoader):
    pass


def make_detection_mask_loader(size="random", *, num_objects="random", extra_dims=(), dtype=torch.uint8):
    # This produces "detection" masks, i.e. `(*, N, H, W)`, where `N` denotes the number of objects
    size = _parse_image_size(size)
    num_objects = int(torch.randint(1, 11, ())) if num_objects == "random" else num_objects

    def fn(shape, dtype, device):
        data = torch.testing.make_tensor(shape, low=0, high=2, dtype=dtype, device=device)
        return features.Mask(data)

    return MaskLoader(fn, shape=(*extra_dims, num_objects, *size), dtype=dtype)


make_detection_mask = from_loader(make_detection_mask_loader)


def make_detection_mask_loaders(
    sizes=DEFAULT_IMAGE_SIZES,
    num_objects=(1, 0, "random"),
    extra_dims=DEFAULT_EXTRA_DIMS,
    dtypes=(torch.uint8,),
):
    for params in combinations_grid(size=sizes, num_objects=num_objects, extra_dims=extra_dims, dtype=dtypes):
        yield make_detection_mask_loader(**params)


make_detection_masks = from_loaders(make_detection_mask_loaders)


def make_segmentation_mask_loader(size="random", *, num_categories="random", extra_dims=(), dtype=torch.uint8):
    # This produces "segmentation" masks, i.e. `(*, H, W)`, where the category is encoded in the values
    size = _parse_image_size(size)
    num_categories = int(torch.randint(1, 11, ())) if num_categories == "random" else num_categories

    def fn(shape, dtype, device):
        data = torch.testing.make_tensor(shape, low=0, high=num_categories, dtype=dtype, device=device)
        return features.Mask(data)

    return MaskLoader(fn, shape=(*extra_dims, *size), dtype=dtype)


make_segmentation_mask = from_loader(make_segmentation_mask_loader)


def make_segmentation_mask_loaders(
    *,
    sizes=DEFAULT_IMAGE_SIZES,
    num_categories=(1, 2, "random"),
    extra_dims=DEFAULT_EXTRA_DIMS,
    dtypes=(torch.uint8,),
):
    for params in combinations_grid(size=sizes, num_categories=num_categories, extra_dims=extra_dims, dtype=dtypes):
        yield make_segmentation_mask_loader(**params)


make_segmentation_masks = from_loaders(make_segmentation_mask_loaders)


def make_mask_loaders(
    *,
    sizes=DEFAULT_IMAGE_SIZES,
    num_objects=(1, 0, "random"),
    num_categories=(1, 2, "random"),
    extra_dims=DEFAULT_EXTRA_DIMS,
    dtypes=(torch.uint8,),
):
    yield from make_detection_mask_loaders(sizes=sizes, num_objects=num_objects, extra_dims=extra_dims, dtypes=dtypes)
    yield from make_segmentation_mask_loaders(
        sizes=sizes, num_categories=num_categories, extra_dims=extra_dims, dtypes=dtypes
    )


make_masks = from_loaders(make_mask_loaders)
