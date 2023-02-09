"""This module is separated from common_utils.py to prevent the former to be dependent on torchvision.prototype"""

import collections.abc
import dataclasses
import enum
import functools
import pathlib
from collections import defaultdict
from typing import Callable, Optional, Sequence, Tuple, Union

import PIL.Image
import pytest
import torch
import torch.testing
from datasets_utils import combinations_grid
from torch.nn.functional import one_hot
from torch.testing._comparison import BooleanPair, NonePair, not_close_error_metas, NumberPair, TensorLikePair
from torchvision.prototype import datapoints
from torchvision.prototype.transforms.functional import convert_dtype_image_tensor, to_image_tensor
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
    "make_video",
    "make_videos",
    "TestMark",
    "mark_framework_limitation",
    "InfoBase",
]


class ImagePair(TensorLikePair):
    def __init__(
        self,
        actual,
        expected,
        *,
        mae=False,
        **other_parameters,
    ):
        if all(isinstance(input, PIL.Image.Image) for input in [actual, expected]):
            actual, expected = [to_image_tensor(input) for input in [actual, expected]]

        super().__init__(actual, expected, **other_parameters)
        self.mae = mae

    def compare(self) -> None:
        actual, expected = self.actual, self.expected

        self._compare_attributes(actual, expected)
        actual, expected = self._equalize_attributes(actual, expected)

        if self.mae:
            actual, expected = self._promote_for_comparison(actual, expected)
            mae = float(torch.abs(actual - expected).float().mean())
            if mae > self.atol:
                self._fail(
                    AssertionError,
                    f"The MAE of the images is {mae}, but only {self.atol} is allowed.",
                )
        else:
            super()._compare_values(actual, expected)


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

    error_metas = not_close_error_metas(
        actual,
        expected,
        pair_types=(
            NonePair,
            BooleanPair,
            NumberPair,
            ImagePair,
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
        **kwargs,
    )

    if error_metas:
        raise error_metas[0].to_error(msg)


assert_equal = functools.partial(assert_close, rtol=0, atol=0)


def parametrized_error_message(*args, **kwargs):
    def to_str(obj):
        if isinstance(obj, torch.Tensor) and obj.numel() > 10:
            return f"tensor(shape={list(obj.shape)}, dtype={obj.dtype}, device={obj.device})"
        elif isinstance(obj, enum.Enum):
            return f"{type(obj).__name__}.{obj.name}"
        else:
            return repr(obj)

    if args or kwargs:
        postfix = "\n".join(
            [
                "",
                "Failure happened for the following parameters:",
                "",
                *[to_str(arg) for arg in args],
                *[f"{name}={to_str(kwarg)}" for name, kwarg in kwargs.items()],
            ]
        )
    else:
        postfix = ""

    def wrapper(msg):
        return msg + postfix

    return wrapper


class ArgsKwargs:
    def __init__(self, *args, **kwargs):
        self.args = args
        self.kwargs = kwargs

    def __iter__(self):
        yield self.args
        yield self.kwargs

    def load(self, device="cpu"):
        return ArgsKwargs(
            *(arg.load(device) if isinstance(arg, TensorLoader) else arg for arg in self.args),
            **{
                keyword: arg.load(device) if isinstance(arg, TensorLoader) else arg
                for keyword, arg in self.kwargs.items()
            },
        )


DEFAULT_SQUARE_SPATIAL_SIZE = 15
DEFAULT_LANDSCAPE_SPATIAL_SIZE = (7, 33)
DEFAULT_PORTRAIT_SPATIAL_SIZE = (31, 9)
DEFAULT_SPATIAL_SIZES = (
    DEFAULT_LANDSCAPE_SPATIAL_SIZE,
    DEFAULT_PORTRAIT_SPATIAL_SIZE,
    DEFAULT_SQUARE_SPATIAL_SIZE,
    "random",
)


def _parse_spatial_size(size, *, name="size"):
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
        device = kwargs.pop("device", "cpu")
        loader = loader_fn(*args, **kwargs)
        return loader.load(device)

    return wrapper


def from_loaders(loaders_fn):
    def wrapper(*args, **kwargs):
        device = kwargs.pop("device", "cpu")
        loaders = loaders_fn(*args, **kwargs)
        for loader in loaders:
            yield loader.load(device)

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
    spatial_size: Tuple[int, int] = dataclasses.field(init=False)
    num_channels: int = dataclasses.field(init=False)

    def __post_init__(self):
        self.spatial_size = self.shape[-2:]
        self.num_channels = self.shape[-3]


NUM_CHANNELS_MAP = {
    "GRAY": 1,
    "GRAY_ALPHA": 2,
    "RGB": 3,
    "RGBA": 4,
}


def get_num_channels(color_space):
    num_channels = NUM_CHANNELS_MAP.get(color_space)
    if not num_channels:
        raise pytest.UsageError(f"Can't determine the number of channels for color space {color_space}")
    return num_channels


def make_image_loader(
    size="random",
    *,
    color_space="RGB",
    extra_dims=(),
    dtype=torch.float32,
    constant_alpha=True,
):
    size = _parse_spatial_size(size)
    num_channels = get_num_channels(color_space)

    def fn(shape, dtype, device):
        max_value = get_max_value(dtype)
        data = torch.testing.make_tensor(shape, low=0, high=max_value, dtype=dtype, device=device)
        if color_space in {"GRAY_ALPHA", "RGBA"} and constant_alpha:
            data[..., -1, :, :] = max_value
        return datapoints.Image(data)

    return ImageLoader(fn, shape=(*extra_dims, num_channels, *size), dtype=dtype)


make_image = from_loader(make_image_loader)


def make_image_loaders(
    *,
    sizes=DEFAULT_SPATIAL_SIZES,
    color_spaces=(
        "GRAY",
        "GRAY_ALPHA",
        "RGB",
        "RGBA",
    ),
    extra_dims=DEFAULT_EXTRA_DIMS,
    dtypes=(torch.float32, torch.uint8),
    constant_alpha=True,
):
    for params in combinations_grid(size=sizes, color_space=color_spaces, extra_dims=extra_dims, dtype=dtypes):
        yield make_image_loader(**params, constant_alpha=constant_alpha)


make_images = from_loaders(make_image_loaders)


def make_image_loader_for_interpolation(size="random", *, color_space="RGB", dtype=torch.uint8):
    size = _parse_spatial_size(size)
    num_channels = get_num_channels(color_space)

    def fn(shape, dtype, device):
        height, width = shape[-2:]

        image_pil = (
            PIL.Image.open(pathlib.Path(__file__).parent / "assets" / "encode_jpeg" / "grace_hopper_517x606.jpg")
            .resize((width, height))
            .convert(
                {
                    "GRAY": "L",
                    "GRAY_ALPHA": "LA",
                    "RGB": "RGB",
                    "RGBA": "RGBA",
                }[color_space]
            )
        )

        image_tensor = convert_dtype_image_tensor(to_image_tensor(image_pil).to(device=device), dtype=dtype)

        return datapoints.Image(image_tensor)

    return ImageLoader(fn, shape=(num_channels, *size), dtype=dtype)


def make_image_loaders_for_interpolation(
    sizes=((233, 147),),
    color_spaces=("RGB",),
    dtypes=(torch.uint8,),
):
    for params in combinations_grid(size=sizes, color_space=color_spaces, dtype=dtypes):
        yield make_image_loader_for_interpolation(**params)


@dataclasses.dataclass
class BoundingBoxLoader(TensorLoader):
    format: datapoints.BoundingBoxFormat
    spatial_size: Tuple[int, int]


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


def make_bounding_box_loader(*, extra_dims=(), format, spatial_size="random", dtype=torch.float32):
    if isinstance(format, str):
        format = datapoints.BoundingBoxFormat[format]
    if format not in {
        datapoints.BoundingBoxFormat.XYXY,
        datapoints.BoundingBoxFormat.XYWH,
        datapoints.BoundingBoxFormat.CXCYWH,
    }:
        raise pytest.UsageError(f"Can't make bounding box in format {format}")

    spatial_size = _parse_spatial_size(spatial_size, name="spatial_size")

    def fn(shape, dtype, device):
        *extra_dims, num_coordinates = shape
        if num_coordinates != 4:
            raise pytest.UsageError()

        if any(dim == 0 for dim in extra_dims):
            return datapoints.BoundingBox(
                torch.empty(*extra_dims, 4, dtype=dtype, device=device), format=format, spatial_size=spatial_size
            )

        height, width = spatial_size

        if format == datapoints.BoundingBoxFormat.XYXY:
            x1 = torch.randint(0, width // 2, extra_dims)
            y1 = torch.randint(0, height // 2, extra_dims)
            x2 = randint_with_tensor_bounds(x1 + 1, width - x1) + x1
            y2 = randint_with_tensor_bounds(y1 + 1, height - y1) + y1
            parts = (x1, y1, x2, y2)
        elif format == datapoints.BoundingBoxFormat.XYWH:
            x = torch.randint(0, width // 2, extra_dims)
            y = torch.randint(0, height // 2, extra_dims)
            w = randint_with_tensor_bounds(1, width - x)
            h = randint_with_tensor_bounds(1, height - y)
            parts = (x, y, w, h)
        else:  # format == features.BoundingBoxFormat.CXCYWH:
            cx = torch.randint(1, width - 1, extra_dims)
            cy = torch.randint(1, height - 1, extra_dims)
            w = randint_with_tensor_bounds(1, torch.minimum(cx, width - cx) + 1)
            h = randint_with_tensor_bounds(1, torch.minimum(cy, height - cy) + 1)
            parts = (cx, cy, w, h)

        return datapoints.BoundingBox(
            torch.stack(parts, dim=-1).to(dtype=dtype, device=device), format=format, spatial_size=spatial_size
        )

    return BoundingBoxLoader(fn, shape=(*extra_dims, 4), dtype=dtype, format=format, spatial_size=spatial_size)


make_bounding_box = from_loader(make_bounding_box_loader)


def make_bounding_box_loaders(
    *,
    extra_dims=DEFAULT_EXTRA_DIMS,
    formats=tuple(datapoints.BoundingBoxFormat),
    spatial_size="random",
    dtypes=(torch.float32, torch.int64),
):
    for params in combinations_grid(extra_dims=extra_dims, format=formats, dtype=dtypes):
        yield make_bounding_box_loader(**params, spatial_size=spatial_size)


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
        return datapoints.Label(data, categories=categories)

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
        return datapoints.OneHotLabel(data, categories=categories)

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
    size = _parse_spatial_size(size)
    num_objects = int(torch.randint(1, 11, ())) if num_objects == "random" else num_objects

    def fn(shape, dtype, device):
        data = torch.testing.make_tensor(shape, low=0, high=2, dtype=dtype, device=device)
        return datapoints.Mask(data)

    return MaskLoader(fn, shape=(*extra_dims, num_objects, *size), dtype=dtype)


make_detection_mask = from_loader(make_detection_mask_loader)


def make_detection_mask_loaders(
    sizes=DEFAULT_SPATIAL_SIZES,
    num_objects=(1, 0, "random"),
    extra_dims=DEFAULT_EXTRA_DIMS,
    dtypes=(torch.uint8,),
):
    for params in combinations_grid(size=sizes, num_objects=num_objects, extra_dims=extra_dims, dtype=dtypes):
        yield make_detection_mask_loader(**params)


make_detection_masks = from_loaders(make_detection_mask_loaders)


def make_segmentation_mask_loader(size="random", *, num_categories="random", extra_dims=(), dtype=torch.uint8):
    # This produces "segmentation" masks, i.e. `(*, H, W)`, where the category is encoded in the values
    size = _parse_spatial_size(size)
    num_categories = int(torch.randint(1, 11, ())) if num_categories == "random" else num_categories

    def fn(shape, dtype, device):
        data = torch.testing.make_tensor(shape, low=0, high=num_categories, dtype=dtype, device=device)
        return datapoints.Mask(data)

    return MaskLoader(fn, shape=(*extra_dims, *size), dtype=dtype)


make_segmentation_mask = from_loader(make_segmentation_mask_loader)


def make_segmentation_mask_loaders(
    *,
    sizes=DEFAULT_SPATIAL_SIZES,
    num_categories=(1, 2, "random"),
    extra_dims=DEFAULT_EXTRA_DIMS,
    dtypes=(torch.uint8,),
):
    for params in combinations_grid(size=sizes, num_categories=num_categories, extra_dims=extra_dims, dtype=dtypes):
        yield make_segmentation_mask_loader(**params)


make_segmentation_masks = from_loaders(make_segmentation_mask_loaders)


def make_mask_loaders(
    *,
    sizes=DEFAULT_SPATIAL_SIZES,
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


class VideoLoader(ImageLoader):
    pass


def make_video_loader(
    size="random",
    *,
    color_space="RGB",
    num_frames="random",
    extra_dims=(),
    dtype=torch.uint8,
):
    size = _parse_spatial_size(size)
    num_frames = int(torch.randint(1, 5, ())) if num_frames == "random" else num_frames

    def fn(shape, dtype, device):
        video = make_image(size=shape[-2:], extra_dims=shape[:-3], dtype=dtype, device=device)
        return datapoints.Video(video)

    return VideoLoader(fn, shape=(*extra_dims, num_frames, get_num_channels(color_space), *size), dtype=dtype)


make_video = from_loader(make_video_loader)


def make_video_loaders(
    *,
    sizes=DEFAULT_SPATIAL_SIZES,
    color_spaces=(
        "GRAY",
        "RGB",
    ),
    num_frames=(1, 0, "random"),
    extra_dims=DEFAULT_EXTRA_DIMS,
    dtypes=(torch.uint8,),
):
    for params in combinations_grid(
        size=sizes, color_space=color_spaces, num_frames=num_frames, extra_dims=extra_dims, dtype=dtypes
    ):
        yield make_video_loader(**params)


make_videos = from_loaders(make_video_loaders)


class TestMark:
    def __init__(
        self,
        # Tuple of test class name and test function name that identifies the test the mark is applied to. If there is
        # no test class, i.e. a standalone test function, use `None`.
        test_id,
        # `pytest.mark.*` to apply, e.g. `pytest.mark.skip` or `pytest.mark.xfail`
        mark,
        *,
        # Callable, that will be passed an `ArgsKwargs` and should return a boolean to indicate if the mark will be
        # applied. If omitted, defaults to always apply.
        condition=None,
    ):
        self.test_id = test_id
        self.mark = mark
        self.condition = condition or (lambda args_kwargs: True)


def mark_framework_limitation(test_id, reason):
    # The purpose of this function is to have a single entry point for skip marks that are only there, because the test
    # framework cannot handle the kernel in general or a specific parameter combination.
    # As development progresses, we can change the `mark.skip` to `mark.xfail` from time to time to see if the skip is
    # still justified.
    # We don't want to use `mark.xfail` all the time, because that actually runs the test until an error happens. Thus,
    # we are wasting CI resources for no reason for most of the time
    return TestMark(test_id, pytest.mark.skip(reason=reason))


class InfoBase:
    def __init__(
        self,
        *,
        # Identifier if the info that shows up the parametrization.
        id,
        # Test markers that will be (conditionally) applied to an `ArgsKwargs` parametrization.
        # See the `TestMark` class for details
        test_marks=None,
        # Additional parameters, e.g. `rtol=1e-3`, passed to `assert_close`. Keys are a 3-tuple of `test_id` (see
        # `TestMark`), the dtype, and the device.
        closeness_kwargs=None,
    ):
        self.id = id

        self.test_marks = test_marks or []
        test_marks_map = defaultdict(list)
        for test_mark in self.test_marks:
            test_marks_map[test_mark.test_id].append(test_mark)
        self._test_marks_map = dict(test_marks_map)

        self.closeness_kwargs = closeness_kwargs or dict()

    def get_marks(self, test_id, args_kwargs):
        return [
            test_mark.mark for test_mark in self._test_marks_map.get(test_id, []) if test_mark.condition(args_kwargs)
        ]

    def get_closeness_kwargs(self, test_id, *, dtype, device):
        if not (isinstance(test_id, tuple) and len(test_id) == 2):
            msg = "`test_id` should be a `Tuple[Optional[str], str]` denoting the test class and function name"
            if callable(test_id):
                msg += ". Did you forget to add the `test_id` fixture to parameters of the test?"
            else:
                msg += f", but got {test_id} instead."
            raise pytest.UsageError(msg)
        if isinstance(device, torch.device):
            device = device.type
        return self.closeness_kwargs.get((test_id, dtype, device), dict())
