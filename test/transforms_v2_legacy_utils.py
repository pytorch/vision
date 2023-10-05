"""
As the name implies, these are legacy utilities that are hopefully removed soon. The future of
transforms v2 testing is in test/test_transforms_v2_refactored.py. All new test should be
implemented there and must not use any of the utilities here.

The following legacy modules depend on this module

- test_transforms_v2_consistency.py
"""

import collections.abc
import dataclasses
import enum
import itertools
import pathlib
from collections import defaultdict
from typing import Callable, Sequence, Tuple, Union

import PIL.Image
import pytest
import torch

from torchvision import tv_tensors
from torchvision.transforms._functional_tensor import _max_value as get_max_value
from torchvision.transforms.v2.functional import to_dtype_image, to_image, to_pil_image


def combinations_grid(**kwargs):
    """Creates a grid of input combinations.

    Each element in the returned sequence is a dictionary containing one possible combination as values.

    Example:
        >>> combinations_grid(foo=("bar", "baz"), spam=("eggs", "ham"))
        [
            {'foo': 'bar', 'spam': 'eggs'},
            {'foo': 'bar', 'spam': 'ham'},
            {'foo': 'baz', 'spam': 'eggs'},
            {'foo': 'baz', 'spam': 'ham'}
        ]
    """
    return [dict(zip(kwargs.keys(), values)) for values in itertools.product(*kwargs.values())]


DEFAULT_SIZE = (17, 11)

NUM_CHANNELS_MAP = {
    "GRAY": 1,
    "GRAY_ALPHA": 2,
    "RGB": 3,
    "RGBA": 4,
}


def make_image(
    size=DEFAULT_SIZE,
    *,
    color_space="RGB",
    batch_dims=(),
    dtype=None,
    device="cpu",
    memory_format=torch.contiguous_format,
):
    num_channels = NUM_CHANNELS_MAP[color_space]
    dtype = dtype or torch.uint8
    max_value = get_max_value(dtype)
    data = torch.testing.make_tensor(
        (*batch_dims, num_channels, *size),
        low=0,
        high=max_value,
        dtype=dtype,
        device=device,
        memory_format=memory_format,
    )
    if color_space in {"GRAY_ALPHA", "RGBA"}:
        data[..., -1, :, :] = max_value

    return tv_tensors.Image(data)


def make_image_tensor(*args, **kwargs):
    return make_image(*args, **kwargs).as_subclass(torch.Tensor)


def make_image_pil(*args, **kwargs):
    return to_pil_image(make_image(*args, **kwargs))


def make_bounding_boxes(
    canvas_size=DEFAULT_SIZE,
    *,
    format=tv_tensors.BoundingBoxFormat.XYXY,
    batch_dims=(),
    dtype=None,
    device="cpu",
):
    def sample_position(values, max_value):
        # We cannot use torch.randint directly here, because it only allows integer scalars as values for low and high.
        # However, if we have batch_dims, we need tensors as limits.
        return torch.stack([torch.randint(max_value - v, ()) for v in values.flatten().tolist()]).reshape(values.shape)

    if isinstance(format, str):
        format = tv_tensors.BoundingBoxFormat[format]

    dtype = dtype or torch.float32

    if any(dim == 0 for dim in batch_dims):
        return tv_tensors.BoundingBoxes(
            torch.empty(*batch_dims, 4, dtype=dtype, device=device), format=format, canvas_size=canvas_size
        )

    h, w = [torch.randint(1, c, batch_dims) for c in canvas_size]
    y = sample_position(h, canvas_size[0])
    x = sample_position(w, canvas_size[1])

    if format is tv_tensors.BoundingBoxFormat.XYWH:
        parts = (x, y, w, h)
    elif format is tv_tensors.BoundingBoxFormat.XYXY:
        x1, y1 = x, y
        x2 = x1 + w
        y2 = y1 + h
        parts = (x1, y1, x2, y2)
    elif format is tv_tensors.BoundingBoxFormat.CXCYWH:
        cx = x + w / 2
        cy = y + h / 2
        parts = (cx, cy, w, h)
    else:
        raise ValueError(f"Format {format} is not supported")

    return tv_tensors.BoundingBoxes(
        torch.stack(parts, dim=-1).to(dtype=dtype, device=device), format=format, canvas_size=canvas_size
    )


def make_detection_mask(size=DEFAULT_SIZE, *, num_objects=5, batch_dims=(), dtype=None, device="cpu"):
    """Make a "detection" mask, i.e. (*, N, H, W), where each object is encoded as one of N boolean masks"""
    return tv_tensors.Mask(
        torch.testing.make_tensor(
            (*batch_dims, num_objects, *size),
            low=0,
            high=2,
            dtype=dtype or torch.bool,
            device=device,
        )
    )


def make_segmentation_mask(size=DEFAULT_SIZE, *, num_categories=10, batch_dims=(), dtype=None, device="cpu"):
    """Make a "segmentation" mask, i.e. (*, H, W), where the category is encoded as pixel value"""
    return tv_tensors.Mask(
        torch.testing.make_tensor(
            (*batch_dims, *size),
            low=0,
            high=num_categories,
            dtype=dtype or torch.uint8,
            device=device,
        )
    )


def make_video(size=DEFAULT_SIZE, *, num_frames=3, batch_dims=(), **kwargs):
    return tv_tensors.Video(make_image(size, batch_dims=(*batch_dims, num_frames), **kwargs))


def make_video_tensor(*args, **kwargs):
    return make_video(*args, **kwargs).as_subclass(torch.Tensor)


DEFAULT_SQUARE_SPATIAL_SIZE = 15
DEFAULT_LANDSCAPE_SPATIAL_SIZE = (7, 33)
DEFAULT_PORTRAIT_SPATIAL_SIZE = (31, 9)
DEFAULT_SPATIAL_SIZES = (
    DEFAULT_LANDSCAPE_SPATIAL_SIZE,
    DEFAULT_PORTRAIT_SPATIAL_SIZE,
    DEFAULT_SQUARE_SPATIAL_SIZE,
)


def _parse_size(size, *, name="size"):
    if size == "random":
        raise ValueError("This should never happen")
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


def get_num_channels(color_space):
    num_channels = NUM_CHANNELS_MAP.get(color_space)
    if not num_channels:
        raise pytest.UsageError(f"Can't determine the number of channels for color space {color_space}")
    return num_channels


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
    memory_format: torch.memory_format = torch.contiguous_format
    canvas_size: Tuple[int, int] = dataclasses.field(init=False)

    def __post_init__(self):
        self.spatial_size = self.canvas_size = self.shape[-2:]
        self.num_channels = self.shape[-3]

    def load(self, device):
        return self.fn(self.shape, self.dtype, device, memory_format=self.memory_format)


def make_image_loader(
    size=DEFAULT_PORTRAIT_SPATIAL_SIZE,
    *,
    color_space="RGB",
    extra_dims=(),
    dtype=torch.float32,
    constant_alpha=True,
    memory_format=torch.contiguous_format,
):
    if not constant_alpha:
        raise ValueError("This should never happen")
    size = _parse_size(size)
    num_channels = get_num_channels(color_space)

    def fn(shape, dtype, device, memory_format):
        *batch_dims, _, height, width = shape
        return make_image(
            (height, width),
            color_space=color_space,
            batch_dims=batch_dims,
            dtype=dtype,
            device=device,
            memory_format=memory_format,
        )

    return ImageLoader(fn, shape=(*extra_dims, num_channels, *size), dtype=dtype, memory_format=memory_format)


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
    dtypes=(torch.float32, torch.float64, torch.uint8),
    constant_alpha=True,
):
    for params in combinations_grid(size=sizes, color_space=color_spaces, extra_dims=extra_dims, dtype=dtypes):
        yield make_image_loader(**params, constant_alpha=constant_alpha)


make_images = from_loaders(make_image_loaders)


def make_image_loader_for_interpolation(
    size=(233, 147), *, color_space="RGB", dtype=torch.uint8, memory_format=torch.contiguous_format
):
    size = _parse_size(size)
    num_channels = get_num_channels(color_space)

    def fn(shape, dtype, device, memory_format):
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

        image_tensor = to_image(image_pil)
        if memory_format == torch.contiguous_format:
            image_tensor = image_tensor.to(device=device, memory_format=memory_format, copy=True)
        else:
            image_tensor = image_tensor.to(device=device)
        image_tensor = to_dtype_image(image_tensor, dtype=dtype, scale=True)

        return tv_tensors.Image(image_tensor)

    return ImageLoader(fn, shape=(num_channels, *size), dtype=dtype, memory_format=memory_format)


def make_image_loaders_for_interpolation(
    sizes=((233, 147),),
    color_spaces=("RGB",),
    dtypes=(torch.uint8,),
    memory_formats=(torch.contiguous_format, torch.channels_last),
):
    for params in combinations_grid(size=sizes, color_space=color_spaces, dtype=dtypes, memory_format=memory_formats):
        yield make_image_loader_for_interpolation(**params)


@dataclasses.dataclass
class BoundingBoxesLoader(TensorLoader):
    format: tv_tensors.BoundingBoxFormat
    spatial_size: Tuple[int, int]
    canvas_size: Tuple[int, int] = dataclasses.field(init=False)

    def __post_init__(self):
        self.canvas_size = self.spatial_size


def make_bounding_box_loader(*, extra_dims=(), format, spatial_size=DEFAULT_PORTRAIT_SPATIAL_SIZE, dtype=torch.float32):
    if isinstance(format, str):
        format = tv_tensors.BoundingBoxFormat[format]

    spatial_size = _parse_size(spatial_size, name="spatial_size")

    def fn(shape, dtype, device):
        *batch_dims, num_coordinates = shape
        if num_coordinates != 4:
            raise pytest.UsageError()

        return make_bounding_boxes(
            format=format, canvas_size=spatial_size, batch_dims=batch_dims, dtype=dtype, device=device
        )

    return BoundingBoxesLoader(fn, shape=(*extra_dims[-1:], 4), dtype=dtype, format=format, spatial_size=spatial_size)


def make_bounding_box_loaders(
    *,
    extra_dims=tuple(d for d in DEFAULT_EXTRA_DIMS if len(d) < 2),
    formats=tuple(tv_tensors.BoundingBoxFormat),
    spatial_size=DEFAULT_PORTRAIT_SPATIAL_SIZE,
    dtypes=(torch.float32, torch.float64, torch.int64),
):
    for params in combinations_grid(extra_dims=extra_dims, format=formats, dtype=dtypes):
        yield make_bounding_box_loader(**params, spatial_size=spatial_size)


make_multiple_bounding_boxes = from_loaders(make_bounding_box_loaders)


class MaskLoader(TensorLoader):
    pass


def make_detection_mask_loader(size=DEFAULT_PORTRAIT_SPATIAL_SIZE, *, num_objects=5, extra_dims=(), dtype=torch.uint8):
    # This produces "detection" masks, i.e. `(*, N, H, W)`, where `N` denotes the number of objects
    size = _parse_size(size)

    def fn(shape, dtype, device):
        *batch_dims, num_objects, height, width = shape
        return make_detection_mask(
            (height, width), num_objects=num_objects, batch_dims=batch_dims, dtype=dtype, device=device
        )

    return MaskLoader(fn, shape=(*extra_dims, num_objects, *size), dtype=dtype)


def make_detection_mask_loaders(
    sizes=DEFAULT_SPATIAL_SIZES,
    num_objects=(1, 0, 5),
    extra_dims=DEFAULT_EXTRA_DIMS,
    dtypes=(torch.uint8,),
):
    for params in combinations_grid(size=sizes, num_objects=num_objects, extra_dims=extra_dims, dtype=dtypes):
        yield make_detection_mask_loader(**params)


make_detection_masks = from_loaders(make_detection_mask_loaders)


def make_segmentation_mask_loader(
    size=DEFAULT_PORTRAIT_SPATIAL_SIZE, *, num_categories=10, extra_dims=(), dtype=torch.uint8
):
    # This produces "segmentation" masks, i.e. `(*, H, W)`, where the category is encoded in the values
    size = _parse_size(size)

    def fn(shape, dtype, device):
        *batch_dims, height, width = shape
        return make_segmentation_mask(
            (height, width), num_categories=num_categories, batch_dims=batch_dims, dtype=dtype, device=device
        )

    return MaskLoader(fn, shape=(*extra_dims, *size), dtype=dtype)


def make_segmentation_mask_loaders(
    *,
    sizes=DEFAULT_SPATIAL_SIZES,
    num_categories=(1, 2, 10),
    extra_dims=DEFAULT_EXTRA_DIMS,
    dtypes=(torch.uint8,),
):
    for params in combinations_grid(size=sizes, num_categories=num_categories, extra_dims=extra_dims, dtype=dtypes):
        yield make_segmentation_mask_loader(**params)


make_segmentation_masks = from_loaders(make_segmentation_mask_loaders)


def make_mask_loaders(
    *,
    sizes=DEFAULT_SPATIAL_SIZES,
    num_objects=(1, 0, 5),
    num_categories=(1, 2, 10),
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
    size=DEFAULT_PORTRAIT_SPATIAL_SIZE,
    *,
    color_space="RGB",
    num_frames=3,
    extra_dims=(),
    dtype=torch.uint8,
):
    size = _parse_size(size)

    def fn(shape, dtype, device, memory_format):
        *batch_dims, num_frames, _, height, width = shape
        return make_video(
            (height, width),
            num_frames=num_frames,
            batch_dims=batch_dims,
            color_space=color_space,
            dtype=dtype,
            device=device,
            memory_format=memory_format,
        )

    return VideoLoader(fn, shape=(*extra_dims, num_frames, get_num_channels(color_space), *size), dtype=dtype)


def make_video_loaders(
    *,
    sizes=DEFAULT_SPATIAL_SIZES,
    color_spaces=(
        "GRAY",
        "RGB",
    ),
    num_frames=(1, 0, 3),
    extra_dims=DEFAULT_EXTRA_DIMS,
    dtypes=(torch.uint8, torch.float32, torch.float64),
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


def mark_framework_limitation(test_id, reason, condition=None):
    # The purpose of this function is to have a single entry point for skip marks that are only there, because the test
    # framework cannot handle the kernel in general or a specific parameter combination.
    # As development progresses, we can change the `mark.skip` to `mark.xfail` from time to time to see if the skip is
    # still justified.
    # We don't want to use `mark.xfail` all the time, because that actually runs the test until an error happens. Thus,
    # we are wasting CI resources for no reason for most of the time
    return TestMark(test_id, pytest.mark.skip(reason=reason), condition=condition)


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


def parametrized_error_message(*args, **kwargs):
    def to_str(obj):
        if isinstance(obj, torch.Tensor) and obj.numel() > 30:
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
