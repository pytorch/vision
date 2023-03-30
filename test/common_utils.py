import collections.abc
import contextlib
import dataclasses
import enum
import functools
import itertools
import os
import pathlib
import random
import shutil
import sys
import tempfile
from collections import defaultdict
from subprocess import CalledProcessError, check_output, STDOUT
from typing import Callable, Sequence, Tuple, Union

import numpy as np

import PIL.Image
import pytest
import torch
import torch.testing
from PIL import Image

from torch.testing._comparison import BooleanPair, NonePair, not_close_error_metas, NumberPair, TensorLikePair
from torchvision import datapoints, io
from torchvision.transforms._functional_tensor import _max_value as get_max_value
from torchvision.transforms.v2.functional import convert_dtype_image_tensor, to_image_tensor


IN_OSS_CI = any(os.getenv(var) == "true" for var in ["CIRCLECI", "GITHUB_ACTIONS"])
IN_RE_WORKER = os.environ.get("INSIDE_RE_WORKER") is not None
IN_FBCODE = os.environ.get("IN_FBCODE_TORCHVISION") == "1"
CUDA_NOT_AVAILABLE_MSG = "CUDA device not available"
OSS_CI_GPU_NO_CUDA_MSG = "We're in an OSS GPU machine, and this test doesn't need cuda."


@contextlib.contextmanager
def get_tmp_dir(src=None, **kwargs):
    tmp_dir = tempfile.mkdtemp(**kwargs)
    if src is not None:
        os.rmdir(tmp_dir)
        shutil.copytree(src, tmp_dir)
    try:
        yield tmp_dir
    finally:
        shutil.rmtree(tmp_dir)


def set_rng_seed(seed):
    torch.manual_seed(seed)
    random.seed(seed)


class MapNestedTensorObjectImpl:
    def __init__(self, tensor_map_fn):
        self.tensor_map_fn = tensor_map_fn

    def __call__(self, object):
        if isinstance(object, torch.Tensor):
            return self.tensor_map_fn(object)

        elif isinstance(object, dict):
            mapped_dict = {}
            for key, value in object.items():
                mapped_dict[self(key)] = self(value)
            return mapped_dict

        elif isinstance(object, (list, tuple)):
            mapped_iter = []
            for iter in object:
                mapped_iter.append(self(iter))
            return mapped_iter if not isinstance(object, tuple) else tuple(mapped_iter)

        else:
            return object


def map_nested_tensor_object(object, tensor_map_fn):
    impl = MapNestedTensorObjectImpl(tensor_map_fn)
    return impl(object)


def is_iterable(obj):
    try:
        iter(obj)
        return True
    except TypeError:
        return False


@contextlib.contextmanager
def freeze_rng_state():
    rng_state = torch.get_rng_state()
    if torch.cuda.is_available():
        cuda_rng_state = torch.cuda.get_rng_state()
    yield
    if torch.cuda.is_available():
        torch.cuda.set_rng_state(cuda_rng_state)
    torch.set_rng_state(rng_state)


def cycle_over(objs):
    for idx, obj1 in enumerate(objs):
        for obj2 in objs[:idx] + objs[idx + 1 :]:
            yield obj1, obj2


def int_dtypes():
    return (torch.uint8, torch.int8, torch.int16, torch.int32, torch.int64)


def float_dtypes():
    return (torch.float32, torch.float64)


@contextlib.contextmanager
def disable_console_output():
    with contextlib.ExitStack() as stack, open(os.devnull, "w") as devnull:
        stack.enter_context(contextlib.redirect_stdout(devnull))
        stack.enter_context(contextlib.redirect_stderr(devnull))
        yield


def cpu_and_gpu():
    import pytest  # noqa

    return ("cpu", pytest.param("cuda", marks=pytest.mark.needs_cuda))


def needs_cuda(test_func):
    import pytest  # noqa

    return pytest.mark.needs_cuda(test_func)


def _create_data(height=3, width=3, channels=3, device="cpu"):
    # TODO: When all relevant tests are ported to pytest, turn this into a module-level fixture
    tensor = torch.randint(0, 256, (channels, height, width), dtype=torch.uint8, device=device)
    data = tensor.permute(1, 2, 0).contiguous().cpu().numpy()
    mode = "RGB"
    if channels == 1:
        mode = "L"
        data = data[..., 0]
    pil_img = Image.fromarray(data, mode=mode)
    return tensor, pil_img


def _create_data_batch(height=3, width=3, channels=3, num_samples=4, device="cpu"):
    # TODO: When all relevant tests are ported to pytest, turn this into a module-level fixture
    batch_tensor = torch.randint(0, 256, (num_samples, channels, height, width), dtype=torch.uint8, device=device)
    return batch_tensor


def get_list_of_videos(tmpdir, num_videos=5, sizes=None, fps=None):
    names = []
    for i in range(num_videos):
        if sizes is None:
            size = 5 * (i + 1)
        else:
            size = sizes[i]
        if fps is None:
            f = 5
        else:
            f = fps[i]
        data = torch.randint(0, 256, (size, 300, 400, 3), dtype=torch.uint8)
        name = os.path.join(tmpdir, f"{i}.mp4")
        names.append(name)
        io.write_video(name, data, fps=f)

    return names


def _assert_equal_tensor_to_pil(tensor, pil_image, msg=None):
    # FIXME: this is handled automatically by `assert_equal` below. Let's remove this in favor of it
    np_pil_image = np.array(pil_image)
    if np_pil_image.ndim == 2:
        np_pil_image = np_pil_image[:, :, None]
    pil_tensor = torch.as_tensor(np_pil_image.transpose((2, 0, 1)))
    if msg is None:
        msg = f"tensor:\n{tensor} \ndid not equal PIL tensor:\n{pil_tensor}"
    assert_equal(tensor.cpu(), pil_tensor, msg=msg)


def _assert_approx_equal_tensor_to_pil(
    tensor, pil_image, tol=1e-5, msg=None, agg_method="mean", allowed_percentage_diff=None
):
    # FIXME: this is handled automatically by `assert_close` below. Let's remove this in favor of it
    # TODO: we could just merge this into _assert_equal_tensor_to_pil
    np_pil_image = np.array(pil_image)
    if np_pil_image.ndim == 2:
        np_pil_image = np_pil_image[:, :, None]
    pil_tensor = torch.as_tensor(np_pil_image.transpose((2, 0, 1))).to(tensor)

    if allowed_percentage_diff is not None:
        # Assert that less than a given %age of pixels are different
        assert (tensor != pil_tensor).to(torch.float).mean() <= allowed_percentage_diff

    # error value can be mean absolute error, max abs error
    # Convert to float to avoid underflow when computing absolute difference
    tensor = tensor.to(torch.float)
    pil_tensor = pil_tensor.to(torch.float)
    err = getattr(torch, agg_method)(torch.abs(tensor - pil_tensor)).item()
    assert err < tol, f"{err} vs {tol}"


def _test_fn_on_batch(batch_tensors, fn, scripted_fn_atol=1e-8, **fn_kwargs):
    transformed_batch = fn(batch_tensors, **fn_kwargs)
    for i in range(len(batch_tensors)):
        img_tensor = batch_tensors[i, ...]
        transformed_img = fn(img_tensor, **fn_kwargs)
        torch.testing.assert_close(transformed_img, transformed_batch[i, ...], rtol=0, atol=1e-6)

    if scripted_fn_atol >= 0:
        scripted_fn = torch.jit.script(fn)
        # scriptable function test
        s_transformed_batch = scripted_fn(batch_tensors, **fn_kwargs)
        torch.testing.assert_close(transformed_batch, s_transformed_batch, rtol=1e-5, atol=scripted_fn_atol)


def cache(fn):
    """Similar to :func:`functools.cache` (Python >= 3.8) or :func:`functools.lru_cache` with infinite cache size,
    but this also caches exceptions.
    """
    sentinel = object()
    out_cache = {}
    exc_tb_cache = {}

    @functools.wraps(fn)
    def wrapper(*args, **kwargs):
        key = args + tuple(kwargs.values())

        out = out_cache.get(key, sentinel)
        if out is not sentinel:
            return out

        exc_tb = exc_tb_cache.get(key, sentinel)
        if exc_tb is not sentinel:
            raise exc_tb[0].with_traceback(exc_tb[1])

        try:
            out = fn(*args, **kwargs)
        except Exception as exc:
            # We need to cache the traceback here as well. Otherwise, each re-raise will add the internal pytest
            # traceback frames anew, but they will only be removed once. Thus, the traceback will be ginormous hiding
            # the actual information in the noise. See https://github.com/pytest-dev/pytest/issues/10363 for details.
            exc_tb_cache[key] = exc, exc.__traceback__
            raise exc

        out_cache[key] = out
        return out

    return wrapper


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
    dtypes=(torch.float32, torch.float64, torch.uint8),
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
    dtypes=(torch.float32, torch.float64, torch.int64),
):
    for params in combinations_grid(extra_dims=extra_dims, format=formats, dtype=dtypes):
        yield make_bounding_box_loader(**params, spatial_size=spatial_size)


make_bounding_boxes = from_loaders(make_bounding_box_loaders)


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


def assert_run_python_script(source_code):
    """Utility to check assertions in an independent Python subprocess.

    The script provided in the source code should return 0 and not print
    anything on stderr or stdout. Modified from scikit-learn test utils.

    Args:
        source_code (str): The Python source code to execute.
    """
    with get_tmp_dir() as root:
        path = pathlib.Path(root) / "main.py"
        with open(path, "w") as file:
            file.write(source_code)

        try:
            out = check_output([sys.executable, str(path)], stderr=STDOUT)
        except CalledProcessError as e:
            raise RuntimeError(f"script errored with output:\n{e.output.decode()}")
        if out != b"":
            raise AssertionError(out.decode())
