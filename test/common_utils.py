import contextlib
import functools
import itertools
import os
import pathlib
import random
import re
import shutil
import sys
import tempfile
import warnings
from subprocess import CalledProcessError, check_output, STDOUT

import numpy as np
import PIL.Image
import pytest
import torch
import torch.testing
from PIL import Image

from torch.testing._comparison import BooleanPair, NonePair, not_close_error_metas, NumberPair, TensorLikePair
from torchvision import io, tv_tensors
from torchvision.transforms._functional_tensor import _max_value as get_max_value
from torchvision.transforms.v2.functional import to_image, to_pil_image


IN_OSS_CI = any(os.getenv(var) == "true" for var in ["CIRCLECI", "GITHUB_ACTIONS"])
IN_RE_WORKER = os.environ.get("INSIDE_RE_WORKER") is not None
IN_FBCODE = os.environ.get("IN_FBCODE_TORCHVISION") == "1"
CUDA_NOT_AVAILABLE_MSG = "CUDA device not available"
MPS_NOT_AVAILABLE_MSG = "MPS device not available"
OSS_CI_GPU_NO_CUDA_MSG = "We're in an OSS GPU machine, and this test doesn't need cuda."


@contextlib.contextmanager
def get_tmp_dir(src=None, **kwargs):
    with tempfile.TemporaryDirectory(
        **kwargs,
    ) as tmp_dir:
        if src is not None:
            shutil.copytree(src, tmp_dir)
        yield tmp_dir


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


def cpu_and_cuda():
    import pytest  # noqa

    return ("cpu", pytest.param("cuda", marks=pytest.mark.needs_cuda))


def cpu_and_cuda_and_mps():
    return cpu_and_cuda() + (pytest.param("mps", marks=pytest.mark.needs_mps),)


def needs_cuda(test_func):
    import pytest  # noqa

    return pytest.mark.needs_cuda(test_func)


def needs_mps(test_func):
    import pytest  # noqa

    return pytest.mark.needs_mps(test_func)


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
            actual, expected = [to_image(input) for input in [actual, expected]]

        super().__init__(actual, expected, **other_parameters)
        self.mae = mae

    def compare(self) -> None:
        actual, expected = self.actual, self.expected

        self._compare_attributes(actual, expected)
        actual, expected = self._equalize_attributes(actual, expected)

        if self.mae:
            if actual.dtype is torch.uint8:
                actual, expected = actual.to(torch.int), expected.to(torch.int)
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
    num_boxes=1,
    dtype=None,
    device="cpu",
):
    def sample_position(values, max_value):
        # We cannot use torch.randint directly here, because it only allows integer scalars as values for low and high.
        # However, if we have batch_dims, we need tensors as limits.
        return torch.stack([torch.randint(max_value - v, ()) for v in values.tolist()])

    if isinstance(format, str):
        format = tv_tensors.BoundingBoxFormat[format]

    dtype = dtype or torch.float32

    h, w = [torch.randint(1, s, (num_boxes,)) for s in canvas_size]
    y = sample_position(h, canvas_size[0])
    x = sample_position(w, canvas_size[1])
    r = -360 * torch.rand((num_boxes,)) + 180

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
    elif format is tv_tensors.BoundingBoxFormat.XYWHR:
        parts = (x, y, w, h, r)
    elif format is tv_tensors.BoundingBoxFormat.CXCYWHR:
        cx = x + w / 2
        cy = y + h / 2
        parts = (cx, cy, w, h, r)
    elif format is tv_tensors.BoundingBoxFormat.XYXYXYXY:
        r_rad = r * torch.pi / 180.0
        cos, sin = torch.cos(r_rad), torch.sin(r_rad)
        x1, y1 = x, y
        x3 = x1 + w * cos
        y3 = y1 - w * sin
        x2 = x3 + h * sin
        y2 = y3 + h * cos
        x4 = x1 + h * sin
        y4 = y1 + h * cos
        parts = (x1, y1, x3, y3, x2, y2, x4, y4)
    else:
        raise ValueError(f"Format {format} is not supported")

    return tv_tensors.BoundingBoxes(
        torch.stack(parts, dim=-1).to(dtype=dtype, device=device), format=format, canvas_size=canvas_size
    )


def make_detection_masks(size=DEFAULT_SIZE, *, num_masks=1, dtype=None, device="cpu"):
    """Make a "detection" mask, i.e. (*, N, H, W), where each object is encoded as one of N boolean masks"""
    return tv_tensors.Mask(
        torch.testing.make_tensor(
            (num_masks, *size),
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


@contextlib.contextmanager
def assert_no_warnings():
    # The name `catch_warnings` is a misnomer as the context manager does **not** catch any warnings, but rather scopes
    # the warning filters. All changes that are made to the filters while in this context, will be reset upon exit.
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        yield


@contextlib.contextmanager
def ignore_jit_no_profile_information_warning():
    # Calling a scripted object often triggers a warning like
    # `UserWarning: operator() profile_node %$INT1 : int[] = prim::profile_ivalue($INT2) does not have profile information`
    # with varying `INT1` and `INT2`. Since these are uninteresting for us and only clutter the test summary, we ignore
    # them.
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message=re.escape("operator() profile_node %"), category=UserWarning)
        yield
