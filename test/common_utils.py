import contextlib
import functools
import os
import random
import shutil
import tempfile

import numpy as np
import torch
from PIL import Image
from torchvision import io

import __main__  # noqa: 401


IN_CIRCLE_CI = os.getenv("CIRCLECI", False) == "true"
IN_RE_WORKER = os.environ.get("INSIDE_RE_WORKER") is not None
IN_FBCODE = os.environ.get("IN_FBCODE_TORCHVISION") == "1"
CUDA_NOT_AVAILABLE_MSG = "CUDA device not available"
CIRCLECI_GPU_NO_CUDA_MSG = "We're in a CircleCI GPU machine, and this test doesn't need cuda."


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


assert_equal = functools.partial(torch.testing.assert_close, rtol=0, atol=0)


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
