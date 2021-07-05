import os
import shutil
import tempfile
import contextlib
import unittest
import argparse
import sys
import torch
import __main__
import random
import inspect

from numbers import Number
from torch._six import string_classes
from collections import OrderedDict

import numpy as np
from PIL import Image

from _assert_utils import assert_equal

IS_PY39 = sys.version_info.major == 3 and sys.version_info.minor == 9
PY39_SEGFAULT_SKIP_MSG = "Segmentation fault with Python 3.9, see https://github.com/pytorch/vision/issues/3367"
PY39_SKIP = unittest.skipIf(IS_PY39, PY39_SEGFAULT_SKIP_MSG)
IN_CIRCLE_CI = os.getenv("CIRCLECI", False) == 'true'
IN_RE_WORKER = os.environ.get("INSIDE_RE_WORKER") is not None
IN_FBCODE = os.environ.get("IN_FBCODE_TORCHVISION") == "1"
CUDA_NOT_AVAILABLE_MSG = 'CUDA device not available'
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
    np.random.seed(seed)


class MapNestedTensorObjectImpl(object):
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


# adapted from TestCase in torch/test/common_utils to accept non-string
# inputs and set maximum binary size
class TestCase(unittest.TestCase):
    precision = 1e-5

    def assertEqual(self, x, y, prec=None, message='', allow_inf=False):
        """
        This is copied from pytorch/test/common_utils.py's TestCase.assertEqual
        """
        if isinstance(prec, str) and message == '':
            message = prec
            prec = None
        if prec is None:
            prec = self.precision

        if isinstance(x, torch.Tensor) and isinstance(y, Number):
            self.assertEqual(x.item(), y, prec=prec, message=message,
                             allow_inf=allow_inf)
        elif isinstance(y, torch.Tensor) and isinstance(x, Number):
            self.assertEqual(x, y.item(), prec=prec, message=message,
                             allow_inf=allow_inf)
        elif isinstance(x, torch.Tensor) and isinstance(y, torch.Tensor):
            def assertTensorsEqual(a, b):
                super(TestCase, self).assertEqual(a.size(), b.size(), message)
                if a.numel() > 0:
                    if (a.device.type == 'cpu' and (a.dtype == torch.float16 or a.dtype == torch.bfloat16)):
                        # CPU half and bfloat16 tensors don't have the methods we need below
                        a = a.to(torch.float32)
                    b = b.to(a)

                    if (a.dtype == torch.bool) != (b.dtype == torch.bool):
                        raise TypeError("Was expecting both tensors to be bool type.")
                    else:
                        if a.dtype == torch.bool and b.dtype == torch.bool:
                            # we want to respect precision but as bool doesn't support substraction,
                            # boolean tensor has to be converted to int
                            a = a.to(torch.int)
                            b = b.to(torch.int)

                        diff = a - b
                        if a.is_floating_point():
                            # check that NaNs are in the same locations
                            nan_mask = torch.isnan(a)
                            self.assertTrue(torch.equal(nan_mask, torch.isnan(b)), message)
                            diff[nan_mask] = 0
                            # inf check if allow_inf=True
                            if allow_inf:
                                inf_mask = torch.isinf(a)
                                inf_sign = inf_mask.sign()
                                self.assertTrue(torch.equal(inf_sign, torch.isinf(b).sign()), message)
                                diff[inf_mask] = 0
                        # TODO: implement abs on CharTensor (int8)
                        if diff.is_signed() and diff.dtype != torch.int8:
                            diff = diff.abs()
                        max_err = diff.max()
                        tolerance = prec + prec * abs(a.max())
                        self.assertLessEqual(max_err, tolerance, message)
            super(TestCase, self).assertEqual(x.is_sparse, y.is_sparse, message)
            super(TestCase, self).assertEqual(x.is_quantized, y.is_quantized, message)
            if x.is_sparse:
                x = self.safeCoalesce(x)
                y = self.safeCoalesce(y)
                assertTensorsEqual(x._indices(), y._indices())
                assertTensorsEqual(x._values(), y._values())
            elif x.is_quantized and y.is_quantized:
                self.assertEqual(x.qscheme(), y.qscheme(), prec=prec,
                                 message=message, allow_inf=allow_inf)
                if x.qscheme() == torch.per_tensor_affine:
                    self.assertEqual(x.q_scale(), y.q_scale(), prec=prec,
                                     message=message, allow_inf=allow_inf)
                    self.assertEqual(x.q_zero_point(), y.q_zero_point(),
                                     prec=prec, message=message,
                                     allow_inf=allow_inf)
                elif x.qscheme() == torch.per_channel_affine:
                    self.assertEqual(x.q_per_channel_scales(), y.q_per_channel_scales(), prec=prec,
                                     message=message, allow_inf=allow_inf)
                    self.assertEqual(x.q_per_channel_zero_points(), y.q_per_channel_zero_points(),
                                     prec=prec, message=message,
                                     allow_inf=allow_inf)
                    self.assertEqual(x.q_per_channel_axis(), y.q_per_channel_axis(),
                                     prec=prec, message=message)
                self.assertEqual(x.dtype, y.dtype)
                self.assertEqual(x.int_repr().to(torch.int32),
                                 y.int_repr().to(torch.int32), prec=prec,
                                 message=message, allow_inf=allow_inf)
            else:
                assertTensorsEqual(x, y)
        elif isinstance(x, string_classes) and isinstance(y, string_classes):
            super(TestCase, self).assertEqual(x, y, message)
        elif type(x) == set and type(y) == set:
            super(TestCase, self).assertEqual(x, y, message)
        elif isinstance(x, dict) and isinstance(y, dict):
            if isinstance(x, OrderedDict) and isinstance(y, OrderedDict):
                self.assertEqual(x.items(), y.items(), prec=prec,
                                 message=message, allow_inf=allow_inf)
            else:
                self.assertEqual(set(x.keys()), set(y.keys()), prec=prec,
                                 message=message, allow_inf=allow_inf)
                key_list = list(x.keys())
                self.assertEqual([x[k] for k in key_list],
                                 [y[k] for k in key_list],
                                 prec=prec, message=message,
                                 allow_inf=allow_inf)
        elif is_iterable(x) and is_iterable(y):
            super(TestCase, self).assertEqual(len(x), len(y), message)
            for x_, y_ in zip(x, y):
                self.assertEqual(x_, y_, prec=prec, message=message,
                                 allow_inf=allow_inf)
        elif isinstance(x, bool) and isinstance(y, bool):
            super(TestCase, self).assertEqual(x, y, message)
        elif isinstance(x, Number) and isinstance(y, Number):
            inf = float("inf")
            if abs(x) == inf or abs(y) == inf:
                if allow_inf:
                    super(TestCase, self).assertEqual(x, y, message)
                else:
                    self.fail("Expected finite numeric values - x={}, y={}".format(x, y))
                return
            super(TestCase, self).assertLessEqual(abs(x - y), prec, message)
        else:
            super(TestCase, self).assertEqual(x, y, message)


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
        for obj2 in objs[:idx] + objs[idx + 1:]:
            yield obj1, obj2


def int_dtypes():
    return torch.testing.integral_types()


def float_dtypes():
    return torch.testing.floating_types()


@contextlib.contextmanager
def disable_console_output():
    with contextlib.ExitStack() as stack, open(os.devnull, "w") as devnull:
        stack.enter_context(contextlib.redirect_stdout(devnull))
        stack.enter_context(contextlib.redirect_stderr(devnull))
        yield


def cpu_and_gpu():
    import pytest  # noqa
    return ('cpu', pytest.param('cuda', marks=pytest.mark.needs_cuda))


def needs_cuda(test_func):
    import pytest  # noqa
    return pytest.mark.needs_cuda(test_func)


def _create_data(height=3, width=3, channels=3, device="cpu"):
    # TODO: When all relevant tests are ported to pytest, turn this into a module-level fixture
    tensor = torch.randint(0, 256, (channels, height, width), dtype=torch.uint8, device=device)
    pil_img = Image.fromarray(tensor.permute(1, 2, 0).contiguous().cpu().numpy())
    return tensor, pil_img


def _create_data_batch(height=3, width=3, channels=3, num_samples=4, device="cpu"):
    # TODO: When all relevant tests are ported to pytest, turn this into a module-level fixture
    batch_tensor = torch.randint(
        0, 256,
        (num_samples, channels, height, width),
        dtype=torch.uint8,
        device=device
    )
    return batch_tensor


def _assert_equal_tensor_to_pil(tensor, pil_image, msg=None):
    np_pil_image = np.array(pil_image)
    if np_pil_image.ndim == 2:
        np_pil_image = np_pil_image[:, :, None]
    pil_tensor = torch.as_tensor(np_pil_image.transpose((2, 0, 1)))
    if msg is None:
        msg = "tensor:\n{} \ndid not equal PIL tensor:\n{}".format(tensor, pil_tensor)
    assert_equal(tensor.cpu(), pil_tensor, check_stride=False, msg=msg)


def _assert_approx_equal_tensor_to_pil(tensor, pil_image, tol=1e-5, msg=None, agg_method="mean",
                                       allowed_percentage_diff=None):
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
    assert err < tol


def _test_fn_on_batch(batch_tensors, fn, scripted_fn_atol=1e-8, **fn_kwargs):
    transformed_batch = fn(batch_tensors, **fn_kwargs)
    for i in range(len(batch_tensors)):
        img_tensor = batch_tensors[i, ...]
        transformed_img = fn(img_tensor, **fn_kwargs)
        assert_equal(transformed_img, transformed_batch[i, ...])

    if scripted_fn_atol >= 0:
        scripted_fn = torch.jit.script(fn)
        # scriptable function test
        s_transformed_batch = scripted_fn(batch_tensors, **fn_kwargs)
        torch.testing.assert_close(transformed_batch, s_transformed_batch, rtol=1e-5, atol=scripted_fn_atol)
