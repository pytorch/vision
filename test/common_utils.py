import os
import shutil
import tempfile
import contextlib
import unittest
import argparse
import sys
import io
import torch
import warnings
import __main__
import random
import inspect

from numbers import Number
from torch._six import string_classes
from collections import OrderedDict
from _utils_internal import get_relative_path

import numpy as np
from PIL import Image

IS_PY39 = sys.version_info.major == 3 and sys.version_info.minor == 9
PY39_SEGFAULT_SKIP_MSG = "Segmentation fault with Python 3.9, see https://github.com/pytorch/vision/issues/3367"
PY39_SKIP = unittest.skipIf(IS_PY39, PY39_SEGFAULT_SKIP_MSG)
IN_CIRCLE_CI = os.getenv("CIRCLECI", False) == 'true'
IN_RE_WORKER = os.environ.get("INSIDE_RE_WORKER") is not None
IN_FBCODE = os.environ.get("IN_FBCODE_TORCHVISION") == "1"
CUDA_NOT_AVAILABLE_MSG = 'CUDA device not available'


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


ACCEPT = os.getenv('EXPECTTEST_ACCEPT', '0') == '1'
TEST_WITH_SLOW = os.getenv('PYTORCH_TEST_WITH_SLOW', '0') == '1'


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

    def _get_expected_file(self, name=None):
        # NB: we take __file__ from the module that defined the test
        # class, so we place the expect directory where the test script
        # lives, NOT where test/common_utils.py lives.
        module_id = self.__class__.__module__

        # Determine expected file based on environment
        expected_file_base = get_relative_path(
            os.path.realpath(sys.modules[module_id].__file__),
            "expect")

        # Note: for legacy reasons, the reference file names all had "ModelTest.test_" in their names
        # We hardcode it here to avoid having to re-generate the reference files
        expected_file = expected_file = os.path.join(expected_file_base, 'ModelTester.test_' + name)
        expected_file += "_expect.pkl"

        if not ACCEPT and not os.path.exists(expected_file):
            raise RuntimeError(
                f"No expect file exists for {os.path.basename(expected_file)} in {expected_file}; "
                "to accept the current output, re-run the failing test after setting the EXPECTTEST_ACCEPT "
                "env variable. For example: EXPECTTEST_ACCEPT=1 pytest test/test_models.py -k alexnet"
            )

        return expected_file

    def assertExpected(self, output, name, prec=None):
        r"""
        Test that a python value matches the recorded contents of a file
        based on a "check" name. The value must be
        pickable with `torch.save`. This file
        is placed in the 'expect' directory in the same directory
        as the test script. You can automatically update the recorded test
        output using an EXPECTTEST_ACCEPT=1 env variable.
        """
        expected_file = self._get_expected_file(name)

        if ACCEPT:
            filename = {os.path.basename(expected_file)}
            print("Accepting updated output for {}:\n\n{}".format(filename, output))
            torch.save(output, expected_file)
            MAX_PICKLE_SIZE = 50 * 1000  # 50 KB
            binary_size = os.path.getsize(expected_file)
            if binary_size > MAX_PICKLE_SIZE:
                raise RuntimeError("The output for {}, is larger than 50kb".format(filename))
        else:
            expected = torch.load(expected_file)
            self.assertEqual(output, expected, prec=prec)

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

    def check_jit_scriptable(self, nn_module, args, unwrapper=None, skip=False):
        """
        Check that a nn.Module's results in TorchScript match eager and that it
        can be exported
        """
        if not TEST_WITH_SLOW or skip:
            # TorchScript is not enabled, skip these tests
            msg = "The check_jit_scriptable test for {} was skipped. " \
                  "This test checks if the module's results in TorchScript " \
                  "match eager and that it can be exported. To run these " \
                  "tests make sure you set the environment variable " \
                  "PYTORCH_TEST_WITH_SLOW=1 and that the test is not " \
                  "manually skipped.".format(nn_module.__class__.__name__)
            warnings.warn(msg, RuntimeWarning)
            return None

        sm = torch.jit.script(nn_module)

        with freeze_rng_state():
            eager_out = nn_module(*args)

        with freeze_rng_state():
            script_out = sm(*args)
            if unwrapper:
                script_out = unwrapper(script_out)

        self.assertEqual(eager_out, script_out, prec=1e-4)
        self.assertExportImportModule(sm, args)

        return sm

    def getExportImportCopy(self, m):
        """
        Save and load a TorchScript model
        """
        buffer = io.BytesIO()
        torch.jit.save(m, buffer)
        buffer.seek(0)
        imported = torch.jit.load(buffer)
        return imported

    def assertExportImportModule(self, m, args):
        """
        Check that the results of a model are the same after saving and loading
        """
        m_import = self.getExportImportCopy(m)
        with freeze_rng_state():
            results = m(*args)
        with freeze_rng_state():
            results_from_imported = m_import(*args)
        self.assertEqual(results, results_from_imported, prec=3e-5)


@contextlib.contextmanager
def freeze_rng_state():
    rng_state = torch.get_rng_state()
    if torch.cuda.is_available():
        cuda_rng_state = torch.cuda.get_rng_state()
    yield
    if torch.cuda.is_available():
        torch.cuda.set_rng_state(cuda_rng_state)
    torch.set_rng_state(rng_state)


class TransformsTester(unittest.TestCase):

    def _create_data(self, height=3, width=3, channels=3, device="cpu"):
        tensor = torch.randint(0, 256, (channels, height, width), dtype=torch.uint8, device=device)
        pil_img = Image.fromarray(tensor.permute(1, 2, 0).contiguous().cpu().numpy())
        return tensor, pil_img

    def _create_data_batch(self, height=3, width=3, channels=3, num_samples=4, device="cpu"):
        batch_tensor = torch.randint(
            0, 256,
            (num_samples, channels, height, width),
            dtype=torch.uint8,
            device=device
        )
        return batch_tensor

    def compareTensorToPIL(self, tensor, pil_image, msg=None):
        np_pil_image = np.array(pil_image)
        if np_pil_image.ndim == 2:
            np_pil_image = np_pil_image[:, :, None]
        pil_tensor = torch.as_tensor(np_pil_image.transpose((2, 0, 1)))
        if msg is None:
            msg = "tensor:\n{} \ndid not equal PIL tensor:\n{}".format(tensor, pil_tensor)
        self.assertTrue(tensor.cpu().equal(pil_tensor), msg)

    def approxEqualTensorToPIL(self, tensor, pil_image, tol=1e-5, msg=None, agg_method="mean",
                               allowed_percentage_diff=None):
        np_pil_image = np.array(pil_image)
        if np_pil_image.ndim == 2:
            np_pil_image = np_pil_image[:, :, None]
        pil_tensor = torch.as_tensor(np_pil_image.transpose((2, 0, 1))).to(tensor)

        if allowed_percentage_diff is not None:
            # Assert that less than a given %age of pixels are different
            self.assertTrue(
                (tensor != pil_tensor).to(torch.float).mean() <= allowed_percentage_diff
            )
        # error value can be mean absolute error, max abs error
        # Convert to float to avoid underflow when computing absolute difference
        tensor = tensor.to(torch.float)
        pil_tensor = pil_tensor.to(torch.float)
        err = getattr(torch, agg_method)(torch.abs(tensor - pil_tensor)).item()
        self.assertTrue(
            err < tol,
            msg="{}: err={}, tol={}: \n{}\nvs\n{}".format(msg, err, tol, tensor[0, :10, :10], pil_tensor[0, :10, :10])
        )


def cycle_over(objs):
    for idx, obj in enumerate(objs):
        yield obj, objs[:idx] + objs[idx + 1:]


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


def call_args_to_kwargs_only(call_args, *callable_or_arg_names):
    callable_or_arg_name = callable_or_arg_names[0]
    if callable(callable_or_arg_name):
        argspec = inspect.getfullargspec(callable_or_arg_name)
        arg_names = argspec.args
        if isinstance(callable_or_arg_name, type):
            # remove self
            arg_names.pop(0)
    else:
        arg_names = callable_or_arg_names

    args, kwargs = call_args
    kwargs_only = kwargs.copy()
    kwargs_only.update(dict(zip(arg_names, args)))
    return kwargs_only


def cpu_and_gpu():
    import pytest  # noqa

    # ignore CPU tests in RE as they're already covered by another contbuild
    devices = [] if IN_RE_WORKER else ['cpu']

    if torch.cuda.is_available():
        cuda_marks = ()
    elif IN_FBCODE:
        # Dont collect cuda tests on fbcode if the machine doesnt have a GPU
        # This avoids skipping the tests. More robust would be to detect if
        # we're in sancastle instead of fbcode?
        cuda_marks = pytest.mark.dont_collect()
    else:
        cuda_marks = pytest.mark.skip(reason=CUDA_NOT_AVAILABLE_MSG)

    devices.append(pytest.param('cuda', marks=cuda_marks))

    return devices


def needs_cuda(test_func):
    import pytest  # noqa

    if IN_FBCODE and not IN_RE_WORKER:
        # We don't want to skip in fbcode, so we just don't collect
        # TODO: slightly more robust way would be to detect if we're in a sandcastle instance
        # so that the test will still be collected (and skipped) in the devvms.
        return pytest.mark.dont_collect(test_func)
    elif torch.cuda.is_available():
        return test_func
    else:
        return pytest.mark.skip(reason=CUDA_NOT_AVAILABLE_MSG)(test_func)
