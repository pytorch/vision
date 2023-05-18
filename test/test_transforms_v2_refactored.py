from unittest import mock

import pytest
import torch

from common_utils import cache, cpu_and_gpu
from torch.testing import assert_close
from torch.utils._pytree import tree_map
from torchvision import datapoints

from torchvision.transforms.v2 import functional as F
from torchvision.transforms.v2.utils import is_simple_tensor


def _check_kernel_smoke(kernel, input, *args, **kwargs):
    initial_input_version = input._version

    output = kernel(input.as_subclass(torch.Tensor), *args, **kwargs)

    # check that no inplace operation happened
    assert input._version == initial_input_version

    assert output.dtype == input.dtype
    assert output.device == input.device


def _check_logging(fn, *args, **kwargs):
    with mock.patch("torch._C._log_api_usage_once", wraps=torch._C._log_api_usage_once) as spy:
        fn(*args, **kwargs)

        spy.assert_any_call(f"{fn.__module__}.{fn.__name__}")


def _check_kernel_cuda_vs_cpu(kernel, input_cuda, *other_args, **kwargs):
    input_cuda = input_cuda.as_subclass(torch.Tensor)
    input_cpu = input_cuda.to("cpu")

    actual = kernel(input_cuda, *other_args, **kwargs)
    expected = kernel(input_cpu, *other_args, **kwargs)

    assert_close(actual, expected)


@cache
def _script(fn):
    try:
        return torch.jit.script(fn)
    except Exception as error:
        raise AssertionError(f"Trying to `torch.jit.script` '{fn.__name__}' raised the error above.") from error


def _check_kernel_scripted_vs_eager(kernel_eager, input, *other_args, **kwargs):
    kernel_scripted = _script(kernel_eager)

    input = input.as_subclass(torch.Tensor)
    actual = kernel_scripted(input, *other_args, **kwargs)
    expected = kernel_eager(input, *other_args, **kwargs)

    assert_close(actual, expected)


def _unbatch(batch, *, data_dims):
    if isinstance(batch, torch.Tensor):
        batched_tensor = batch
        metadata = ()
    else:
        batched_tensor, *metadata = batch

    if batched_tensor.ndim == data_dims:
        return batch

    return [
        _unbatch(unbatched, data_dims=data_dims)
        for unbatched in (
            batched_tensor.unbind(0) if not metadata else [(t, *metadata) for t in batched_tensor.unbind(0)]
        )
    ]


def _check_kernel_batched_vs_single(kernel, batched_input, *other_args, **kwargs):
    input_type = datapoints.Image if is_simple_tensor(batched_input) else type(batched_input)
    # This dictionary contains the number of rightmost dimensions that contain the actual data.
    # Everything to the left is considered a batch dimension.
    data_dims = {
        datapoints.Image: 3,
        datapoints.BoundingBox: 1,
        # `Mask`'s are special in the sense that the data dimensions depend on the type of mask. For detection masks
        # it is 3 `(*, N, H, W)`, but for segmentation masks it is 2 `(*, H, W)`. Since both a grouped under one
        # type all kernels should also work without differentiating between the two. Thus, we go with 2 here as
        # common ground.
        datapoints.Mask: 2,
        datapoints.Video: 4,
    }.get(input_type)
    if data_dims is None:
        raise pytest.UsageError(
            f"The number of data dimensions cannot be determined for input of type {input_type.__name__}."
        ) from None
    elif batched_input.ndim <= data_dims or not all(batched_input.shape[:-data_dims]):
        # input is not batched or has a degenerate batch shape
        return

    batched_input = batched_input.as_subclass(torch.Tensor)
    batched_output = kernel(batched_input, *other_args, **kwargs)
    actual = _unbatch(batched_output, data_dims=data_dims)

    single_inputs = _unbatch(batched_input, data_dims=data_dims)
    expected = tree_map(lambda single_input: kernel(single_input, *other_args, **kwargs), single_inputs)

    assert_close(actual, expected)


def check_kernel(
    kernel,
    input,
    *other_kernel_args,
    # Most kernels don't log because that is done through the dispatcher. Meaning if we set the default to True,
    # we'll have to set it to False in almost any test. That would be more explicit though
    check_logging=False,
    check_cuda_vs_cpu=True,
    check_scripted_vs_eager=True,
    check_batched_vs_single=True,
    **kernel_kwargs,
    # TODO: tolerances!
):
    # TODO: we can improve performance here by not computing the regular output of the kernel over and over

    _check_kernel_smoke(kernel, input, *other_kernel_args, **kernel_kwargs)

    if check_logging:
        # We need to unwrap the input here manually, because `_check_logging` is not only used for kernels and thus
        # cannot do this internally
        _check_logging(kernel, input.as_subclass(torch.Tensor), *other_kernel_args, **kernel_kwargs)

    if check_cuda_vs_cpu and input.device.type == "cuda":
        _check_kernel_cuda_vs_cpu(kernel, input, *other_kernel_args, **kernel_kwargs)

    if check_scripted_vs_eager:
        _check_kernel_scripted_vs_eager(kernel, input, *other_kernel_args, **kernel_kwargs)

    if check_batched_vs_single:
        _check_kernel_batched_vs_single(kernel, input, *other_kernel_args, **kernel_kwargs)


def check_dispatcher():
    pass


def check_transform():
    pass


class TestResize:
    @pytest.mark.parametrize("size", [(11, 17), (15, 13)])
    @pytest.mark.parametrize("antialias", [True, False])
    @pytest.mark.parametrize("device", cpu_and_gpu())
    def test_resize_image_tensor(self, size, antialias, device):
        image = torch.rand((3, 14, 16), dtype=torch.float32, device=device)
        check_kernel(F.resize_image_tensor, image, size=size, antialias=antialias)

    def test_resize_bounding_box(self):
        pass

    def test_resize(self):
        pass

    def test_Resize(self):
        pass
