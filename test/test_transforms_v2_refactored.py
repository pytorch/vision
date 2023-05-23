import inspect
import re
from typing import get_type_hints
from unittest import mock

import PIL.Image
import pytest

import torch
from common_utils import cache, cpu_and_gpu, make_bounding_box, make_image
from torch.testing import assert_close
from torch.utils._pytree import tree_map
from torchvision import datapoints
from torchvision.transforms.v2 import functional as F
from torchvision.transforms.v2.utils import is_simple_tensor


def _to_tolerances(maybe_tolerance_dict):
    default_tolerances = dict(rtol=None, atol=None)
    if not isinstance(maybe_tolerance_dict, dict):
        return default_tolerances

    missing = default_tolerances.keys() - maybe_tolerance_dict.keys()
    if missing:
        raise pytest.UsageError("ADDME")

    extra = maybe_tolerance_dict.keys() - default_tolerances.keys()
    if extra:
        raise pytest.UsageError("ADDME")

    return maybe_tolerance_dict


def _check_kernel_cuda_vs_cpu(kernel, tolerances, input, *args, **kwargs):
    if input.device.type != "cuda":
        return

    input_cuda = input.as_subclass(torch.Tensor)
    input_cpu = input_cuda.to("cpu")

    actual = kernel(input_cuda, *args, **kwargs)
    expected = kernel(input_cpu, *args, **kwargs)

    assert_close(actual, expected, **tolerances, check_device=False)


@cache
def _script(fn):
    try:
        return torch.jit.script(fn)
    except Exception as error:
        raise AssertionError(f"Trying to `torch.jit.script` '{fn.__name__}' raised the error above.") from error


def _check_kernel_scripted_vs_eager(kernel, tolerances, input, *args, **kwargs):
    kernel_scripted = _script(kernel)

    input = input.as_subclass(torch.Tensor)
    actual = kernel_scripted(input, *args, **kwargs)
    expected = kernel(input, *args, **kwargs)

    assert_close(actual, expected, **tolerances)


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


def _check_kernel_batched_vs_single(kernel, tolerances, input, *args, **kwargs):
    input_type = datapoints.Image if is_simple_tensor(input) else type(input)
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
        )
    elif input.ndim <= data_dims or not all(input.shape[:-data_dims]):
        # input is not batched or has a degenerate batch shape
        return

    batched_input = input.as_subclass(torch.Tensor)
    batched_output = kernel(batched_input, *args, **kwargs)
    actual = _unbatch(batched_output, data_dims=data_dims)

    single_inputs = _unbatch(batched_input, data_dims=data_dims)
    expected = tree_map(lambda single_input: kernel(single_input, *args, **kwargs), single_inputs)

    assert_close(actual, expected, **tolerances)


def check_kernel(
    kernel,
    input,
    *args,
    check_cuda_vs_cpu=True,
    check_scripted_vs_eager=True,
    check_batched_vs_single=True,
    **kwargs,
):
    initial_input_version = input._version

    output = kernel(input.as_subclass(torch.Tensor), *args, **kwargs)
    # Most kernels just return a tensor, but some also return some additional metadata
    if not isinstance(output, torch.Tensor):
        output, *_ = output

    # check that no inplace operation happened
    assert input._version == initial_input_version

    assert output.dtype == input.dtype
    assert output.device == input.device

    # TODO: we can improve performance here by not computing the regular output of the kernel over and over
    if check_cuda_vs_cpu:
        _check_kernel_cuda_vs_cpu(kernel, _to_tolerances(check_cuda_vs_cpu), input, *args, **kwargs)

    if check_scripted_vs_eager:
        _check_kernel_scripted_vs_eager(kernel, _to_tolerances(check_scripted_vs_eager), input, *args, **kwargs)

    if check_batched_vs_single:
        _check_kernel_batched_vs_single(kernel, _to_tolerances(check_batched_vs_single), input, *args, **kwargs)


def _check_dispatcher_dispatch_simple_tensor(dispatcher, kernel, input, *args, **kwargs):
    if not isinstance(input, datapoints.Image):
        return

    with mock.patch(f"{dispatcher.__module__}.{kernel.__name__}", wraps=kernel) as spy:
        output = dispatcher(input.as_subclass(torch.Tensor), *args, **kwargs)

        spy.assert_called_once()

    # We cannot use `isinstance` here since all datapoints are instances of `torch.Tensor` as well
    assert type(output) is torch.Tensor


def _check_dispatcher_dispatch_datapoint(dispatcher, kernel, input, *args, **kwargs):
    # Due to our complex dispatch architecture for datapoints, we cannot spy on the kernel directly,
    # but rather have to patch the `Datapoint.__F` attribute to contain the spied on kernel.
    spy = mock.MagicMock(wraps=kernel)
    with mock.patch.object(F, kernel.__name__, spy):
        # Due to Python's name mangling, the `Datapoint.__F` attribute is only accessible from inside the class.
        # Since that is not the case here, we need to prefix f"_{cls.__name__}"
        # See https://docs.python.org/3/tutorial/classes.html#private-variables for details
        with mock.patch.object(datapoints._datapoint.Datapoint, "_Datapoint__F", new=F):
            output = dispatcher(input, *args, **kwargs)

    spy.assert_called_once()
    assert isinstance(output, type(input))

    if isinstance(input, datapoints.BoundingBox):
        assert output.format == input.format


def _check_dispatcher_dispatch_pil(dispatcher, input, *args, **kwargs):
    if not (isinstance(input, datapoints.Image) and input.dtype is torch.uint8):
        return

    kernel = getattr(F, f"{dispatcher.__name__}_image_pil")

    with mock.patch(f"{dispatcher.__module__}.{kernel.__name__}", wraps=kernel) as spy:
        output = dispatcher(F.to_image_pil(input), *args, **kwargs)

        spy.assert_called_once()

    assert isinstance(output, PIL.Image.Image)


def check_dispatcher(
    dispatcher,
    kernel,
    input,
    *args,
    check_dispatch_simple_tensor=True,
    check_dispatch_datapoint=True,
    check_dispatch_pil=True,
    **kwargs,
):
    with mock.patch("torch._C._log_api_usage_once", wraps=torch._C._log_api_usage_once) as spy:
        dispatcher(input, *args, **kwargs)

        spy.assert_any_call(f"{dispatcher.__module__}.{dispatcher.__name__}")

    unknown_input = object()
    with pytest.raises(TypeError, match=re.escape(str(type(unknown_input)))):
        dispatcher(unknown_input, *args, **kwargs)

    dispatcher_scripted = _script(dispatcher)
    dispatcher_scripted(input.as_subclass(torch.Tensor), *args, **kwargs)

    if check_dispatch_simple_tensor:
        _check_dispatcher_dispatch_simple_tensor(dispatcher, kernel, input, *args, **kwargs)

    if check_dispatch_datapoint:
        _check_dispatcher_dispatch_datapoint(dispatcher, kernel, input, *args, **kwargs)

    if check_dispatch_pil:
        _check_dispatcher_dispatch_pil(dispatcher, input, *args, **kwargs)


def _check_dispatcher_kernel_signature_match(dispatcher, *, kernel, datapoint_type):
    dispatcher_signature = inspect.signature(dispatcher)
    dispatcher_params = list(dispatcher_signature.parameters.values())[1:]

    kernel_signature = inspect.signature(kernel)
    kernel_params = list(kernel_signature.parameters.values())[1:]

    # We filter out metadata that is implicitly passed to the dispatcher through the input datapoint, but has to be
    # explicit passed to the kernel.
    kernel_params = [param for param in kernel_params if param.name not in datapoint_type.__annotations__.keys()]

    dispatcher_params = iter(dispatcher_params)
    for dispatcher_param, kernel_param in zip(dispatcher_params, kernel_params):
        try:
            # In general, the dispatcher parameters are a superset of the kernel parameters. Thus, we filter out
            # dispatcher parameters that have no kernel equivalent while keeping the order intact.
            while dispatcher_param.name != kernel_param.name:
                dispatcher_param = next(dispatcher_params)
        except StopIteration:
            raise AssertionError(
                f"Parameter `{kernel_param.name}` of kernel `{kernel.__name__}` "
                f"has no corresponding parameter on the dispatcher `{dispatcher.__name__}`."
            ) from None

        assert dispatcher_param == kernel_param


def _check_dispatcher_datapoint_signature_match(dispatcher):
    dispatcher_signature = inspect.signature(dispatcher)
    dispatcher_params = list(dispatcher_signature.parameters.values())[1:]

    datapoint_method = getattr(datapoints._datapoint.Datapoint, dispatcher.__name__)
    datapoint_signature = inspect.signature(datapoint_method)
    datapoint_params = list(datapoint_signature.parameters.values())[1:]

    # Because we use `from __future__ import annotations` inside the module where `datapoints._datapoint` is
    # defined, the annotations are stored as strings. This makes them concrete again, so they can be compared to the
    # natively concrete dispatcher annotations.
    datapoint_annotations = get_type_hints(datapoint_method)
    for param in datapoint_params:
        param._annotation = datapoint_annotations[param.name]

    assert dispatcher_params == datapoint_params


def check_dispatcher_signatures_match(dispatcher, *, kernel, datapoint_type):
    _check_dispatcher_kernel_signature_match(dispatcher, kernel=kernel, datapoint_type=datapoint_type)
    _check_dispatcher_datapoint_signature_match(dispatcher)


def check_transform():
    pass


class TestResize:
    @pytest.mark.parametrize("size", [(11, 17), (15, 13)])
    @pytest.mark.parametrize("antialias", [True, False])
    @pytest.mark.parametrize("dtype", [torch.float32, torch.uint8])
    @pytest.mark.parametrize("device", cpu_and_gpu())
    def test_kernel_image_tensor(self, size, antialias, dtype, device):
        image = make_image(size=(14, 16), dtype=dtype, device=device)
        check_kernel(F.resize_image_tensor, image, size=size, antialias=antialias)

    @pytest.mark.parametrize("size", [(11, 17), (15, 13)])
    @pytest.mark.parametrize("format", list(datapoints.BoundingBoxFormat))
    @pytest.mark.parametrize("dtype", [torch.float32, torch.int64])
    @pytest.mark.parametrize("device", cpu_and_gpu())
    def test_kernel_bounding_box(self, size, format, dtype, device):
        spatial_size = (14, 16)
        bounding_box = make_bounding_box(format=format, spatial_size=spatial_size, dtype=dtype, device=device)
        check_kernel(F.resize_bounding_box, bounding_box, spatial_size=spatial_size, size=size)

    @pytest.mark.parametrize("kernel", [F.resize_image_tensor, F.resize_bounding_box])
    @pytest.mark.parametrize("size", [(11, 17), (15, 13)])
    @pytest.mark.parametrize("antialias", [True, False])
    @pytest.mark.parametrize("dtype", [torch.float32, torch.uint8, torch.int64])
    @pytest.mark.parametrize("device", cpu_and_gpu())
    def test_dispatcher(self, kernel, size, antialias, dtype, device):
        spatial_size = (14, 16)
        if kernel is F.resize_image_tensor:
            input = make_image(size=spatial_size, dtype=dtype, device=device)
        elif kernel is F.resize_bounding_box:
            input = make_bounding_box(
                format=datapoints.BoundingBoxFormat.XYXY, spatial_size=spatial_size, dtype=dtype, device=device
            )

        check_dispatcher(F.resize, kernel, input, size=size, antialias=antialias)

    @pytest.mark.parametrize(
        ("kernel", "datapoint_type"),
        [
            (F.resize_image_tensor, datapoints.Image),
            (F.resize_bounding_box, datapoints.BoundingBox),
        ],
    )
    def test_dispatcher_signature(self, kernel, datapoint_type):
        check_dispatcher_signatures_match(F.resize, kernel=kernel, datapoint_type=datapoint_type)

    def test_transform(self):
        pass
