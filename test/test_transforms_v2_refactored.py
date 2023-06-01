import inspect
import re
from typing import get_type_hints
from unittest import mock

import PIL.Image
import pytest

import torch
import torchvision.transforms.v2 as transforms
from common_utils import (
    cache,
    cpu_and_gpu,
    make_bounding_box,
    make_detection_mask,
    make_image,
    make_segmentation_mask,
    make_video,
)
from torch.testing import assert_close
from torchvision import datapoints
from torchvision.transforms.v2 import functional as F


def _to_tolerances(maybe_tolerance_dict):
    if not isinstance(maybe_tolerance_dict, dict):
        return dict(rtol=None, atol=None)
    return dict(rtol=0, atol=0, **maybe_tolerance_dict)


def _check_kernel_cuda_vs_cpu(kernel, input, *args, rtol, atol, **kwargs):
    if input.device.type != "cuda":
        return

    input_cuda = input.as_subclass(torch.Tensor)
    input_cpu = input_cuda.to("cpu")

    actual = kernel(input_cuda, *args, **kwargs)
    expected = kernel(input_cpu, *args, **kwargs)

    assert_close(actual, expected, check_device=False, rtol=rtol, atol=atol)


@cache
def _script(fn):
    try:
        return torch.jit.script(fn)
    except Exception as error:
        raise AssertionError(f"Trying to `torch.jit.script` '{fn.__name__}' raised the error above.") from error


def _check_kernel_scripted_vs_eager(kernel, input, *args, rtol, atol, **kwargs):
    if input.device.type != "cpu":
        return

    kernel_scripted = _script(kernel)

    input = input.as_subclass(torch.Tensor)
    actual = kernel_scripted(input, *args, **kwargs)
    expected = kernel(input, *args, **kwargs)

    assert_close(actual, expected, rtol=rtol, atol=atol)


def _check_kernel_batched_vs_single(kernel, input, *args, rtol, atol, **kwargs):
    single_input = input.as_subclass(torch.Tensor)

    for batch_dims in [(4,), (2, 3)]:
        repeats = [*batch_dims, *[1] * input.ndim]

        actual = kernel(single_input.repeat(repeats), *args, **kwargs)

        expected = kernel(single_input, *args, **kwargs)
        # We can't directly call `.repeat()` on the output, since some kernel also return some additional metadata
        if isinstance(expected, torch.Tensor):
            expected = expected.repeat(repeats)
        else:
            tensor, *metadata = expected
            expected = (tensor.repeat(repeats), *metadata)

        assert_close(actual, expected, rtol=rtol, atol=atol)

    for degenerate_batch_dims in [(0,), (5, 0), (0, 5)]:
        degenerate_batched_input = torch.empty(
            degenerate_batch_dims + input.shape, dtype=input.dtype, device=input.device
        )

        output = kernel(degenerate_batched_input, *args, **kwargs)
        # Most kernels just return a tensor, but some also return some additional metadata
        if not isinstance(output, torch.Tensor):
            output, *_ = output

        assert output.shape[: -input.ndim] == degenerate_batch_dims


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

    if check_cuda_vs_cpu:
        _check_kernel_cuda_vs_cpu(kernel, input, *args, **kwargs, **_to_tolerances(check_cuda_vs_cpu))

    if check_scripted_vs_eager:
        _check_kernel_scripted_vs_eager(kernel, input, *args, **kwargs, **_to_tolerances(check_scripted_vs_eager))

    if check_batched_vs_single:
        _check_kernel_batched_vs_single(kernel, input, *args, **kwargs, **_to_tolerances(check_batched_vs_single))


def _check_dispatcher_scripted_smoke(dispatcher, input, *args, **kwargs):
    if not isinstance(input, datapoints.Image):
        return

    dispatcher_scripted = _script(dispatcher)
    dispatcher_scripted(input.as_subclass(torch.Tensor), *args, **kwargs)


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
    check_scripted_smoke=True,
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

    if check_scripted_smoke:
        _check_dispatcher_scripted_smoke(dispatcher, input, *args, **kwargs)

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


# We cannot use `list(transforms.InterpolationMode)` here, since it includes some PIL-only ones as well
INTERPOLATION_MODES = [
    transforms.InterpolationMode.NEAREST,
    transforms.InterpolationMode.NEAREST_EXACT,
    transforms.InterpolationMode.BILINEAR,
    transforms.InterpolationMode.BICUBIC,
]


class TestResize:
    SPATIAL_SIZE = (17, 11)
    SIZES = [17, [17], (17,), [12, 13], (12, 13)]

    def _make_max_size_kwarg(self, *, use_max_size, size):
        if use_max_size:
            if not (isinstance(size, int) or len(size) == 1):
                # This would result in an `ValueError`
                return None

            max_size = (size if isinstance(size, int) else size[0]) + 1
        else:
            max_size = None

        return dict(max_size=max_size)

    @pytest.mark.parametrize("size", SIZES)
    @pytest.mark.parametrize("interpolation", INTERPOLATION_MODES)
    @pytest.mark.parametrize("use_max_size", [True, False])
    @pytest.mark.parametrize("antialias", [True, False])
    @pytest.mark.parametrize("dtype", [torch.float32, torch.uint8])
    @pytest.mark.parametrize("device", cpu_and_gpu())
    def test_kernel_image_tensor(self, size, interpolation, use_max_size, antialias, dtype, device):
        if not (max_size_kwarg := self._make_max_size_kwarg(use_max_size=use_max_size, size=size)):
            return

        image = make_image(size=self.SPATIAL_SIZE, dtype=dtype, device=device)
        check_kernel(
            F.resize_image_tensor,
            image,
            size=size,
            interpolation=interpolation,
            **max_size_kwarg,
            antialias=antialias,
            check_scripted_vs_eager=not isinstance(size, int),
        )

    @pytest.mark.parametrize("size", SIZES)
    @pytest.mark.parametrize("format", list(datapoints.BoundingBoxFormat))
    @pytest.mark.parametrize("use_max_size", [True, False])
    @pytest.mark.parametrize("dtype", [torch.float32, torch.int64])
    @pytest.mark.parametrize("device", cpu_and_gpu())
    def test_kernel_bounding_box(self, size, format, use_max_size, dtype, device):
        if not (max_size_kwarg := self._make_max_size_kwarg(use_max_size=use_max_size, size=size)):
            return

        bounding_box = make_bounding_box(format=format, spatial_size=self.SPATIAL_SIZE, dtype=dtype, device=device)
        check_kernel(
            F.resize_bounding_box,
            bounding_box,
            spatial_size=bounding_box.spatial_size,
            size=size,
            **max_size_kwarg,
            check_scripted_vs_eager=not isinstance(size, int),
        )

    @pytest.mark.parametrize(
        "dtype_and_make_mask", [(torch.uint8, make_segmentation_mask), (torch.bool, make_detection_mask)]
    )
    def test_kernel_mask(self, dtype_and_make_mask):
        dtype, make_mask = dtype_and_make_mask
        mask = make_mask(size=self.SPATIAL_SIZE, dtype=dtype)
        check_kernel(F.resize_mask, mask, size=self.SIZES[-1])

    def test_kernel_video(self):
        video = make_video(size=self.SPATIAL_SIZE)
        check_kernel(F.resize_video, video, size=self.SIZES[-1])

    def _make_dispatcher_input(self, kernel):
        if kernel is F.resize_image_tensor:
            input = make_image(size=self.SPATIAL_SIZE)
        elif kernel is F.resize_bounding_box:
            input = make_bounding_box(format=datapoints.BoundingBoxFormat.XYXY, spatial_size=self.SPATIAL_SIZE)
        elif kernel is F.resize_mask:
            input = make_segmentation_mask(size=self.SPATIAL_SIZE)
        elif kernel is F.resize_video:
            input = make_video(size=self.SPATIAL_SIZE)

        return input

    @pytest.mark.parametrize("kernel", [F.resize_image_tensor, F.resize_bounding_box, F.resize_mask, F.resize_video])
    @pytest.mark.parametrize("size", SIZES)
    def test_dispatcher(self, kernel, size):
        check_dispatcher(
            F.resize,
            kernel,
            self._make_dispatcher_input(kernel),
            size=size,
            check_scripted_smoke=not isinstance(size, int),
        )

    @pytest.mark.parametrize(
        ("datapoint_type", "kernel"),
        [
            (datapoints.Image, F.resize_image_tensor),
            (datapoints.BoundingBox, F.resize_bounding_box),
            (datapoints.Mask, F.resize_mask),
            (datapoints.Video, F.resize_video),
        ],
    )
    def test_dispatcher_signature(self, kernel, datapoint_type):
        check_dispatcher_signatures_match(F.resize, kernel=kernel, datapoint_type=datapoint_type)

    def test_transform(self):
        pass
