import contextlib
import functools
import inspect
import re
from typing import get_type_hints
from unittest import mock

import PIL.Image
import pytest

import torch
import torchvision.transforms.v2 as transforms
from common_utils import (
    assert_equal,
    assert_no_warnings,
    cache,
    cpu_and_gpu,
    ignore_jit_no_profile_information_warning,
    make_bounding_box,
    make_detection_mask,
    make_image,
    make_segmentation_mask,
    make_video,
)
from torch.testing import assert_close
from torchvision import datapoints
from torchvision.transforms.functional import pil_modes_mapping
from torchvision.transforms.v2 import functional as F


def _to_tolerances(maybe_tolerance_dict):
    if not isinstance(maybe_tolerance_dict, dict):
        return dict(rtol=None, atol=None)

    tolerances = dict(rtol=0, atol=0)
    tolerances.update(maybe_tolerance_dict)
    return tolerances


def _check_kernel_cuda_vs_cpu(kernel, input, *args, rtol, atol, **kwargs):
    """Checks if the kernel produces closes results for inputs on GPU and CPU."""
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
    """Checks if the kernel is scriptable and if the scripted output is close to the eager one."""
    if input.device.type != "cpu":
        return

    kernel_scripted = _script(kernel)

    input = input.as_subclass(torch.Tensor)
    with ignore_jit_no_profile_information_warning():
        actual = kernel_scripted(input, *args, **kwargs)
    expected = kernel(input, *args, **kwargs)

    assert_close(actual, expected, rtol=rtol, atol=atol)


def _check_kernel_batched_vs_unbatched(kernel, input, *args, rtol, atol, **kwargs):
    """Checks if the kernel produces close results for batched and unbatched inputs."""
    unbatched_input = input.as_subclass(torch.Tensor)

    for batch_dims in [(2,), (2, 1)]:
        repeats = [*batch_dims, *[1] * input.ndim]

        actual = kernel(unbatched_input.repeat(repeats), *args, **kwargs)

        expected = kernel(unbatched_input, *args, **kwargs)
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
    check_batched_vs_unbatched=True,
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

    if check_batched_vs_unbatched:
        _check_kernel_batched_vs_unbatched(kernel, input, *args, **kwargs, **_to_tolerances(check_batched_vs_unbatched))


def _check_dispatcher_scripted_smoke(dispatcher, input, *args, **kwargs):
    """Checks if the dispatcher can be scripted and the scripted version can be called without error."""
    if not isinstance(input, datapoints.Image):
        return

    dispatcher_scripted = _script(dispatcher)
    with ignore_jit_no_profile_information_warning():
        dispatcher_scripted(input.as_subclass(torch.Tensor), *args, **kwargs)


def _check_dispatcher_dispatch(dispatcher, kernel, input, *args, **kwargs):
    """Checks if the dispatcher correctly dispatches the input to the corresponding kernel and that the input type is
    preserved in doing so. For bounding boxes also checks that the format is preserved.
    """
    if isinstance(input, datapoints._datapoint.Datapoint):
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
    else:
        with mock.patch(f"{dispatcher.__module__}.{kernel.__name__}", wraps=kernel) as spy:
            output = dispatcher(input, *args, **kwargs)

            spy.assert_called_once()

    assert isinstance(output, type(input))

    if isinstance(input, datapoints.BoundingBox):
        assert output.format == input.format


def check_dispatcher(
    dispatcher,
    kernel,
    input,
    *args,
    check_scripted_smoke=True,
    check_dispatch=True,
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

    if check_dispatch:
        _check_dispatcher_dispatch(dispatcher, kernel, input, *args, **kwargs)


def _check_dispatcher_kernel_signature_match(dispatcher, *, kernel, input_type):
    """Checks if the signature of the dispatcher matches the kernel signature."""
    dispatcher_signature = inspect.signature(dispatcher)
    dispatcher_params = list(dispatcher_signature.parameters.values())[1:]

    kernel_signature = inspect.signature(kernel)
    kernel_params = list(kernel_signature.parameters.values())[1:]

    if issubclass(input_type, datapoints._datapoint.Datapoint):
        # We filter out metadata that is implicitly passed to the dispatcher through the input datapoint, but has to be
        # explicitly passed to the kernel.
        kernel_params = [param for param in kernel_params if param.name not in input_type.__annotations__.keys()]

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

        if issubclass(input_type, PIL.Image.Image):
            # PIL kernels often have more correct annotations, since they are not limited by JIT. Thus, we don't check
            # them in the first place.
            dispatcher_param._annotation = kernel_param._annotation = inspect.Parameter.empty

        assert dispatcher_param == kernel_param


def _check_dispatcher_datapoint_signature_match(dispatcher):
    """Checks if the signature of the dispatcher matches the corresponding method signature on the Datapoint class."""
    dispatcher_signature = inspect.signature(dispatcher)
    dispatcher_params = list(dispatcher_signature.parameters.values())[1:]

    datapoint_method = getattr(datapoints._datapoint.Datapoint, dispatcher.__name__)
    datapoint_signature = inspect.signature(datapoint_method)
    datapoint_params = list(datapoint_signature.parameters.values())[1:]

    # Some annotations in the `datapoints._datapoint` module
    # are stored as strings. The block below makes them concrete again (non-strings), so they can be compared to the
    # natively concrete dispatcher annotations.
    datapoint_annotations = get_type_hints(datapoint_method)
    for param in datapoint_params:
        param._annotation = datapoint_annotations[param.name]

    assert dispatcher_params == datapoint_params


def check_dispatcher_signatures_match(dispatcher, *, kernel, input_type):
    _check_dispatcher_kernel_signature_match(dispatcher, kernel=kernel, input_type=input_type)
    _check_dispatcher_datapoint_signature_match(dispatcher)


def _check_transform_v1_compatibility(transform, input):
    """If the transform defines the ``_v1_transform_cls`` attribute, checks if the transform has a public, static
    ``get_params`` method, is scriptable, and the scripted version can be called without error."""
    if not hasattr(transform, "_v1_transform_cls"):
        return

    if type(input) is not torch.Tensor:
        return

    if hasattr(transform._v1_transform_cls, "get_params"):
        assert type(transform).get_params is transform._v1_transform_cls.get_params

    scripted_transform = _script(transform)
    with ignore_jit_no_profile_information_warning():
        scripted_transform(input)


def check_transform(transform_cls, input, *args, **kwargs):
    transform = transform_cls(*args, **kwargs)

    output = transform(input)
    assert isinstance(output, type(input))

    if isinstance(input, datapoints.BoundingBox):
        assert output.format == input.format

    _check_transform_v1_compatibility(transform, input)


def make_parametrizable(fn):
    def wrapper(*args, **kwargs):
        return functools.partial(fn, *args, **kwargs)

    return wrapper


# We cannot use `list(transforms.InterpolationMode)` here, since it includes some PIL-only ones as well
INTERPOLATION_MODES = [
    transforms.InterpolationMode.NEAREST,
    transforms.InterpolationMode.NEAREST_EXACT,
    transforms.InterpolationMode.BILINEAR,
    transforms.InterpolationMode.BICUBIC,
]


@contextlib.contextmanager
def assert_warns_antialias_default_value():
    with pytest.warns(UserWarning, match="The default value of the antialias parameter of all the resizing transforms"):
        yield


class TestResize:
    INPUT_SIZE = (17, 11)
    OUTPUT_SIZES = [17, [17], (17,), [12, 13], (12, 13)]

    def _make_max_size_kwarg(self, *, use_max_size, size):
        if use_max_size:
            if not (isinstance(size, int) or len(size) == 1):
                # This would result in an `ValueError`
                return None

            max_size = (size if isinstance(size, int) else size[0]) + 1
        else:
            max_size = None

        return dict(max_size=max_size)

    def _make_input(self, input_type, *, dtype=None, device="cpu", **kwargs):
        if input_type in {torch.Tensor, PIL.Image.Image, datapoints.Image}:
            input = make_image(size=self.INPUT_SIZE, dtype=dtype or torch.uint8, device=device, **kwargs)
            if input_type is torch.Tensor:
                input = input.as_subclass(torch.Tensor)
            elif input_type is PIL.Image.Image:
                input = F.to_image_pil(input)
        elif input_type is datapoints.BoundingBox:
            kwargs.setdefault("format", datapoints.BoundingBoxFormat.XYXY)
            input = make_bounding_box(
                spatial_size=self.INPUT_SIZE,
                dtype=dtype or torch.float32,
                device=device,
                **kwargs,
            )
        elif input_type is datapoints.Mask:
            input = make_segmentation_mask(size=self.INPUT_SIZE, dtype=dtype or torch.uint8, device=device, **kwargs)
        elif input_type is datapoints.Video:
            input = make_video(size=self.INPUT_SIZE, dtype=dtype or torch.uint8, device=device, **kwargs)

        return input

    def _check_size(self, input, output, *, size, max_size):
        if isinstance(size, int) or len(size) == 1:
            if not isinstance(size, int):
                size = size[0]

            old_height, old_width = F.get_spatial_size(input)
            ratio = old_width / old_height
            if ratio > 1:
                new_height = size
                new_width = int(ratio * new_height)
            else:
                new_width = size
                new_height = int(new_width / ratio)

            if max_size is not None and max(new_height, new_width) > max_size:
                # Need to recompute the aspect ratio, since it might have changed due to rounding
                ratio = new_width / new_height
                if ratio > 1:
                    new_width = max_size
                    new_height = int(new_width / ratio)
                else:
                    new_height = max_size
                    new_width = int(new_height * ratio)

        else:
            new_height, new_width = size

        assert F.get_spatial_size(output) == [new_height, new_width]

    @pytest.mark.parametrize("size", OUTPUT_SIZES)
    @pytest.mark.parametrize("interpolation", INTERPOLATION_MODES)
    @pytest.mark.parametrize("use_max_size", [True, False])
    @pytest.mark.parametrize("antialias", [True, False])
    @pytest.mark.parametrize("dtype", [torch.float32, torch.uint8])
    @pytest.mark.parametrize("device", cpu_and_gpu())
    def test_kernel_image_tensor(self, size, interpolation, use_max_size, antialias, dtype, device):
        if not (max_size_kwarg := self._make_max_size_kwarg(use_max_size=use_max_size, size=size)):
            return

        check_kernel(
            F.resize_image_tensor,
            self._make_input(datapoints.Image, dtype=dtype, device=device),
            size=size,
            interpolation=interpolation,
            **max_size_kwarg,
            antialias=antialias,
            # The `InterpolationMode.BICUBIC` implementation on CUDA does not match PILs implementation well. Thus,
            # instead of testing with an enormous tolerance, we disable the check all together.
            check_cuda_vs_cpu=False
            if interpolation is transforms.InterpolationMode.BICUBIC
            else dict(rtol=0, atol=1 / 255 if dtype.is_floating_point else 1),
            check_scripted_vs_eager=not isinstance(size, int),
        )

    @pytest.mark.parametrize("size", OUTPUT_SIZES)
    @pytest.mark.parametrize("format", list(datapoints.BoundingBoxFormat))
    @pytest.mark.parametrize("use_max_size", [True, False])
    @pytest.mark.parametrize("dtype", [torch.float32, torch.int64])
    @pytest.mark.parametrize("device", cpu_and_gpu())
    def test_kernel_bounding_box(self, size, format, use_max_size, dtype, device):
        if not (max_size_kwarg := self._make_max_size_kwarg(use_max_size=use_max_size, size=size)):
            return

        bounding_box = self._make_input(datapoints.BoundingBox, dtype=dtype, device=device, format=format)
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
        check_kernel(F.resize_mask, make_mask(dtype=dtype), size=self.OUTPUT_SIZES[-1])

    def test_kernel_video(self):
        check_kernel(F.resize_video, self._make_input(datapoints.Video), size=self.OUTPUT_SIZES[-1], antialias=True)

    @pytest.mark.parametrize("size", OUTPUT_SIZES)
    @pytest.mark.parametrize(
        "input_type_and_kernel",
        [
            (torch.Tensor, F.resize_image_tensor),
            (PIL.Image.Image, F.resize_image_pil),
            (datapoints.Image, F.resize_image_tensor),
            (datapoints.BoundingBox, F.resize_bounding_box),
            (datapoints.Mask, F.resize_mask),
            (datapoints.Video, F.resize_video),
        ],
    )
    def test_dispatcher(self, size, input_type_and_kernel):
        input_type, kernel = input_type_and_kernel
        check_dispatcher(
            F.resize,
            kernel,
            self._make_input(input_type),
            size=size,
            antialias=True,
            check_scripted_smoke=not isinstance(size, int),
        )

    @pytest.mark.parametrize(
        ("input_type", "kernel"),
        [
            (torch.Tensor, F.resize_image_tensor),
            (PIL.Image.Image, F.resize_image_pil),
            (datapoints.Image, F.resize_image_tensor),
            (datapoints.BoundingBox, F.resize_bounding_box),
            (datapoints.Mask, F.resize_mask),
            (datapoints.Video, F.resize_video),
        ],
    )
    def test_dispatcher_signature(self, kernel, input_type):
        check_dispatcher_signatures_match(F.resize, kernel=kernel, input_type=input_type)

    @pytest.mark.parametrize("size", OUTPUT_SIZES)
    @pytest.mark.parametrize("device", cpu_and_gpu())
    @pytest.mark.parametrize(
        "input_type",
        [torch.Tensor, PIL.Image.Image, datapoints.Image, datapoints.BoundingBox, datapoints.Mask, datapoints.Video],
    )
    def test_transform(self, size, device, input_type):
        input = self._make_input(input_type, device=device)

        check_transform(
            transforms.Resize,
            input,
            size=size,
            antialias=True,
        )

    @pytest.mark.parametrize("size", OUTPUT_SIZES)
    # `InterpolationMode.NEAREST` is modeled after the buggy `INTER_NEAREST` interpolation of CV2.
    # The PIL equivalent of `InterpolationMode.NEAREST` is `InterpolationMode.NEAREST_EXACT`
    @pytest.mark.parametrize("interpolation", set(INTERPOLATION_MODES) - {transforms.InterpolationMode.NEAREST})
    @pytest.mark.parametrize("use_max_size", [True, False])
    @pytest.mark.parametrize("make_fn", [make_parametrizable(F.resize_image_tensor), transforms.Resize])
    def test_image_correctness(self, size, interpolation, use_max_size, make_fn):
        if not (max_size_kwarg := self._make_max_size_kwarg(use_max_size=use_max_size, size=size)):
            return

        fn = make_fn(size=size, interpolation=interpolation, **max_size_kwarg, antialias=True)

        image = self._make_input(torch.Tensor, dtype=torch.uint8, device="cpu")

        actual = fn(image)
        expected = F.to_image_tensor(
            F.resize_image_pil(F.to_image_pil(image), size=size, interpolation=interpolation, **max_size_kwarg)
        )

        self._check_size(image, actual, size=size, **max_size_kwarg)

        torch.testing.assert_close(actual, expected, atol=1, rtol=0)

    @pytest.mark.parametrize("interpolation", set(transforms.InterpolationMode) - set(INTERPOLATION_MODES))
    @pytest.mark.parametrize(
        "input_type",
        [torch.Tensor, PIL.Image.Image, datapoints.Image, datapoints.Video],
    )
    def test_pil_interpolation_compat_smoke(self, interpolation, input_type):
        input = self._make_input(input_type)

        with (
            contextlib.nullcontext()
            if isinstance(input, PIL.Image.Image)
            # This error is triggered in PyTorch core
            else pytest.raises(NotImplementedError, match=f"got {interpolation.value.lower()}")
        ):
            F.resize(
                input,
                size=self.OUTPUT_SIZES[0],
                interpolation=interpolation,
            )

    def test_dispatcher_pil_antialias_warning(self):
        with pytest.warns(UserWarning, match="Anti-alias option is always applied for PIL Image input"):
            F.resize(self._make_input(PIL.Image.Image), size=self.OUTPUT_SIZES[0], antialias=False)

    @pytest.mark.parametrize("size", OUTPUT_SIZES)
    @pytest.mark.parametrize(
        "input_type",
        [torch.Tensor, PIL.Image.Image, datapoints.Image, datapoints.BoundingBox, datapoints.Mask, datapoints.Video],
    )
    def test_max_size_error(self, size, input_type):
        if isinstance(size, int) or len(size) == 1:
            max_size = (size if isinstance(size, int) else size[0]) - 1
            match = "must be strictly greater than the requested size"
        else:
            # value can be anything other than None
            max_size = -1
            match = "size should be an int or a sequence of length 1"

        with pytest.raises(ValueError, match=match):
            F.resize(self._make_input(input_type), size=size, max_size=max_size, antialias=True)

    @pytest.mark.parametrize("interpolation", INTERPOLATION_MODES)
    @pytest.mark.parametrize(
        "input_type",
        [torch.Tensor, datapoints.Image, datapoints.Video],
    )
    def test_antialias_warning(self, interpolation, input_type):
        with (
            assert_warns_antialias_default_value()
            if interpolation in {transforms.InterpolationMode.BILINEAR, transforms.InterpolationMode.BICUBIC}
            else assert_no_warnings()
        ):
            F.resize(self._make_input(input_type), size=self.OUTPUT_SIZES[0], interpolation=interpolation)

    @pytest.mark.parametrize("interpolation", INTERPOLATION_MODES)
    @pytest.mark.parametrize(
        "input_type",
        [torch.Tensor, PIL.Image.Image, datapoints.Image, datapoints.Video],
    )
    def test_interpolation_int(self, interpolation, input_type):
        # `InterpolationMode.NEAREST_EXACT` has no proper corresponding integer equivalent. Internally, we map it to
        # `0` to be the same as `InterpolationMode.NEAREST` for PIL. However, for the tensor backend there is a
        # difference and thus we don't test it here.
        if issubclass(input_type, torch.Tensor) and interpolation is transforms.InterpolationMode.NEAREST_EXACT:
            return

        input = self._make_input(input_type)

        expected = F.resize(input, size=self.OUTPUT_SIZES[0], interpolation=interpolation, antialias=True)
        actual = F.resize(
            input, size=self.OUTPUT_SIZES[0], interpolation=pil_modes_mapping[interpolation], antialias=True
        )

        assert_equal(actual, expected)

    def test_transform_unknown_size_error(self):
        with pytest.raises(ValueError, match="size can either be an integer or a list or tuple of one or two integers"):
            transforms.Resize(size=object())

    @pytest.mark.parametrize(
        "size", [min(INPUT_SIZE), [min(INPUT_SIZE)], (min(INPUT_SIZE),), list(INPUT_SIZE), tuple(INPUT_SIZE)]
    )
    @pytest.mark.parametrize(
        "input_type",
        [torch.Tensor, PIL.Image.Image, datapoints.Image, datapoints.BoundingBox, datapoints.Mask, datapoints.Video],
    )
    def test_noop(self, size, input_type):
        input = self._make_input(input_type)

        output = F.resize(input, size=size, antialias=True)

        # This identity check is not a requirement. It is here to avoid breaking the behavior by accident. If there
        # is a good reason to break this, feel free to downgrade to an equality check.
        if isinstance(input, datapoints._datapoint.Datapoint):
            # We can't test identity directly, since that checks for the identity of the Python object. Since all
            # datapoints unwrap before a kernel and wrap again afterwards, the Python object changes. Thus, we check
            # that the underlying storage is the same
            assert output.data_ptr() == input.data_ptr()
        else:
            assert output is input

    @pytest.mark.parametrize(
        "input_type",
        [torch.Tensor, PIL.Image.Image, datapoints.Image, datapoints.BoundingBox, datapoints.Mask, datapoints.Video],
    )
    def test_no_regression_5405(self, input_type):
        # Checks that `max_size` is not ignored if `size == small_edge_size`
        # See https://github.com/pytorch/vision/issues/5405

        input = self._make_input(input_type)

        size = min(F.get_spatial_size(input))
        max_size = size + 1
        output = F.resize(input, size=size, max_size=max_size, antialias=True)

        assert max(F.get_spatial_size(output)) == max_size
