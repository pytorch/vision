import contextlib
import inspect
import math
import re
from typing import get_type_hints
from unittest import mock

import numpy as np
import PIL.Image
import pytest

import torch
import torchvision.transforms.v2 as transforms
from common_utils import (
    assert_equal,
    assert_no_warnings,
    cache,
    cpu_and_cuda,
    ignore_jit_no_profile_information_warning,
    make_bounding_box,
    make_detection_mask,
    make_image,
    make_image_pil,
    make_image_tensor,
    make_segmentation_mask,
    make_video,
    needs_cuda,
    set_rng_seed,
)
from torch.testing import assert_close
from torch.utils.data import DataLoader, default_collate
from torchvision import datapoints

from torchvision.transforms._functional_tensor import _max_value as get_max_value
from torchvision.transforms.functional import pil_modes_mapping
from torchvision.transforms.v2 import functional as F


@pytest.fixture(autouse=True)
def fix_rng_seed():
    set_rng_seed(0)
    yield


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
        spy = mock.MagicMock(wraps=kernel, name=kernel.__name__)
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


def transform_cls_to_functional(transform_cls, **transform_specific_kwargs):
    def wrapper(input, *args, **kwargs):
        transform = transform_cls(*args, **transform_specific_kwargs, **kwargs)
        return transform(input)

    wrapper.__name__ = transform_cls.__name__

    return wrapper


def param_value_parametrization(**kwargs):
    """Helper function to turn

    @pytest.mark.parametrize(
        ("param", "value"),
        ("a", 1),
        ("a", 2),
        ("a", 3),
        ("b", -1.0)
        ("b", 1.0)
    )

    into

    @param_value_parametrization(a=[1, 2, 3], b=[-1.0, 1.0])
    """
    return pytest.mark.parametrize(
        ("param", "value"),
        [(param, value) for param, values in kwargs.items() for value in values],
    )


def adapt_fill(value, *, dtype):
    """Adapt fill values in the range [0.0, 1.0] to the value range of the dtype"""
    if value is None:
        return value

    max_value = get_max_value(dtype)

    if isinstance(value, (int, float)):
        return type(value)(value * max_value)
    elif isinstance(value, (list, tuple)):
        return type(value)(type(v)(v * max_value) for v in value)
    else:
        raise ValueError(f"fill should be an int or float, or a list or tuple of the former, but got '{value}'.")


EXHAUSTIVE_TYPE_FILLS = [
    None,
    1,
    0.5,
    [1],
    [0.2],
    (0,),
    (0.7,),
    [1, 0, 1],
    [0.1, 0.2, 0.3],
    (0, 1, 0),
    (0.9, 0.234, 0.314),
]
CORRECTNESS_FILLS = [
    v for v in EXHAUSTIVE_TYPE_FILLS if v is None or isinstance(v, float) or (isinstance(v, list) and len(v) > 1)
]


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


def reference_affine_bounding_box_helper(bounding_box, *, format, spatial_size, affine_matrix):
    def transform(bbox):
        # Go to float before converting to prevent precision loss in case of CXCYWH -> XYXY and W or H is 1
        in_dtype = bbox.dtype
        if not torch.is_floating_point(bbox):
            bbox = bbox.float()
        bbox_xyxy = F.convert_format_bounding_box(
            bbox.as_subclass(torch.Tensor),
            old_format=format,
            new_format=datapoints.BoundingBoxFormat.XYXY,
            inplace=True,
        )
        points = np.array(
            [
                [bbox_xyxy[0].item(), bbox_xyxy[1].item(), 1.0],
                [bbox_xyxy[2].item(), bbox_xyxy[1].item(), 1.0],
                [bbox_xyxy[0].item(), bbox_xyxy[3].item(), 1.0],
                [bbox_xyxy[2].item(), bbox_xyxy[3].item(), 1.0],
            ]
        )
        transformed_points = np.matmul(points, affine_matrix.T)
        out_bbox = torch.tensor(
            [
                np.min(transformed_points[:, 0]).item(),
                np.min(transformed_points[:, 1]).item(),
                np.max(transformed_points[:, 0]).item(),
                np.max(transformed_points[:, 1]).item(),
            ],
            dtype=bbox_xyxy.dtype,
        )
        out_bbox = F.convert_format_bounding_box(
            out_bbox, old_format=datapoints.BoundingBoxFormat.XYXY, new_format=format, inplace=True
        )
        # It is important to clamp before casting, especially for CXCYWH format, dtype=int64
        out_bbox = F.clamp_bounding_box(out_bbox, format=format, spatial_size=spatial_size)
        out_bbox = out_bbox.to(dtype=in_dtype)
        return out_bbox

    return torch.stack([transform(b) for b in bounding_box.reshape(-1, 4).unbind()]).reshape(bounding_box.shape)


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

    def _compute_output_size(self, *, input_size, size, max_size):
        if not (isinstance(size, int) or len(size) == 1):
            return tuple(size)

        if not isinstance(size, int):
            size = size[0]

        old_height, old_width = input_size
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

        return new_height, new_width

    @pytest.mark.parametrize("size", OUTPUT_SIZES)
    @pytest.mark.parametrize("interpolation", INTERPOLATION_MODES)
    @pytest.mark.parametrize("use_max_size", [True, False])
    @pytest.mark.parametrize("antialias", [True, False])
    @pytest.mark.parametrize("dtype", [torch.float32, torch.uint8])
    @pytest.mark.parametrize("device", cpu_and_cuda())
    def test_kernel_image_tensor(self, size, interpolation, use_max_size, antialias, dtype, device):
        if not (max_size_kwarg := self._make_max_size_kwarg(use_max_size=use_max_size, size=size)):
            return

        # In contrast to CPU, there is no native `InterpolationMode.BICUBIC` implementation for uint8 images on CUDA.
        # Internally, it uses the float path. Thus, we need to test with an enormous tolerance here to account for that.
        atol = 30 if transforms.InterpolationMode.BICUBIC and dtype is torch.uint8 else 1
        check_cuda_vs_cpu_tolerances = dict(rtol=0, atol=atol / 255 if dtype.is_floating_point else atol)

        check_kernel(
            F.resize_image_tensor,
            make_image(self.INPUT_SIZE, dtype=dtype, device=device),
            size=size,
            interpolation=interpolation,
            **max_size_kwarg,
            antialias=antialias,
            check_cuda_vs_cpu=check_cuda_vs_cpu_tolerances,
            check_scripted_vs_eager=not isinstance(size, int),
        )

    @pytest.mark.parametrize("format", list(datapoints.BoundingBoxFormat))
    @pytest.mark.parametrize("size", OUTPUT_SIZES)
    @pytest.mark.parametrize("use_max_size", [True, False])
    @pytest.mark.parametrize("dtype", [torch.float32, torch.int64])
    @pytest.mark.parametrize("device", cpu_and_cuda())
    def test_kernel_bounding_box(self, format, size, use_max_size, dtype, device):
        if not (max_size_kwarg := self._make_max_size_kwarg(use_max_size=use_max_size, size=size)):
            return

        bounding_box = make_bounding_box(
            format=format,
            spatial_size=self.INPUT_SIZE,
            dtype=dtype,
            device=device,
        )
        check_kernel(
            F.resize_bounding_box,
            bounding_box,
            spatial_size=bounding_box.spatial_size,
            size=size,
            **max_size_kwarg,
            check_scripted_vs_eager=not isinstance(size, int),
        )

    @pytest.mark.parametrize("make_mask", [make_segmentation_mask, make_detection_mask])
    def test_kernel_mask(self, make_mask):
        check_kernel(F.resize_mask, make_mask(self.INPUT_SIZE), size=self.OUTPUT_SIZES[-1])

    def test_kernel_video(self):
        check_kernel(F.resize_video, make_video(self.INPUT_SIZE), size=self.OUTPUT_SIZES[-1], antialias=True)

    @pytest.mark.parametrize("size", OUTPUT_SIZES)
    @pytest.mark.parametrize(
        ("kernel", "make_input"),
        [
            (F.resize_image_tensor, make_image_tensor),
            (F.resize_image_pil, make_image_pil),
            (F.resize_image_tensor, make_image),
            (F.resize_bounding_box, make_bounding_box),
            (F.resize_mask, make_segmentation_mask),
            (F.resize_video, make_video),
        ],
    )
    def test_dispatcher(self, size, kernel, make_input):
        check_dispatcher(
            F.resize,
            kernel,
            make_input(self.INPUT_SIZE),
            size=size,
            antialias=True,
            check_scripted_smoke=not isinstance(size, int),
        )

    @pytest.mark.parametrize(
        ("kernel", "input_type"),
        [
            (F.resize_image_tensor, torch.Tensor),
            (F.resize_image_pil, PIL.Image.Image),
            (F.resize_image_tensor, datapoints.Image),
            (F.resize_bounding_box, datapoints.BoundingBox),
            (F.resize_mask, datapoints.Mask),
            (F.resize_video, datapoints.Video),
        ],
    )
    def test_dispatcher_signature(self, kernel, input_type):
        check_dispatcher_signatures_match(F.resize, kernel=kernel, input_type=input_type)

    @pytest.mark.parametrize("size", OUTPUT_SIZES)
    @pytest.mark.parametrize("device", cpu_and_cuda())
    @pytest.mark.parametrize(
        "make_input",
        [
            make_image_tensor,
            make_image_pil,
            make_image,
            make_bounding_box,
            make_segmentation_mask,
            make_detection_mask,
            make_video,
        ],
    )
    def test_transform(self, size, device, make_input):
        check_transform(transforms.Resize, make_input(self.INPUT_SIZE, device=device), size=size, antialias=True)

    def _check_output_size(self, input, output, *, size, max_size):
        assert tuple(F.get_spatial_size(output)) == self._compute_output_size(
            input_size=F.get_spatial_size(input), size=size, max_size=max_size
        )

    @pytest.mark.parametrize("size", OUTPUT_SIZES)
    # `InterpolationMode.NEAREST` is modeled after the buggy `INTER_NEAREST` interpolation of CV2.
    # The PIL equivalent of `InterpolationMode.NEAREST` is `InterpolationMode.NEAREST_EXACT`
    @pytest.mark.parametrize("interpolation", set(INTERPOLATION_MODES) - {transforms.InterpolationMode.NEAREST})
    @pytest.mark.parametrize("use_max_size", [True, False])
    @pytest.mark.parametrize("fn", [F.resize, transform_cls_to_functional(transforms.Resize)])
    def test_image_correctness(self, size, interpolation, use_max_size, fn):
        if not (max_size_kwarg := self._make_max_size_kwarg(use_max_size=use_max_size, size=size)):
            return

        image = make_image(self.INPUT_SIZE, dtype=torch.uint8)

        actual = fn(image, size=size, interpolation=interpolation, **max_size_kwarg, antialias=True)
        expected = F.to_image_tensor(
            F.resize(F.to_image_pil(image), size=size, interpolation=interpolation, **max_size_kwarg)
        )

        self._check_output_size(image, actual, size=size, **max_size_kwarg)
        torch.testing.assert_close(actual, expected, atol=1, rtol=0)

    def _reference_resize_bounding_box(self, bounding_box, *, size, max_size=None):
        old_height, old_width = bounding_box.spatial_size
        new_height, new_width = self._compute_output_size(
            input_size=bounding_box.spatial_size, size=size, max_size=max_size
        )

        if (old_height, old_width) == (new_height, new_width):
            return bounding_box

        affine_matrix = np.array(
            [
                [new_width / old_width, 0, 0],
                [0, new_height / old_height, 0],
            ],
            dtype="float64" if bounding_box.dtype == torch.float64 else "float32",
        )

        expected_bboxes = reference_affine_bounding_box_helper(
            bounding_box,
            format=bounding_box.format,
            spatial_size=(new_height, new_width),
            affine_matrix=affine_matrix,
        )
        return datapoints.BoundingBox.wrap_like(bounding_box, expected_bboxes, spatial_size=(new_height, new_width))

    @pytest.mark.parametrize("format", list(datapoints.BoundingBoxFormat))
    @pytest.mark.parametrize("size", OUTPUT_SIZES)
    @pytest.mark.parametrize("use_max_size", [True, False])
    @pytest.mark.parametrize("fn", [F.resize, transform_cls_to_functional(transforms.Resize)])
    def test_bounding_box_correctness(self, format, size, use_max_size, fn):
        if not (max_size_kwarg := self._make_max_size_kwarg(use_max_size=use_max_size, size=size)):
            return

        bounding_box = make_bounding_box(format=format, spatial_size=self.INPUT_SIZE)

        actual = fn(bounding_box, size=size, **max_size_kwarg)
        expected = self._reference_resize_bounding_box(bounding_box, size=size, **max_size_kwarg)

        self._check_output_size(bounding_box, actual, size=size, **max_size_kwarg)
        torch.testing.assert_close(actual, expected)

    @pytest.mark.parametrize("interpolation", set(transforms.InterpolationMode) - set(INTERPOLATION_MODES))
    @pytest.mark.parametrize(
        "make_input",
        [make_image_tensor, make_image_pil, make_image, make_video],
    )
    def test_pil_interpolation_compat_smoke(self, interpolation, make_input):
        input = make_input(self.INPUT_SIZE)

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
            F.resize(make_image_pil(self.INPUT_SIZE), size=self.OUTPUT_SIZES[0], antialias=False)

    @pytest.mark.parametrize("size", OUTPUT_SIZES)
    @pytest.mark.parametrize(
        "make_input",
        [
            make_image_tensor,
            make_image_pil,
            make_image,
            make_bounding_box,
            make_segmentation_mask,
            make_detection_mask,
            make_video,
        ],
    )
    def test_max_size_error(self, size, make_input):
        if isinstance(size, int) or len(size) == 1:
            max_size = (size if isinstance(size, int) else size[0]) - 1
            match = "must be strictly greater than the requested size"
        else:
            # value can be anything other than None
            max_size = -1
            match = "size should be an int or a sequence of length 1"

        with pytest.raises(ValueError, match=match):
            F.resize(make_input(self.INPUT_SIZE), size=size, max_size=max_size, antialias=True)

    @pytest.mark.parametrize("interpolation", INTERPOLATION_MODES)
    @pytest.mark.parametrize(
        "make_input",
        [make_image_tensor, make_image, make_video],
    )
    def test_antialias_warning(self, interpolation, make_input):
        with (
            assert_warns_antialias_default_value()
            if interpolation in {transforms.InterpolationMode.BILINEAR, transforms.InterpolationMode.BICUBIC}
            else assert_no_warnings()
        ):
            F.resize(
                make_input(self.INPUT_SIZE),
                size=self.OUTPUT_SIZES[0],
                interpolation=interpolation,
            )

    @pytest.mark.parametrize("interpolation", INTERPOLATION_MODES)
    @pytest.mark.parametrize(
        "make_input",
        [make_image_tensor, make_image_pil, make_image, make_video],
    )
    def test_interpolation_int(self, interpolation, make_input):
        input = make_input(self.INPUT_SIZE)

        # `InterpolationMode.NEAREST_EXACT` has no proper corresponding integer equivalent. Internally, we map it to
        # `0` to be the same as `InterpolationMode.NEAREST` for PIL. However, for the tensor backend there is a
        # difference and thus we don't test it here.
        if isinstance(input, torch.Tensor) and interpolation is transforms.InterpolationMode.NEAREST_EXACT:
            return

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
        "make_input",
        [
            make_image_tensor,
            make_image_pil,
            make_image,
            make_bounding_box,
            make_segmentation_mask,
            make_detection_mask,
            make_video,
        ],
    )
    def test_noop(self, size, make_input):
        input = make_input(self.INPUT_SIZE)

        output = F.resize(input, size=F.get_spatial_size(input), antialias=True)

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
        "make_input",
        [
            make_image_tensor,
            make_image_pil,
            make_image,
            make_bounding_box,
            make_segmentation_mask,
            make_detection_mask,
            make_video,
        ],
    )
    def test_no_regression_5405(self, make_input):
        # Checks that `max_size` is not ignored if `size == small_edge_size`
        # See https://github.com/pytorch/vision/issues/5405

        input = make_input(self.INPUT_SIZE)

        size = min(F.get_spatial_size(input))
        max_size = size + 1
        output = F.resize(input, size=size, max_size=max_size, antialias=True)

        assert max(F.get_spatial_size(output)) == max_size


class TestHorizontalFlip:
    @pytest.mark.parametrize("dtype", [torch.float32, torch.uint8])
    @pytest.mark.parametrize("device", cpu_and_cuda())
    def test_kernel_image_tensor(self, dtype, device):
        check_kernel(F.horizontal_flip_image_tensor, make_image(dtype=dtype, device=device))

    @pytest.mark.parametrize("format", list(datapoints.BoundingBoxFormat))
    @pytest.mark.parametrize("dtype", [torch.float32, torch.int64])
    @pytest.mark.parametrize("device", cpu_and_cuda())
    def test_kernel_bounding_box(self, format, dtype, device):
        bounding_box = make_bounding_box(format=format, dtype=dtype, device=device)
        check_kernel(
            F.horizontal_flip_bounding_box,
            bounding_box,
            format=format,
            spatial_size=bounding_box.spatial_size,
        )

    @pytest.mark.parametrize("make_mask", [make_segmentation_mask, make_detection_mask])
    def test_kernel_mask(self, make_mask):
        check_kernel(F.horizontal_flip_mask, make_mask())

    def test_kernel_video(self):
        check_kernel(F.horizontal_flip_video, make_video())

    @pytest.mark.parametrize(
        ("kernel", "make_input"),
        [
            (F.horizontal_flip_image_tensor, make_image_tensor),
            (F.horizontal_flip_image_pil, make_image_pil),
            (F.horizontal_flip_image_tensor, make_image),
            (F.horizontal_flip_bounding_box, make_bounding_box),
            (F.horizontal_flip_mask, make_segmentation_mask),
            (F.horizontal_flip_video, make_video),
        ],
    )
    def test_dispatcher(self, kernel, make_input):
        check_dispatcher(F.horizontal_flip, kernel, make_input())

    @pytest.mark.parametrize(
        ("kernel", "input_type"),
        [
            (F.horizontal_flip_image_tensor, torch.Tensor),
            (F.horizontal_flip_image_pil, PIL.Image.Image),
            (F.horizontal_flip_image_tensor, datapoints.Image),
            (F.horizontal_flip_bounding_box, datapoints.BoundingBox),
            (F.horizontal_flip_mask, datapoints.Mask),
            (F.horizontal_flip_video, datapoints.Video),
        ],
    )
    def test_dispatcher_signature(self, kernel, input_type):
        check_dispatcher_signatures_match(F.horizontal_flip, kernel=kernel, input_type=input_type)

    @pytest.mark.parametrize(
        "make_input",
        [make_image_tensor, make_image_pil, make_image, make_bounding_box, make_segmentation_mask, make_video],
    )
    @pytest.mark.parametrize("device", cpu_and_cuda())
    def test_transform(self, make_input, device):
        check_transform(transforms.RandomHorizontalFlip, make_input(device=device), p=1)

    @pytest.mark.parametrize(
        "fn", [F.horizontal_flip, transform_cls_to_functional(transforms.RandomHorizontalFlip, p=1)]
    )
    def test_image_correctness(self, fn):
        image = make_image(dtype=torch.uint8, device="cpu")

        actual = fn(image)
        expected = F.to_image_tensor(F.horizontal_flip(F.to_image_pil(image)))

        torch.testing.assert_close(actual, expected)

    def _reference_horizontal_flip_bounding_box(self, bounding_box):
        affine_matrix = np.array(
            [
                [-1, 0, bounding_box.spatial_size[1]],
                [0, 1, 0],
            ],
            dtype="float64" if bounding_box.dtype == torch.float64 else "float32",
        )

        expected_bboxes = reference_affine_bounding_box_helper(
            bounding_box,
            format=bounding_box.format,
            spatial_size=bounding_box.spatial_size,
            affine_matrix=affine_matrix,
        )

        return datapoints.BoundingBox.wrap_like(bounding_box, expected_bboxes)

    @pytest.mark.parametrize("format", list(datapoints.BoundingBoxFormat))
    @pytest.mark.parametrize(
        "fn", [F.horizontal_flip, transform_cls_to_functional(transforms.RandomHorizontalFlip, p=1)]
    )
    def test_bounding_box_correctness(self, format, fn):
        bounding_box = make_bounding_box(format=format)

        actual = fn(bounding_box)
        expected = self._reference_horizontal_flip_bounding_box(bounding_box)

        torch.testing.assert_close(actual, expected)

    @pytest.mark.parametrize(
        "make_input",
        [make_image_tensor, make_image_pil, make_image, make_bounding_box, make_segmentation_mask, make_video],
    )
    @pytest.mark.parametrize("device", cpu_and_cuda())
    def test_transform_noop(self, make_input, device):
        input = make_input(device=device)

        transform = transforms.RandomHorizontalFlip(p=0)

        output = transform(input)

        assert_equal(output, input)


class TestAffine:
    _EXHAUSTIVE_TYPE_AFFINE_KWARGS = dict(
        # float, int
        angle=[-10.9, 18],
        # two-list of float, two-list of int, two-tuple of float, two-tuple of int
        translate=[[6.3, -0.6], [1, -3], (16.6, -6.6), (-2, 4)],
        # float
        scale=[0.5],
        # float, int,
        # one-list of float, one-list of int, one-tuple of float, one-tuple of int
        # two-list of float, two-list of int, two-tuple of float, two-tuple of int
        shear=[35.6, 38, [-37.7], [-23], (5.3,), (-52,), [5.4, 21.8], [-47, 51], (-11.2, 36.7), (8, -53)],
        # None
        # two-list of float, two-list of int, two-tuple of float, two-tuple of int
        center=[None, [1.2, 4.9], [-3, 1], (2.5, -4.7), (3, 2)],
    )
    # The special case for shear makes sure we pick a value that is supported while JIT scripting
    _MINIMAL_AFFINE_KWARGS = {
        k: vs[0] if k != "shear" else next(v for v in vs if isinstance(v, list))
        for k, vs in _EXHAUSTIVE_TYPE_AFFINE_KWARGS.items()
    }
    _CORRECTNESS_AFFINE_KWARGS = {
        k: [v for v in vs if v is None or isinstance(v, float) or (isinstance(v, list) and len(v) > 1)]
        for k, vs in _EXHAUSTIVE_TYPE_AFFINE_KWARGS.items()
    }

    _EXHAUSTIVE_TYPE_TRANSFORM_AFFINE_RANGES = dict(
        degrees=[30, (-15, 20)],
        translate=[None, (0.5, 0.5)],
        scale=[None, (0.75, 1.25)],
        shear=[None, (12, 30, -17, 5), 10, (-5, 12)],
    )
    _CORRECTNESS_TRANSFORM_AFFINE_RANGES = {
        k: next(v for v in vs if v is not None) for k, vs in _EXHAUSTIVE_TYPE_TRANSFORM_AFFINE_RANGES.items()
    }

    def _check_kernel(self, kernel, input, *args, **kwargs):
        kwargs_ = self._MINIMAL_AFFINE_KWARGS.copy()
        kwargs_.update(kwargs)
        check_kernel(kernel, input, *args, **kwargs_)

    @param_value_parametrization(
        angle=_EXHAUSTIVE_TYPE_AFFINE_KWARGS["angle"],
        translate=_EXHAUSTIVE_TYPE_AFFINE_KWARGS["translate"],
        shear=_EXHAUSTIVE_TYPE_AFFINE_KWARGS["shear"],
        center=_EXHAUSTIVE_TYPE_AFFINE_KWARGS["center"],
        interpolation=[transforms.InterpolationMode.NEAREST, transforms.InterpolationMode.BILINEAR],
        fill=EXHAUSTIVE_TYPE_FILLS,
    )
    @pytest.mark.parametrize("dtype", [torch.float32, torch.uint8])
    @pytest.mark.parametrize("device", cpu_and_cuda())
    def test_kernel_image_tensor(self, param, value, dtype, device):
        if param == "fill":
            value = adapt_fill(value, dtype=dtype)
        self._check_kernel(
            F.affine_image_tensor,
            make_image(dtype=dtype, device=device),
            **{param: value},
            check_scripted_vs_eager=not (param in {"shear", "fill"} and isinstance(value, (int, float))),
            check_cuda_vs_cpu=dict(atol=1, rtol=0)
            if dtype is torch.uint8 and param == "interpolation" and value is transforms.InterpolationMode.BILINEAR
            else True,
        )

    @param_value_parametrization(
        angle=_EXHAUSTIVE_TYPE_AFFINE_KWARGS["angle"],
        translate=_EXHAUSTIVE_TYPE_AFFINE_KWARGS["translate"],
        shear=_EXHAUSTIVE_TYPE_AFFINE_KWARGS["shear"],
        center=_EXHAUSTIVE_TYPE_AFFINE_KWARGS["center"],
    )
    @pytest.mark.parametrize("format", list(datapoints.BoundingBoxFormat))
    @pytest.mark.parametrize("dtype", [torch.float32, torch.int64])
    @pytest.mark.parametrize("device", cpu_and_cuda())
    def test_kernel_bounding_box(self, param, value, format, dtype, device):
        bounding_box = make_bounding_box(format=format, dtype=dtype, device=device)
        self._check_kernel(
            F.affine_bounding_box,
            bounding_box,
            format=format,
            spatial_size=bounding_box.spatial_size,
            **{param: value},
            check_scripted_vs_eager=not (param == "shear" and isinstance(value, (int, float))),
        )

    @pytest.mark.parametrize("make_mask", [make_segmentation_mask, make_detection_mask])
    def test_kernel_mask(self, make_mask):
        self._check_kernel(F.affine_mask, make_mask())

    def test_kernel_video(self):
        self._check_kernel(F.affine_video, make_video())

    @pytest.mark.parametrize(
        ("kernel", "make_input"),
        [
            (F.affine_image_tensor, make_image_tensor),
            (F.affine_image_pil, make_image_pil),
            (F.affine_image_tensor, make_image),
            (F.affine_bounding_box, make_bounding_box),
            (F.affine_mask, make_segmentation_mask),
            (F.affine_video, make_video),
        ],
    )
    def test_dispatcher(self, kernel, make_input):
        check_dispatcher(F.affine, kernel, make_input(), **self._MINIMAL_AFFINE_KWARGS)

    @pytest.mark.parametrize(
        ("kernel", "input_type"),
        [
            (F.affine_image_tensor, torch.Tensor),
            (F.affine_image_pil, PIL.Image.Image),
            (F.affine_image_tensor, datapoints.Image),
            (F.affine_bounding_box, datapoints.BoundingBox),
            (F.affine_mask, datapoints.Mask),
            (F.affine_video, datapoints.Video),
        ],
    )
    def test_dispatcher_signature(self, kernel, input_type):
        check_dispatcher_signatures_match(F.affine, kernel=kernel, input_type=input_type)

    @pytest.mark.parametrize(
        "make_input",
        [make_image_tensor, make_image_pil, make_image, make_bounding_box, make_segmentation_mask, make_video],
    )
    @pytest.mark.parametrize("device", cpu_and_cuda())
    def test_transform(self, make_input, device):
        input = make_input(device=device)

        check_transform(transforms.RandomAffine, input, **self._CORRECTNESS_TRANSFORM_AFFINE_RANGES)

    @pytest.mark.parametrize("angle", _CORRECTNESS_AFFINE_KWARGS["angle"])
    @pytest.mark.parametrize("translate", _CORRECTNESS_AFFINE_KWARGS["translate"])
    @pytest.mark.parametrize("scale", _CORRECTNESS_AFFINE_KWARGS["scale"])
    @pytest.mark.parametrize("shear", _CORRECTNESS_AFFINE_KWARGS["shear"])
    @pytest.mark.parametrize("center", _CORRECTNESS_AFFINE_KWARGS["center"])
    @pytest.mark.parametrize(
        "interpolation", [transforms.InterpolationMode.NEAREST, transforms.InterpolationMode.BILINEAR]
    )
    @pytest.mark.parametrize("fill", CORRECTNESS_FILLS)
    def test_functional_image_correctness(self, angle, translate, scale, shear, center, interpolation, fill):
        image = make_image(dtype=torch.uint8, device="cpu")

        fill = adapt_fill(fill, dtype=torch.uint8)

        actual = F.affine(
            image,
            angle=angle,
            translate=translate,
            scale=scale,
            shear=shear,
            center=center,
            interpolation=interpolation,
            fill=fill,
        )
        expected = F.to_image_tensor(
            F.affine(
                F.to_image_pil(image),
                angle=angle,
                translate=translate,
                scale=scale,
                shear=shear,
                center=center,
                interpolation=interpolation,
                fill=fill,
            )
        )

        mae = (actual.float() - expected.float()).abs().mean()
        assert mae < 2 if interpolation is transforms.InterpolationMode.NEAREST else 8

    @pytest.mark.parametrize("center", _CORRECTNESS_AFFINE_KWARGS["center"])
    @pytest.mark.parametrize(
        "interpolation", [transforms.InterpolationMode.NEAREST, transforms.InterpolationMode.BILINEAR]
    )
    @pytest.mark.parametrize("fill", CORRECTNESS_FILLS)
    @pytest.mark.parametrize("seed", list(range(5)))
    def test_transform_image_correctness(self, center, interpolation, fill, seed):
        image = make_image(dtype=torch.uint8, device="cpu")

        fill = adapt_fill(fill, dtype=torch.uint8)

        transform = transforms.RandomAffine(
            **self._CORRECTNESS_TRANSFORM_AFFINE_RANGES, center=center, interpolation=interpolation, fill=fill
        )

        torch.manual_seed(seed)
        actual = transform(image)

        torch.manual_seed(seed)
        expected = F.to_image_tensor(transform(F.to_image_pil(image)))

        mae = (actual.float() - expected.float()).abs().mean()
        assert mae < 2 if interpolation is transforms.InterpolationMode.NEAREST else 8

    def _compute_affine_matrix(self, *, angle, translate, scale, shear, center):
        rot = math.radians(angle)
        cx, cy = center
        tx, ty = translate
        sx, sy = [math.radians(s) for s in ([shear, 0.0] if isinstance(shear, (int, float)) else shear)]

        c_matrix = np.array([[1, 0, cx], [0, 1, cy], [0, 0, 1]])
        t_matrix = np.array([[1, 0, tx], [0, 1, ty], [0, 0, 1]])
        c_matrix_inv = np.linalg.inv(c_matrix)
        rs_matrix = np.array(
            [
                [scale * math.cos(rot), -scale * math.sin(rot), 0],
                [scale * math.sin(rot), scale * math.cos(rot), 0],
                [0, 0, 1],
            ]
        )
        shear_x_matrix = np.array([[1, -math.tan(sx), 0], [0, 1, 0], [0, 0, 1]])
        shear_y_matrix = np.array([[1, 0, 0], [-math.tan(sy), 1, 0], [0, 0, 1]])
        rss_matrix = np.matmul(rs_matrix, np.matmul(shear_y_matrix, shear_x_matrix))
        true_matrix = np.matmul(t_matrix, np.matmul(c_matrix, np.matmul(rss_matrix, c_matrix_inv)))
        return true_matrix

    def _reference_affine_bounding_box(self, bounding_box, *, angle, translate, scale, shear, center):
        if center is None:
            center = [s * 0.5 for s in bounding_box.spatial_size[::-1]]

        affine_matrix = self._compute_affine_matrix(
            angle=angle, translate=translate, scale=scale, shear=shear, center=center
        )
        affine_matrix = affine_matrix[:2, :]

        expected_bboxes = reference_affine_bounding_box_helper(
            bounding_box,
            format=bounding_box.format,
            spatial_size=bounding_box.spatial_size,
            affine_matrix=affine_matrix,
        )

        return expected_bboxes

    @pytest.mark.parametrize("format", list(datapoints.BoundingBoxFormat))
    @pytest.mark.parametrize("angle", _CORRECTNESS_AFFINE_KWARGS["angle"])
    @pytest.mark.parametrize("translate", _CORRECTNESS_AFFINE_KWARGS["translate"])
    @pytest.mark.parametrize("scale", _CORRECTNESS_AFFINE_KWARGS["scale"])
    @pytest.mark.parametrize("shear", _CORRECTNESS_AFFINE_KWARGS["shear"])
    @pytest.mark.parametrize("center", _CORRECTNESS_AFFINE_KWARGS["center"])
    def test_functional_bounding_box_correctness(self, format, angle, translate, scale, shear, center):
        bounding_box = make_bounding_box(format=format)

        actual = F.affine(
            bounding_box,
            angle=angle,
            translate=translate,
            scale=scale,
            shear=shear,
            center=center,
        )
        expected = self._reference_affine_bounding_box(
            bounding_box,
            angle=angle,
            translate=translate,
            scale=scale,
            shear=shear,
            center=center,
        )

        torch.testing.assert_close(actual, expected)

    @pytest.mark.parametrize("format", list(datapoints.BoundingBoxFormat))
    @pytest.mark.parametrize("center", _CORRECTNESS_AFFINE_KWARGS["center"])
    @pytest.mark.parametrize("seed", list(range(5)))
    def test_transform_bounding_box_correctness(self, format, center, seed):
        bounding_box = make_bounding_box(format=format)

        transform = transforms.RandomAffine(**self._CORRECTNESS_TRANSFORM_AFFINE_RANGES, center=center)

        torch.manual_seed(seed)
        params = transform._get_params([bounding_box])

        torch.manual_seed(seed)
        actual = transform(bounding_box)

        expected = self._reference_affine_bounding_box(bounding_box, **params, center=center)

        torch.testing.assert_close(actual, expected)

    @pytest.mark.parametrize("degrees", _EXHAUSTIVE_TYPE_TRANSFORM_AFFINE_RANGES["degrees"])
    @pytest.mark.parametrize("translate", _EXHAUSTIVE_TYPE_TRANSFORM_AFFINE_RANGES["translate"])
    @pytest.mark.parametrize("scale", _EXHAUSTIVE_TYPE_TRANSFORM_AFFINE_RANGES["scale"])
    @pytest.mark.parametrize("shear", _EXHAUSTIVE_TYPE_TRANSFORM_AFFINE_RANGES["shear"])
    @pytest.mark.parametrize("seed", list(range(10)))
    def test_transform_get_params_bounds(self, degrees, translate, scale, shear, seed):
        image = make_image()
        height, width = F.get_spatial_size(image)

        transform = transforms.RandomAffine(degrees=degrees, translate=translate, scale=scale, shear=shear)

        torch.manual_seed(seed)
        params = transform._get_params([image])

        if isinstance(degrees, (int, float)):
            assert -degrees <= params["angle"] <= degrees
        else:
            assert degrees[0] <= params["angle"] <= degrees[1]

        if translate is not None:
            width_max = int(round(translate[0] * width))
            height_max = int(round(translate[1] * height))
            assert -width_max <= params["translate"][0] <= width_max
            assert -height_max <= params["translate"][1] <= height_max
        else:
            assert params["translate"] == (0, 0)

        if scale is not None:
            assert scale[0] <= params["scale"] <= scale[1]
        else:
            assert params["scale"] == 1.0

        if shear is not None:
            if isinstance(shear, (int, float)):
                assert -shear <= params["shear"][0] <= shear
                assert params["shear"][1] == 0.0
            elif len(shear) == 2:
                assert shear[0] <= params["shear"][0] <= shear[1]
                assert params["shear"][1] == 0.0
            elif len(shear) == 4:
                assert shear[0] <= params["shear"][0] <= shear[1]
                assert shear[2] <= params["shear"][1] <= shear[3]
        else:
            assert params["shear"] == (0, 0)

    @pytest.mark.parametrize("param", ["degrees", "translate", "scale", "shear", "center"])
    @pytest.mark.parametrize("value", [0, [0], [0, 0, 0]])
    def test_transform_sequence_len_errors(self, param, value):
        if param in {"degrees", "shear"} and not isinstance(value, list):
            return

        kwargs = {param: value}
        if param != "degrees":
            kwargs["degrees"] = 0

        with pytest.raises(
            ValueError if isinstance(value, list) else TypeError, match=f"{param} should be a sequence of length 2"
        ):
            transforms.RandomAffine(**kwargs)

    def test_transform_negative_degrees_error(self):
        with pytest.raises(ValueError, match="If degrees is a single number, it must be positive"):
            transforms.RandomAffine(degrees=-1)

    @pytest.mark.parametrize("translate", [[-1, 0], [2, 0], [-1, 2]])
    def test_transform_translate_range_error(self, translate):
        with pytest.raises(ValueError, match="translation values should be between 0 and 1"):
            transforms.RandomAffine(degrees=0, translate=translate)

    @pytest.mark.parametrize("scale", [[-1, 0], [0, -1], [-1, -1]])
    def test_transform_scale_range_error(self, scale):
        with pytest.raises(ValueError, match="scale values should be positive"):
            transforms.RandomAffine(degrees=0, scale=scale)

    def test_transform_negative_shear_error(self):
        with pytest.raises(ValueError, match="If shear is a single number, it must be positive"):
            transforms.RandomAffine(degrees=0, shear=-1)

    def test_transform_unknown_fill_error(self):
        with pytest.raises(TypeError, match="Got inappropriate fill arg"):
            transforms.RandomAffine(degrees=0, fill="fill")


class TestVerticalFlip:
    @pytest.mark.parametrize("dtype", [torch.float32, torch.uint8])
    @pytest.mark.parametrize("device", cpu_and_cuda())
    def test_kernel_image_tensor(self, dtype, device):
        check_kernel(F.vertical_flip_image_tensor, make_image(dtype=dtype, device=device))

    @pytest.mark.parametrize("format", list(datapoints.BoundingBoxFormat))
    @pytest.mark.parametrize("dtype", [torch.float32, torch.int64])
    @pytest.mark.parametrize("device", cpu_and_cuda())
    def test_kernel_bounding_box(self, format, dtype, device):
        bounding_box = make_bounding_box(format=format, dtype=dtype, device=device)
        check_kernel(
            F.vertical_flip_bounding_box,
            bounding_box,
            format=format,
            spatial_size=bounding_box.spatial_size,
        )

    @pytest.mark.parametrize("make_mask", [make_segmentation_mask, make_detection_mask])
    def test_kernel_mask(self, make_mask):
        check_kernel(F.vertical_flip_mask, make_mask())

    def test_kernel_video(self):
        check_kernel(F.vertical_flip_video, make_video())

    @pytest.mark.parametrize(
        ("kernel", "make_input"),
        [
            (F.vertical_flip_image_tensor, make_image_tensor),
            (F.vertical_flip_image_pil, make_image_pil),
            (F.vertical_flip_image_tensor, make_image),
            (F.vertical_flip_bounding_box, make_bounding_box),
            (F.vertical_flip_mask, make_segmentation_mask),
            (F.vertical_flip_video, make_video),
        ],
    )
    def test_dispatcher(self, kernel, make_input):
        check_dispatcher(F.vertical_flip, kernel, make_input())

    @pytest.mark.parametrize(
        ("kernel", "input_type"),
        [
            (F.vertical_flip_image_tensor, torch.Tensor),
            (F.vertical_flip_image_pil, PIL.Image.Image),
            (F.vertical_flip_image_tensor, datapoints.Image),
            (F.vertical_flip_bounding_box, datapoints.BoundingBox),
            (F.vertical_flip_mask, datapoints.Mask),
            (F.vertical_flip_video, datapoints.Video),
        ],
    )
    def test_dispatcher_signature(self, kernel, input_type):
        check_dispatcher_signatures_match(F.vertical_flip, kernel=kernel, input_type=input_type)

    @pytest.mark.parametrize(
        "make_input",
        [make_image_tensor, make_image_pil, make_image, make_bounding_box, make_segmentation_mask, make_video],
    )
    @pytest.mark.parametrize("device", cpu_and_cuda())
    def test_transform(self, make_input, device):
        check_transform(transforms.RandomVerticalFlip, make_input(device=device), p=1)

    @pytest.mark.parametrize("fn", [F.vertical_flip, transform_cls_to_functional(transforms.RandomVerticalFlip, p=1)])
    def test_image_correctness(self, fn):
        image = make_image(dtype=torch.uint8, device="cpu")

        actual = fn(image)
        expected = F.to_image_tensor(F.vertical_flip(F.to_image_pil(image)))

        torch.testing.assert_close(actual, expected)

    def _reference_vertical_flip_bounding_box(self, bounding_box):
        affine_matrix = np.array(
            [
                [1, 0, 0],
                [0, -1, bounding_box.spatial_size[0]],
            ],
            dtype="float64" if bounding_box.dtype == torch.float64 else "float32",
        )

        expected_bboxes = reference_affine_bounding_box_helper(
            bounding_box,
            format=bounding_box.format,
            spatial_size=bounding_box.spatial_size,
            affine_matrix=affine_matrix,
        )

        return datapoints.BoundingBox.wrap_like(bounding_box, expected_bboxes)

    @pytest.mark.parametrize("format", list(datapoints.BoundingBoxFormat))
    @pytest.mark.parametrize("fn", [F.vertical_flip, transform_cls_to_functional(transforms.RandomVerticalFlip, p=1)])
    def test_bounding_box_correctness(self, format, fn):
        bounding_box = make_bounding_box(format=format)

        actual = fn(bounding_box)
        expected = self._reference_vertical_flip_bounding_box(bounding_box)

        torch.testing.assert_close(actual, expected)

    @pytest.mark.parametrize(
        "make_input",
        [make_image_tensor, make_image_pil, make_image, make_bounding_box, make_segmentation_mask, make_video],
    )
    @pytest.mark.parametrize("device", cpu_and_cuda())
    def test_transform_noop(self, make_input, device):
        input = make_input(device=device)

        transform = transforms.RandomVerticalFlip(p=0)

        output = transform(input)

        assert_equal(output, input)


class TestRotate:
    _EXHAUSTIVE_TYPE_AFFINE_KWARGS = dict(
        # float, int
        angle=[-10.9, 18],
        # None
        # two-list of float, two-list of int, two-tuple of float, two-tuple of int
        center=[None, [1.2, 4.9], [-3, 1], (2.5, -4.7), (3, 2)],
    )
    _MINIMAL_AFFINE_KWARGS = {k: vs[0] for k, vs in _EXHAUSTIVE_TYPE_AFFINE_KWARGS.items()}
    _CORRECTNESS_AFFINE_KWARGS = {
        k: [v for v in vs if v is None or isinstance(v, float) or isinstance(v, list)]
        for k, vs in _EXHAUSTIVE_TYPE_AFFINE_KWARGS.items()
    }

    _EXHAUSTIVE_TYPE_TRANSFORM_AFFINE_RANGES = dict(
        degrees=[30, (-15, 20)],
    )
    _CORRECTNESS_TRANSFORM_AFFINE_RANGES = {k: vs[0] for k, vs in _EXHAUSTIVE_TYPE_TRANSFORM_AFFINE_RANGES.items()}

    @param_value_parametrization(
        angle=_EXHAUSTIVE_TYPE_AFFINE_KWARGS["angle"],
        interpolation=[transforms.InterpolationMode.NEAREST, transforms.InterpolationMode.BILINEAR],
        expand=[False, True],
        center=_EXHAUSTIVE_TYPE_AFFINE_KWARGS["center"],
        fill=EXHAUSTIVE_TYPE_FILLS,
    )
    @pytest.mark.parametrize("dtype", [torch.float32, torch.uint8])
    @pytest.mark.parametrize("device", cpu_and_cuda())
    def test_kernel_image_tensor(self, param, value, dtype, device):
        kwargs = {param: value}
        if param != "angle":
            kwargs["angle"] = self._MINIMAL_AFFINE_KWARGS["angle"]
        check_kernel(
            F.rotate_image_tensor,
            make_image(dtype=dtype, device=device),
            **kwargs,
            check_scripted_vs_eager=not (param == "fill" and isinstance(value, (int, float))),
        )

    @param_value_parametrization(
        angle=_EXHAUSTIVE_TYPE_AFFINE_KWARGS["angle"],
        expand=[False, True],
        center=_EXHAUSTIVE_TYPE_AFFINE_KWARGS["center"],
    )
    @pytest.mark.parametrize("format", list(datapoints.BoundingBoxFormat))
    @pytest.mark.parametrize("dtype", [torch.float32, torch.uint8])
    @pytest.mark.parametrize("device", cpu_and_cuda())
    def test_kernel_bounding_box(self, param, value, format, dtype, device):
        kwargs = {param: value}
        if param != "angle":
            kwargs["angle"] = self._MINIMAL_AFFINE_KWARGS["angle"]

        bounding_box = make_bounding_box(format=format, dtype=dtype, device=device)

        check_kernel(
            F.rotate_bounding_box,
            bounding_box,
            format=format,
            spatial_size=bounding_box.spatial_size,
            **kwargs,
        )

    @pytest.mark.parametrize("make_mask", [make_segmentation_mask, make_detection_mask])
    def test_kernel_mask(self, make_mask):
        check_kernel(F.rotate_mask, make_mask(), **self._MINIMAL_AFFINE_KWARGS)

    def test_kernel_video(self):
        check_kernel(F.rotate_video, make_video(), **self._MINIMAL_AFFINE_KWARGS)

    @pytest.mark.parametrize(
        ("kernel", "make_input"),
        [
            (F.rotate_image_tensor, make_image_tensor),
            (F.rotate_image_pil, make_image_pil),
            (F.rotate_image_tensor, make_image),
            (F.rotate_bounding_box, make_bounding_box),
            (F.rotate_mask, make_segmentation_mask),
            (F.rotate_video, make_video),
        ],
    )
    def test_dispatcher(self, kernel, make_input):
        check_dispatcher(F.rotate, kernel, make_input(), **self._MINIMAL_AFFINE_KWARGS)

    @pytest.mark.parametrize(
        ("kernel", "input_type"),
        [
            (F.rotate_image_tensor, torch.Tensor),
            (F.rotate_image_pil, PIL.Image.Image),
            (F.rotate_image_tensor, datapoints.Image),
            (F.rotate_bounding_box, datapoints.BoundingBox),
            (F.rotate_mask, datapoints.Mask),
            (F.rotate_video, datapoints.Video),
        ],
    )
    def test_dispatcher_signature(self, kernel, input_type):
        check_dispatcher_signatures_match(F.rotate, kernel=kernel, input_type=input_type)

    @pytest.mark.parametrize(
        "make_input",
        [make_image_tensor, make_image_pil, make_image, make_bounding_box, make_segmentation_mask, make_video],
    )
    @pytest.mark.parametrize("device", cpu_and_cuda())
    def test_transform(self, make_input, device):
        check_transform(
            transforms.RandomRotation, make_input(device=device), **self._CORRECTNESS_TRANSFORM_AFFINE_RANGES
        )

    @pytest.mark.parametrize("angle", _CORRECTNESS_AFFINE_KWARGS["angle"])
    @pytest.mark.parametrize("center", _CORRECTNESS_AFFINE_KWARGS["center"])
    @pytest.mark.parametrize(
        "interpolation", [transforms.InterpolationMode.NEAREST, transforms.InterpolationMode.BILINEAR]
    )
    @pytest.mark.parametrize("expand", [False, True])
    @pytest.mark.parametrize("fill", CORRECTNESS_FILLS)
    def test_functional_image_correctness(self, angle, center, interpolation, expand, fill):
        image = make_image(dtype=torch.uint8, device="cpu")

        fill = adapt_fill(fill, dtype=torch.uint8)

        actual = F.rotate(image, angle=angle, center=center, interpolation=interpolation, expand=expand, fill=fill)
        expected = F.to_image_tensor(
            F.rotate(
                F.to_image_pil(image), angle=angle, center=center, interpolation=interpolation, expand=expand, fill=fill
            )
        )

        mae = (actual.float() - expected.float()).abs().mean()
        assert mae < 1 if interpolation is transforms.InterpolationMode.NEAREST else 6

    @pytest.mark.parametrize("center", _CORRECTNESS_AFFINE_KWARGS["center"])
    @pytest.mark.parametrize(
        "interpolation", [transforms.InterpolationMode.NEAREST, transforms.InterpolationMode.BILINEAR]
    )
    @pytest.mark.parametrize("expand", [False, True])
    @pytest.mark.parametrize("fill", CORRECTNESS_FILLS)
    @pytest.mark.parametrize("seed", list(range(5)))
    def test_transform_image_correctness(self, center, interpolation, expand, fill, seed):
        image = make_image(dtype=torch.uint8, device="cpu")

        fill = adapt_fill(fill, dtype=torch.uint8)

        transform = transforms.RandomRotation(
            **self._CORRECTNESS_TRANSFORM_AFFINE_RANGES,
            center=center,
            interpolation=interpolation,
            expand=expand,
            fill=fill,
        )

        torch.manual_seed(seed)
        actual = transform(image)

        torch.manual_seed(seed)
        expected = F.to_image_tensor(transform(F.to_image_pil(image)))

        mae = (actual.float() - expected.float()).abs().mean()
        assert mae < 1 if interpolation is transforms.InterpolationMode.NEAREST else 6

    def _reference_rotate_bounding_box(self, bounding_box, *, angle, expand, center):
        # FIXME
        if expand:
            raise ValueError("This reference currently does not support expand=True")

        if center is None:
            center = [s * 0.5 for s in bounding_box.spatial_size[::-1]]

        a = np.cos(angle * np.pi / 180.0)
        b = np.sin(angle * np.pi / 180.0)
        cx = center[0]
        cy = center[1]
        affine_matrix = np.array(
            [
                [a, b, cx - cx * a - b * cy],
                [-b, a, cy + cx * b - a * cy],
            ],
            dtype="float64" if bounding_box.dtype == torch.float64 else "float32",
        )

        expected_bboxes = reference_affine_bounding_box_helper(
            bounding_box,
            format=bounding_box.format,
            spatial_size=bounding_box.spatial_size,
            affine_matrix=affine_matrix,
        )

        return expected_bboxes

    @pytest.mark.parametrize("format", list(datapoints.BoundingBoxFormat))
    @pytest.mark.parametrize("angle", _CORRECTNESS_AFFINE_KWARGS["angle"])
    # TODO: add support for expand=True in the reference
    @pytest.mark.parametrize("expand", [False])
    @pytest.mark.parametrize("center", _CORRECTNESS_AFFINE_KWARGS["center"])
    def test_functional_bounding_box_correctness(self, format, angle, expand, center):
        bounding_box = make_bounding_box(format=format)

        actual = F.rotate(bounding_box, angle=angle, expand=expand, center=center)
        expected = self._reference_rotate_bounding_box(bounding_box, angle=angle, expand=expand, center=center)

        torch.testing.assert_close(actual, expected)

    @pytest.mark.parametrize("format", list(datapoints.BoundingBoxFormat))
    # TODO: add support for expand=True in the reference
    @pytest.mark.parametrize("expand", [False])
    @pytest.mark.parametrize("center", _CORRECTNESS_AFFINE_KWARGS["center"])
    @pytest.mark.parametrize("seed", list(range(5)))
    def test_transform_bounding_box_correctness(self, format, expand, center, seed):
        bounding_box = make_bounding_box(format=format)

        transform = transforms.RandomRotation(**self._CORRECTNESS_TRANSFORM_AFFINE_RANGES, expand=expand, center=center)

        torch.manual_seed(seed)
        params = transform._get_params([bounding_box])

        torch.manual_seed(seed)
        actual = transform(bounding_box)

        expected = self._reference_rotate_bounding_box(bounding_box, **params, expand=expand, center=center)

        torch.testing.assert_close(actual, expected)

    @pytest.mark.parametrize("degrees", _EXHAUSTIVE_TYPE_TRANSFORM_AFFINE_RANGES["degrees"])
    @pytest.mark.parametrize("seed", list(range(10)))
    def test_transform_get_params_bounds(self, degrees, seed):
        transform = transforms.RandomRotation(degrees=degrees)

        torch.manual_seed(seed)
        params = transform._get_params([])

        if isinstance(degrees, (int, float)):
            assert -degrees <= params["angle"] <= degrees
        else:
            assert degrees[0] <= params["angle"] <= degrees[1]

    @pytest.mark.parametrize("param", ["degrees", "center"])
    @pytest.mark.parametrize("value", [0, [0], [0, 0, 0]])
    def test_transform_sequence_len_errors(self, param, value):
        if param == "degrees" and not isinstance(value, list):
            return

        kwargs = {param: value}
        if param != "degrees":
            kwargs["degrees"] = 0

        with pytest.raises(
            ValueError if isinstance(value, list) else TypeError, match=f"{param} should be a sequence of length 2"
        ):
            transforms.RandomRotation(**kwargs)

    def test_transform_negative_degrees_error(self):
        with pytest.raises(ValueError, match="If degrees is a single number, it must be positive"):
            transforms.RandomAffine(degrees=-1)

    def test_transform_unknown_fill_error(self):
        with pytest.raises(TypeError, match="Got inappropriate fill arg"):
            transforms.RandomAffine(degrees=0, fill="fill")


class TestCutMixMixUp:
    class DummyDataset:
        def __init__(self, size, one_hot, num_categories):
            self.one_hot = one_hot
            self.size = size
            self.num_categories = num_categories
            assert size < num_categories

        def __getitem__(self, idx):
            img = torch.rand(3, 100, 100)
            label = idx  # This ensures all labels in a batch are unique and makes testing easier
            if self.one_hot:
                label = torch.nn.functional.one_hot(torch.tensor(label), num_classes=self.num_categories)
            return img, label

        def __len__(self):
            return self.size

    @pytest.mark.parametrize("T", [transforms.Cutmix, transforms.Mixup, "CutMixMixUp", "MixUpCutMix"])
    @pytest.mark.parametrize("one_hot", [True, False])
    def test_supported_input_structure(self, T, one_hot):

        batch_size = 32
        num_categories = 100

        dataset = self.DummyDataset(size=batch_size, one_hot=one_hot, num_categories=num_categories)

        if isinstance(T, str):
            cutmix = transforms.Cutmix(alpha=0.5, num_categories=num_categories)
            mixup = transforms.Mixup(alpha=0.5, num_categories=num_categories)
            if T == "CutMixMixUp":
                cutmix_mixup = transforms.Compose([cutmix, mixup])
            else:
                cutmix_mixup = transforms.Compose([mixup, cutmix])
            expected_num_non_zero_labels = 3
        else:
            cutmix_mixup = T(alpha=0.5, num_categories=num_categories)
            expected_num_non_zero_labels = 2

        dl = DataLoader(dataset, batch_size=batch_size)

        # Input sanity checks
        img, target = next(iter(dl))
        input_img_size = img.shape[-3:]
        assert isinstance(img, torch.Tensor) and isinstance(target, torch.Tensor)
        assert target.shape == (batch_size, num_categories) if one_hot else (batch_size,)

        def check_output(img, target):
            assert img.shape == (batch_size, *input_img_size)
            assert target.shape == (batch_size, num_categories)
            torch.testing.assert_close(target.sum(axis=-1), torch.ones(batch_size))
            num_non_zero_labels = (target != 0).sum(axis=-1)
            assert (num_non_zero_labels == expected_num_non_zero_labels).all()

        # After Dataloader, as unpacked input
        img, target = next(iter(dl))
        assert target.shape == (batch_size, num_categories) if one_hot else (batch_size,)
        img, target = cutmix_mixup(img, target)
        check_output(img, target)

        # After Dataloader, as packed input
        packed_from_dl = next(iter(dl))
        assert isinstance(packed_from_dl, list)
        img, target = cutmix_mixup(packed_from_dl)
        check_output(img, target)

        # As collation function. We expect default_collate to be used by users.
        def collate_fn_1(batch):
            return cutmix_mixup(default_collate(batch))

        def collate_fn_2(batch):
            return cutmix_mixup(*default_collate(batch))

        for collate_fn in (collate_fn_1, collate_fn_2):
            dl = DataLoader(dataset, batch_size=batch_size, collate_fn=collate_fn)
            img, target = next(iter(dl))
            check_output(img, target)

    @needs_cuda
    @pytest.mark.parametrize("T", [transforms.Cutmix, transforms.Mixup])
    def test_cpu_vs_gpu(self, T):
        num_categories = 10
        batch_size = 3
        H, W = 12, 12

        imgs = torch.rand(batch_size, 3, H, W).to("cuda")
        labels = torch.randint(0, num_categories, (batch_size,)).to("cuda")
        cutmix_mixup = T(alpha=0.5, num_categories=num_categories)

        _check_kernel_cuda_vs_cpu(cutmix_mixup, input=(imgs, labels), rtol=None, atol=None)

    @pytest.mark.parametrize("T", [transforms.Cutmix, transforms.Mixup])
    def test_error(self, T):

        num_categories = 10
        batch_size = 9

        imgs = torch.rand(batch_size, 3, 12, 12)
        cutmix_mixup = T(alpha=0.5)

        for input_with_bad_type in (
            F.to_pil_image(imgs[0]),
            datapoints.Mask(torch.rand(12, 12)),
            datapoints.BoundingBox(torch.rand(2, 4), format="XYXY", spatial_size=12),
        ):
            with pytest.raises(ValueError, match="does not support PIL images, "):
                cutmix_mixup(input_with_bad_type)

        with pytest.raises(ValueError, match="Could not infer where the labels are"):
            cutmix_mixup({"img": imgs, "Nothing_else": 3})

        with pytest.raises(ValueError, match="labels should be index based"):
            # Note: the error message isn't ideal, but that's because the label heuristic found the img as the label
            # It's OK, it's an edge-case. The important thing is that this fails loudly instead of passing silently
            cutmix_mixup(imgs)

        with pytest.raises(ValueError, match="When using the default labels_getter"):
            cutmix_mixup(imgs, "not_a_tensor")

        with pytest.raises(ValueError, match="When passing 2D labels"):
            wrong_num_categories = num_categories + 1
            T(alpha=0.5, num_categories=num_categories)(
                imgs, torch.randint(0, 2, size=(batch_size, wrong_num_categories))
            )

        with pytest.raises(ValueError, match="but got a tensor of shape"):
            cutmix_mixup(imgs, torch.randint(0, 2, size=(2, 3, 4)))

        with pytest.raises(ValueError, match="Expected a batched input with 4 dims"):
            cutmix_mixup(imgs[None, None], torch.randint(0, num_categories, size=(batch_size,)))

        with pytest.raises(ValueError, match="does not match the batch size of the labels"):
            cutmix_mixup(imgs, torch.randint(0, num_categories, size=(batch_size + 1,)))

        with pytest.raises(ValueError, match="num_categories must be passed"):
            cutmix_mixup(imgs, torch.randint(0, num_categories, size=(batch_size,)))
