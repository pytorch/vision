import contextlib
import decimal
import functools
import inspect
import itertools
import math
import pickle
import random
import re
import sys
from copy import deepcopy
from pathlib import Path
from unittest import mock

import numpy as np
import PIL.Image
import pytest

import torch
import torchvision.ops
import torchvision.transforms.v2 as transforms

from common_utils import (
    assert_equal,
    cache,
    cpu_and_cuda,
    freeze_rng_state,
    ignore_jit_no_profile_information_warning,
    make_bounding_boxes,
    make_detection_masks,
    make_image,
    make_image_pil,
    make_image_tensor,
    make_segmentation_mask,
    make_video,
    make_video_tensor,
    needs_cuda,
    set_rng_seed,
)

from torch import nn
from torch.testing import assert_close
from torch.utils._pytree import tree_flatten, tree_map
from torch.utils.data import DataLoader, default_collate
from torchvision import tv_tensors
from torchvision.ops.boxes import box_iou

from torchvision.transforms._functional_tensor import _max_value as get_max_value
from torchvision.transforms.functional import pil_modes_mapping, to_pil_image
from torchvision.transforms.v2 import functional as F
from torchvision.transforms.v2._utils import check_type, is_pure_tensor
from torchvision.transforms.v2.functional._geometry import _get_perspective_coeffs
from torchvision.transforms.v2.functional._utils import _get_kernel, _register_kernel_internal


# turns all warnings into errors for this module
pytestmark = pytest.mark.filterwarnings("error")


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

    with freeze_rng_state():
        actual = kernel(input_cuda, *args, **kwargs)
    with freeze_rng_state():
        expected = kernel(input_cpu, *args, **kwargs)

    assert_close(actual, expected, check_device=False, rtol=rtol, atol=atol)


@cache
def _script(obj):
    try:
        return torch.jit.script(obj)
    except Exception as error:
        name = getattr(obj, "__name__", obj.__class__.__name__)
        raise AssertionError(f"Trying to `torch.jit.script` '{name}' raised the error above.") from error


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

    if kernel not in {F.to_dtype_image, F.to_dtype_video}:
        assert output.dtype == input.dtype
    assert output.device == input.device

    if check_cuda_vs_cpu:
        _check_kernel_cuda_vs_cpu(kernel, input, *args, **kwargs, **_to_tolerances(check_cuda_vs_cpu))

    if check_scripted_vs_eager:
        _check_kernel_scripted_vs_eager(kernel, input, *args, **kwargs, **_to_tolerances(check_scripted_vs_eager))

    if check_batched_vs_unbatched:
        _check_kernel_batched_vs_unbatched(kernel, input, *args, **kwargs, **_to_tolerances(check_batched_vs_unbatched))


def _check_functional_scripted_smoke(functional, input, *args, **kwargs):
    """Checks if the functional can be scripted and the scripted version can be called without error."""
    if not isinstance(input, tv_tensors.Image):
        return

    functional_scripted = _script(functional)
    with ignore_jit_no_profile_information_warning():
        functional_scripted(input.as_subclass(torch.Tensor), *args, **kwargs)


def check_functional(functional, input, *args, check_scripted_smoke=True, **kwargs):
    unknown_input = object()
    with pytest.raises(TypeError, match=re.escape(str(type(unknown_input)))):
        functional(unknown_input, *args, **kwargs)

    with mock.patch("torch._C._log_api_usage_once", wraps=torch._C._log_api_usage_once) as spy:
        output = functional(input, *args, **kwargs)

        spy.assert_any_call(f"{functional.__module__}.{functional.__name__}")

    assert isinstance(output, type(input))

    if isinstance(input, tv_tensors.BoundingBoxes) and functional is not F.convert_bounding_box_format:
        assert output.format == input.format

    if check_scripted_smoke:
        _check_functional_scripted_smoke(functional, input, *args, **kwargs)


def check_functional_kernel_signature_match(functional, *, kernel, input_type):
    """Checks if the signature of the functional matches the kernel signature."""
    functional_params = list(inspect.signature(functional).parameters.values())[1:]
    kernel_params = list(inspect.signature(kernel).parameters.values())[1:]

    if issubclass(input_type, tv_tensors.TVTensor):
        # We filter out metadata that is implicitly passed to the functional through the input tv_tensor, but has to be
        # explicitly passed to the kernel.
        explicit_metadata = {
            tv_tensors.BoundingBoxes: {"format", "canvas_size"},
        }
        kernel_params = [param for param in kernel_params if param.name not in explicit_metadata.get(input_type, set())]

    functional_params = iter(functional_params)
    for functional_param, kernel_param in zip(functional_params, kernel_params):
        try:
            # In general, the functional parameters are a superset of the kernel parameters. Thus, we filter out
            # functional parameters that have no kernel equivalent while keeping the order intact.
            while functional_param.name != kernel_param.name:
                functional_param = next(functional_params)
        except StopIteration:
            raise AssertionError(
                f"Parameter `{kernel_param.name}` of kernel `{kernel.__name__}` "
                f"has no corresponding parameter on the functional `{functional.__name__}`."
            ) from None

        if issubclass(input_type, PIL.Image.Image):
            # PIL kernels often have more correct annotations, since they are not limited by JIT. Thus, we don't check
            # them in the first place.
            functional_param._annotation = kernel_param._annotation = inspect.Parameter.empty

        assert functional_param == kernel_param


def _check_transform_v1_compatibility(transform, input, *, rtol, atol):
    """If the transform defines the ``_v1_transform_cls`` attribute, checks if the transform has a public, static
    ``get_params`` method that is the v1 equivalent, the output is close to v1, is scriptable, and the scripted version
    can be called without error."""
    if not (type(input) is torch.Tensor or isinstance(input, PIL.Image.Image)):
        return

    v1_transform_cls = transform._v1_transform_cls
    if v1_transform_cls is None:
        return

    if hasattr(v1_transform_cls, "get_params"):
        assert type(transform).get_params is v1_transform_cls.get_params

    v1_transform = v1_transform_cls(**transform._extract_params_for_v1_transform())

    with freeze_rng_state():
        output_v2 = transform(input)

    with freeze_rng_state():
        output_v1 = v1_transform(input)

    assert_close(F.to_image(output_v2), F.to_image(output_v1), rtol=rtol, atol=atol)

    if isinstance(input, PIL.Image.Image):
        return

    _script(v1_transform)(input)


def _make_transform_sample(transform, *, image_or_video, adapter):
    device = image_or_video.device if isinstance(image_or_video, torch.Tensor) else "cpu"
    size = F.get_size(image_or_video)
    input = dict(
        image_or_video=image_or_video,
        image_tv_tensor=make_image(size, device=device),
        video_tv_tensor=make_video(size, device=device),
        image_pil=make_image_pil(size),
        bounding_boxes_xyxy=make_bounding_boxes(size, format=tv_tensors.BoundingBoxFormat.XYXY, device=device),
        bounding_boxes_xywh=make_bounding_boxes(size, format=tv_tensors.BoundingBoxFormat.XYWH, device=device),
        bounding_boxes_cxcywh=make_bounding_boxes(size, format=tv_tensors.BoundingBoxFormat.CXCYWH, device=device),
        bounding_boxes_degenerate_xyxy=tv_tensors.BoundingBoxes(
            [
                [0, 0, 0, 0],  # no height or width
                [0, 0, 0, 1],  # no height
                [0, 0, 1, 0],  # no width
                [2, 0, 1, 1],  # x1 > x2, y1 < y2
                [0, 2, 1, 1],  # x1 < x2, y1 > y2
                [2, 2, 1, 1],  # x1 > x2, y1 > y2
            ],
            format=tv_tensors.BoundingBoxFormat.XYXY,
            canvas_size=size,
            device=device,
        ),
        bounding_boxes_degenerate_xywh=tv_tensors.BoundingBoxes(
            [
                [0, 0, 0, 0],  # no height or width
                [0, 0, 0, 1],  # no height
                [0, 0, 1, 0],  # no width
                [0, 0, 1, -1],  # negative height
                [0, 0, -1, 1],  # negative width
                [0, 0, -1, -1],  # negative height and width
            ],
            format=tv_tensors.BoundingBoxFormat.XYWH,
            canvas_size=size,
            device=device,
        ),
        bounding_boxes_degenerate_cxcywh=tv_tensors.BoundingBoxes(
            [
                [0, 0, 0, 0],  # no height or width
                [0, 0, 0, 1],  # no height
                [0, 0, 1, 0],  # no width
                [0, 0, 1, -1],  # negative height
                [0, 0, -1, 1],  # negative width
                [0, 0, -1, -1],  # negative height and width
            ],
            format=tv_tensors.BoundingBoxFormat.CXCYWH,
            canvas_size=size,
            device=device,
        ),
        detection_mask=make_detection_masks(size, device=device),
        segmentation_mask=make_segmentation_mask(size, device=device),
        int=0,
        float=0.0,
        bool=True,
        none=None,
        str="str",
        path=Path.cwd(),
        object=object(),
        tensor=torch.empty(5),
        array=np.empty(5),
    )
    if adapter is not None:
        input = adapter(transform, input, device)
    return input


def _check_transform_sample_input_smoke(transform, input, *, adapter):
    # This is a bunch of input / output convention checks, using a big sample with different parts as input.

    if not check_type(input, (is_pure_tensor, PIL.Image.Image, tv_tensors.Image, tv_tensors.Video)):
        return

    sample = _make_transform_sample(
        # adapter might change transform inplace
        transform=transform if adapter is None else deepcopy(transform),
        image_or_video=input,
        adapter=adapter,
    )
    for container_type in [dict, list, tuple]:
        if container_type is dict:
            input = sample
        else:
            input = container_type(sample.values())

        input_flat, input_spec = tree_flatten(input)

        with freeze_rng_state():
            torch.manual_seed(0)
            output = transform(input)
        output_flat, output_spec = tree_flatten(output)

        assert output_spec == input_spec

        for output_item, input_item, should_be_transformed in zip(
            output_flat, input_flat, transforms.Transform()._needs_transform_list(input_flat)
        ):
            if should_be_transformed:
                assert type(output_item) is type(input_item)
            else:
                assert output_item is input_item

    # Enforce that the transform does not turn a degenerate bounding box, e.g. marked by RandomIoUCrop (or any other
    # future transform that does this), back into a valid one.
    for degenerate_bounding_boxes in (
        bounding_box
        for name, bounding_box in sample.items()
        if "degenerate" in name and isinstance(bounding_box, tv_tensors.BoundingBoxes)
    ):
        sample = dict(
            boxes=degenerate_bounding_boxes,
            labels=torch.randint(10, (degenerate_bounding_boxes.shape[0],), device=degenerate_bounding_boxes.device),
        )
        assert transforms.SanitizeBoundingBoxes()(sample)["boxes"].shape == (0, 4)


def check_transform(transform, input, check_v1_compatibility=True, check_sample_input=True):
    pickle.loads(pickle.dumps(transform))

    output = transform(input)
    assert isinstance(output, type(input))

    if isinstance(input, tv_tensors.BoundingBoxes) and not isinstance(transform, transforms.ConvertBoundingBoxFormat):
        assert output.format == input.format

    if check_sample_input:
        _check_transform_sample_input_smoke(
            transform, input, adapter=check_sample_input if callable(check_sample_input) else None
        )

    if check_v1_compatibility:
        _check_transform_v1_compatibility(transform, input, **_to_tolerances(check_v1_compatibility))

    return output


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
    value_type = float if dtype.is_floating_point else int

    if isinstance(value, (int, float)):
        return value_type(value * max_value)
    elif isinstance(value, (list, tuple)):
        return type(value)(value_type(v * max_value) for v in value)
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


def reference_affine_bounding_boxes_helper(bounding_boxes, *, affine_matrix, new_canvas_size=None, clamp=True):
    format = bounding_boxes.format
    canvas_size = new_canvas_size or bounding_boxes.canvas_size

    def affine_bounding_boxes(bounding_boxes):
        dtype = bounding_boxes.dtype
        device = bounding_boxes.device

        # Go to float before converting to prevent precision loss in case of CXCYWH -> XYXY and W or H is 1
        input_xyxy = F.convert_bounding_box_format(
            bounding_boxes.to(dtype=torch.float64, device="cpu", copy=True),
            old_format=format,
            new_format=tv_tensors.BoundingBoxFormat.XYXY,
            inplace=True,
        )
        x1, y1, x2, y2 = input_xyxy.squeeze(0).tolist()

        points = np.array(
            [
                [x1, y1, 1.0],
                [x2, y1, 1.0],
                [x1, y2, 1.0],
                [x2, y2, 1.0],
            ]
        )
        transformed_points = np.matmul(points, affine_matrix.astype(points.dtype).T)

        output_xyxy = torch.Tensor(
            [
                float(np.min(transformed_points[:, 0])),
                float(np.min(transformed_points[:, 1])),
                float(np.max(transformed_points[:, 0])),
                float(np.max(transformed_points[:, 1])),
            ]
        )

        output = F.convert_bounding_box_format(
            output_xyxy, old_format=tv_tensors.BoundingBoxFormat.XYXY, new_format=format
        )

        if clamp:
            # It is important to clamp before casting, especially for CXCYWH format, dtype=int64
            output = F.clamp_bounding_boxes(
                output,
                format=format,
                canvas_size=canvas_size,
            )
        else:
            # We leave the bounding box as float64 so the caller gets the full precision to perform any additional
            # operation
            dtype = output.dtype

        return output.to(dtype=dtype, device=device)

    return tv_tensors.BoundingBoxes(
        torch.cat([affine_bounding_boxes(b) for b in bounding_boxes.reshape(-1, 4).unbind()], dim=0).reshape(
            bounding_boxes.shape
        ),
        format=format,
        canvas_size=canvas_size,
    )


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
    def test_kernel_image(self, size, interpolation, use_max_size, antialias, dtype, device):
        if not (max_size_kwarg := self._make_max_size_kwarg(use_max_size=use_max_size, size=size)):
            return

        # In contrast to CPU, there is no native `InterpolationMode.BICUBIC` implementation for uint8 images on CUDA.
        # Internally, it uses the float path. Thus, we need to test with an enormous tolerance here to account for that.
        atol = 30 if (interpolation is transforms.InterpolationMode.BICUBIC and dtype is torch.uint8) else 1
        check_cuda_vs_cpu_tolerances = dict(rtol=0, atol=atol / 255 if dtype.is_floating_point else atol)

        check_kernel(
            F.resize_image,
            make_image(self.INPUT_SIZE, dtype=dtype, device=device),
            size=size,
            interpolation=interpolation,
            **max_size_kwarg,
            antialias=antialias,
            check_cuda_vs_cpu=check_cuda_vs_cpu_tolerances,
            check_scripted_vs_eager=not isinstance(size, int),
        )

    @pytest.mark.parametrize("format", list(tv_tensors.BoundingBoxFormat))
    @pytest.mark.parametrize("size", OUTPUT_SIZES)
    @pytest.mark.parametrize("use_max_size", [True, False])
    @pytest.mark.parametrize("dtype", [torch.float32, torch.int64])
    @pytest.mark.parametrize("device", cpu_and_cuda())
    def test_kernel_bounding_boxes(self, format, size, use_max_size, dtype, device):
        if not (max_size_kwarg := self._make_max_size_kwarg(use_max_size=use_max_size, size=size)):
            return

        bounding_boxes = make_bounding_boxes(
            format=format,
            canvas_size=self.INPUT_SIZE,
            dtype=dtype,
            device=device,
        )
        check_kernel(
            F.resize_bounding_boxes,
            bounding_boxes,
            canvas_size=bounding_boxes.canvas_size,
            size=size,
            **max_size_kwarg,
            check_scripted_vs_eager=not isinstance(size, int),
        )

    @pytest.mark.parametrize("make_mask", [make_segmentation_mask, make_detection_masks])
    def test_kernel_mask(self, make_mask):
        check_kernel(F.resize_mask, make_mask(self.INPUT_SIZE), size=self.OUTPUT_SIZES[-1])

    def test_kernel_video(self):
        check_kernel(F.resize_video, make_video(self.INPUT_SIZE), size=self.OUTPUT_SIZES[-1], antialias=True)

    @pytest.mark.parametrize("size", OUTPUT_SIZES)
    @pytest.mark.parametrize(
        "make_input",
        [make_image_tensor, make_image_pil, make_image, make_bounding_boxes, make_segmentation_mask, make_video],
    )
    def test_functional(self, size, make_input):
        check_functional(
            F.resize,
            make_input(self.INPUT_SIZE),
            size=size,
            antialias=True,
            check_scripted_smoke=not isinstance(size, int),
        )

    @pytest.mark.parametrize(
        ("kernel", "input_type"),
        [
            (F.resize_image, torch.Tensor),
            (F._resize_image_pil, PIL.Image.Image),
            (F.resize_image, tv_tensors.Image),
            (F.resize_bounding_boxes, tv_tensors.BoundingBoxes),
            (F.resize_mask, tv_tensors.Mask),
            (F.resize_video, tv_tensors.Video),
        ],
    )
    def test_functional_signature(self, kernel, input_type):
        check_functional_kernel_signature_match(F.resize, kernel=kernel, input_type=input_type)

    @pytest.mark.parametrize("size", OUTPUT_SIZES)
    @pytest.mark.parametrize("device", cpu_and_cuda())
    @pytest.mark.parametrize(
        "make_input",
        [
            make_image_tensor,
            make_image_pil,
            make_image,
            make_bounding_boxes,
            make_segmentation_mask,
            make_detection_masks,
            make_video,
        ],
    )
    def test_transform(self, size, device, make_input):
        check_transform(
            transforms.Resize(size=size, antialias=True),
            make_input(self.INPUT_SIZE, device=device),
            # atol=1 due to Resize v2 is using native uint8 interpolate path for bilinear and nearest modes
            check_v1_compatibility=dict(rtol=0, atol=1),
        )

    def _check_output_size(self, input, output, *, size, max_size):
        assert tuple(F.get_size(output)) == self._compute_output_size(
            input_size=F.get_size(input), size=size, max_size=max_size
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
        expected = F.to_image(F.resize(F.to_pil_image(image), size=size, interpolation=interpolation, **max_size_kwarg))

        self._check_output_size(image, actual, size=size, **max_size_kwarg)
        torch.testing.assert_close(actual, expected, atol=1, rtol=0)

    def _reference_resize_bounding_boxes(self, bounding_boxes, *, size, max_size=None):
        old_height, old_width = bounding_boxes.canvas_size
        new_height, new_width = self._compute_output_size(
            input_size=bounding_boxes.canvas_size, size=size, max_size=max_size
        )

        if (old_height, old_width) == (new_height, new_width):
            return bounding_boxes

        affine_matrix = np.array(
            [
                [new_width / old_width, 0, 0],
                [0, new_height / old_height, 0],
            ],
        )

        return reference_affine_bounding_boxes_helper(
            bounding_boxes,
            affine_matrix=affine_matrix,
            new_canvas_size=(new_height, new_width),
        )

    @pytest.mark.parametrize("format", list(tv_tensors.BoundingBoxFormat))
    @pytest.mark.parametrize("size", OUTPUT_SIZES)
    @pytest.mark.parametrize("use_max_size", [True, False])
    @pytest.mark.parametrize("fn", [F.resize, transform_cls_to_functional(transforms.Resize)])
    def test_bounding_boxes_correctness(self, format, size, use_max_size, fn):
        if not (max_size_kwarg := self._make_max_size_kwarg(use_max_size=use_max_size, size=size)):
            return

        bounding_boxes = make_bounding_boxes(format=format, canvas_size=self.INPUT_SIZE)

        actual = fn(bounding_boxes, size=size, **max_size_kwarg)
        expected = self._reference_resize_bounding_boxes(bounding_boxes, size=size, **max_size_kwarg)

        self._check_output_size(bounding_boxes, actual, size=size, **max_size_kwarg)
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

    def test_functional_pil_antialias_warning(self):
        with pytest.warns(UserWarning, match="Anti-alias option is always applied for PIL Image input"):
            F.resize(make_image_pil(self.INPUT_SIZE), size=self.OUTPUT_SIZES[0], antialias=False)

    @pytest.mark.parametrize("size", OUTPUT_SIZES)
    @pytest.mark.parametrize(
        "make_input",
        [
            make_image_tensor,
            make_image_pil,
            make_image,
            make_bounding_boxes,
            make_segmentation_mask,
            make_detection_masks,
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
        with pytest.raises(ValueError, match="size can either be an integer or a sequence of one or two integers"):
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
            make_bounding_boxes,
            make_segmentation_mask,
            make_detection_masks,
            make_video,
        ],
    )
    def test_noop(self, size, make_input):
        input = make_input(self.INPUT_SIZE)

        output = F.resize(input, size=F.get_size(input), antialias=True)

        # This identity check is not a requirement. It is here to avoid breaking the behavior by accident. If there
        # is a good reason to break this, feel free to downgrade to an equality check.
        if isinstance(input, tv_tensors.TVTensor):
            # We can't test identity directly, since that checks for the identity of the Python object. Since all
            # tv_tensors unwrap before a kernel and wrap again afterwards, the Python object changes. Thus, we check
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
            make_bounding_boxes,
            make_segmentation_mask,
            make_detection_masks,
            make_video,
        ],
    )
    def test_no_regression_5405(self, make_input):
        # Checks that `max_size` is not ignored if `size == small_edge_size`
        # See https://github.com/pytorch/vision/issues/5405

        input = make_input(self.INPUT_SIZE)

        size = min(F.get_size(input))
        max_size = size + 1
        output = F.resize(input, size=size, max_size=max_size, antialias=True)

        assert max(F.get_size(output)) == max_size

    def _make_image(self, *args, batch_dims=(), memory_format=torch.contiguous_format, **kwargs):
        # torch.channels_last memory_format is only available for 4D tensors, i.e. (B, C, H, W). However, images coming
        # from PIL or our own I/O functions do not have a batch dimensions and are thus 3D, i.e. (C, H, W). Still, the
        # layout of the data in memory is channels last. To emulate this when a 3D input is requested here, we create
        # the image as 4D and create a view with the right shape afterwards. With this the layout in memory is channels
        # last although PyTorch doesn't recognizes it as such.
        emulate_channels_last = memory_format is torch.channels_last and len(batch_dims) != 1

        image = make_image(
            *args,
            batch_dims=(math.prod(batch_dims),) if emulate_channels_last else batch_dims,
            memory_format=memory_format,
            **kwargs,
        )

        if emulate_channels_last:
            image = tv_tensors.wrap(image.view(*batch_dims, *image.shape[-3:]), like=image)

        return image

    def _check_stride(self, image, *, memory_format):
        C, H, W = F.get_dimensions(image)
        if memory_format is torch.contiguous_format:
            expected_stride = (H * W, W, 1)
        elif memory_format is torch.channels_last:
            expected_stride = (1, W * C, C)
        else:
            raise ValueError(f"Unknown memory_format: {memory_format}")

        assert image.stride() == expected_stride

    # TODO: We can remove this test and related torchvision workaround
    #  once we fixed related pytorch issue: https://github.com/pytorch/pytorch/issues/68430
    @pytest.mark.parametrize("interpolation", INTERPOLATION_MODES)
    @pytest.mark.parametrize("antialias", [True, False])
    @pytest.mark.parametrize("memory_format", [torch.contiguous_format, torch.channels_last])
    @pytest.mark.parametrize("dtype", [torch.uint8, torch.float32])
    @pytest.mark.parametrize("device", cpu_and_cuda())
    def test_kernel_image_memory_format_consistency(self, interpolation, antialias, memory_format, dtype, device):
        size = self.OUTPUT_SIZES[0]

        input = self._make_image(self.INPUT_SIZE, dtype=dtype, device=device, memory_format=memory_format)

        # Smoke test to make sure we aren't starting with wrong assumptions
        self._check_stride(input, memory_format=memory_format)

        output = F.resize_image(input, size=size, interpolation=interpolation, antialias=antialias)

        self._check_stride(output, memory_format=memory_format)

    def test_float16_no_rounding(self):
        # Make sure Resize() doesn't round float16 images
        # Non-regression test for https://github.com/pytorch/vision/issues/7667

        input = make_image_tensor(self.INPUT_SIZE, dtype=torch.float16)
        output = F.resize_image(input, size=self.OUTPUT_SIZES[0], antialias=True)

        assert output.dtype is torch.float16
        assert (output.round() - output).abs().sum() > 0


class TestHorizontalFlip:
    @pytest.mark.parametrize("dtype", [torch.float32, torch.uint8])
    @pytest.mark.parametrize("device", cpu_and_cuda())
    def test_kernel_image(self, dtype, device):
        check_kernel(F.horizontal_flip_image, make_image(dtype=dtype, device=device))

    @pytest.mark.parametrize("format", list(tv_tensors.BoundingBoxFormat))
    @pytest.mark.parametrize("dtype", [torch.float32, torch.int64])
    @pytest.mark.parametrize("device", cpu_and_cuda())
    def test_kernel_bounding_boxes(self, format, dtype, device):
        bounding_boxes = make_bounding_boxes(format=format, dtype=dtype, device=device)
        check_kernel(
            F.horizontal_flip_bounding_boxes,
            bounding_boxes,
            format=format,
            canvas_size=bounding_boxes.canvas_size,
        )

    @pytest.mark.parametrize("make_mask", [make_segmentation_mask, make_detection_masks])
    def test_kernel_mask(self, make_mask):
        check_kernel(F.horizontal_flip_mask, make_mask())

    def test_kernel_video(self):
        check_kernel(F.horizontal_flip_video, make_video())

    @pytest.mark.parametrize(
        "make_input",
        [make_image_tensor, make_image_pil, make_image, make_bounding_boxes, make_segmentation_mask, make_video],
    )
    def test_functional(self, make_input):
        check_functional(F.horizontal_flip, make_input())

    @pytest.mark.parametrize(
        ("kernel", "input_type"),
        [
            (F.horizontal_flip_image, torch.Tensor),
            (F._horizontal_flip_image_pil, PIL.Image.Image),
            (F.horizontal_flip_image, tv_tensors.Image),
            (F.horizontal_flip_bounding_boxes, tv_tensors.BoundingBoxes),
            (F.horizontal_flip_mask, tv_tensors.Mask),
            (F.horizontal_flip_video, tv_tensors.Video),
        ],
    )
    def test_functional_signature(self, kernel, input_type):
        check_functional_kernel_signature_match(F.horizontal_flip, kernel=kernel, input_type=input_type)

    @pytest.mark.parametrize(
        "make_input",
        [make_image_tensor, make_image_pil, make_image, make_bounding_boxes, make_segmentation_mask, make_video],
    )
    @pytest.mark.parametrize("device", cpu_and_cuda())
    def test_transform(self, make_input, device):
        check_transform(transforms.RandomHorizontalFlip(p=1), make_input(device=device))

    @pytest.mark.parametrize(
        "fn", [F.horizontal_flip, transform_cls_to_functional(transforms.RandomHorizontalFlip, p=1)]
    )
    def test_image_correctness(self, fn):
        image = make_image(dtype=torch.uint8, device="cpu")

        actual = fn(image)
        expected = F.to_image(F.horizontal_flip(F.to_pil_image(image)))

        torch.testing.assert_close(actual, expected)

    def _reference_horizontal_flip_bounding_boxes(self, bounding_boxes):
        affine_matrix = np.array(
            [
                [-1, 0, bounding_boxes.canvas_size[1]],
                [0, 1, 0],
            ],
        )

        return reference_affine_bounding_boxes_helper(bounding_boxes, affine_matrix=affine_matrix)

    @pytest.mark.parametrize("format", list(tv_tensors.BoundingBoxFormat))
    @pytest.mark.parametrize(
        "fn", [F.horizontal_flip, transform_cls_to_functional(transforms.RandomHorizontalFlip, p=1)]
    )
    def test_bounding_boxes_correctness(self, format, fn):
        bounding_boxes = make_bounding_boxes(format=format)

        actual = fn(bounding_boxes)
        expected = self._reference_horizontal_flip_bounding_boxes(bounding_boxes)

        torch.testing.assert_close(actual, expected)

    @pytest.mark.parametrize(
        "make_input",
        [make_image_tensor, make_image_pil, make_image, make_bounding_boxes, make_segmentation_mask, make_video],
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
    def test_kernel_image(self, param, value, dtype, device):
        if param == "fill":
            value = adapt_fill(value, dtype=dtype)
        self._check_kernel(
            F.affine_image,
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
    @pytest.mark.parametrize("format", list(tv_tensors.BoundingBoxFormat))
    @pytest.mark.parametrize("dtype", [torch.float32, torch.int64])
    @pytest.mark.parametrize("device", cpu_and_cuda())
    def test_kernel_bounding_boxes(self, param, value, format, dtype, device):
        bounding_boxes = make_bounding_boxes(format=format, dtype=dtype, device=device)
        self._check_kernel(
            F.affine_bounding_boxes,
            bounding_boxes,
            format=format,
            canvas_size=bounding_boxes.canvas_size,
            **{param: value},
            check_scripted_vs_eager=not (param == "shear" and isinstance(value, (int, float))),
        )

    @pytest.mark.parametrize("make_mask", [make_segmentation_mask, make_detection_masks])
    def test_kernel_mask(self, make_mask):
        self._check_kernel(F.affine_mask, make_mask())

    def test_kernel_video(self):
        self._check_kernel(F.affine_video, make_video())

    @pytest.mark.parametrize(
        "make_input",
        [make_image_tensor, make_image_pil, make_image, make_bounding_boxes, make_segmentation_mask, make_video],
    )
    def test_functional(self, make_input):
        check_functional(F.affine, make_input(), **self._MINIMAL_AFFINE_KWARGS)

    @pytest.mark.parametrize(
        ("kernel", "input_type"),
        [
            (F.affine_image, torch.Tensor),
            (F._affine_image_pil, PIL.Image.Image),
            (F.affine_image, tv_tensors.Image),
            (F.affine_bounding_boxes, tv_tensors.BoundingBoxes),
            (F.affine_mask, tv_tensors.Mask),
            (F.affine_video, tv_tensors.Video),
        ],
    )
    def test_functional_signature(self, kernel, input_type):
        check_functional_kernel_signature_match(F.affine, kernel=kernel, input_type=input_type)

    @pytest.mark.parametrize(
        "make_input",
        [make_image_tensor, make_image_pil, make_image, make_bounding_boxes, make_segmentation_mask, make_video],
    )
    @pytest.mark.parametrize("device", cpu_and_cuda())
    def test_transform(self, make_input, device):
        input = make_input(device=device)

        check_transform(transforms.RandomAffine(**self._CORRECTNESS_TRANSFORM_AFFINE_RANGES), input)

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
        expected = F.to_image(
            F.affine(
                F.to_pil_image(image),
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
        expected = F.to_image(transform(F.to_pil_image(image)))

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
        return true_matrix[:2, :]

    def _reference_affine_bounding_boxes(self, bounding_boxes, *, angle, translate, scale, shear, center):
        if center is None:
            center = [s * 0.5 for s in bounding_boxes.canvas_size[::-1]]

        return reference_affine_bounding_boxes_helper(
            bounding_boxes,
            affine_matrix=self._compute_affine_matrix(
                angle=angle, translate=translate, scale=scale, shear=shear, center=center
            ),
        )

    @pytest.mark.parametrize("format", list(tv_tensors.BoundingBoxFormat))
    @pytest.mark.parametrize("angle", _CORRECTNESS_AFFINE_KWARGS["angle"])
    @pytest.mark.parametrize("translate", _CORRECTNESS_AFFINE_KWARGS["translate"])
    @pytest.mark.parametrize("scale", _CORRECTNESS_AFFINE_KWARGS["scale"])
    @pytest.mark.parametrize("shear", _CORRECTNESS_AFFINE_KWARGS["shear"])
    @pytest.mark.parametrize("center", _CORRECTNESS_AFFINE_KWARGS["center"])
    def test_functional_bounding_boxes_correctness(self, format, angle, translate, scale, shear, center):
        bounding_boxes = make_bounding_boxes(format=format)

        actual = F.affine(
            bounding_boxes,
            angle=angle,
            translate=translate,
            scale=scale,
            shear=shear,
            center=center,
        )
        expected = self._reference_affine_bounding_boxes(
            bounding_boxes,
            angle=angle,
            translate=translate,
            scale=scale,
            shear=shear,
            center=center,
        )

        torch.testing.assert_close(actual, expected)

    @pytest.mark.parametrize("format", list(tv_tensors.BoundingBoxFormat))
    @pytest.mark.parametrize("center", _CORRECTNESS_AFFINE_KWARGS["center"])
    @pytest.mark.parametrize("seed", list(range(5)))
    def test_transform_bounding_boxes_correctness(self, format, center, seed):
        bounding_boxes = make_bounding_boxes(format=format)

        transform = transforms.RandomAffine(**self._CORRECTNESS_TRANSFORM_AFFINE_RANGES, center=center)

        torch.manual_seed(seed)
        params = transform._get_params([bounding_boxes])

        torch.manual_seed(seed)
        actual = transform(bounding_boxes)

        expected = self._reference_affine_bounding_boxes(bounding_boxes, **params, center=center)

        torch.testing.assert_close(actual, expected)

    @pytest.mark.parametrize("degrees", _EXHAUSTIVE_TYPE_TRANSFORM_AFFINE_RANGES["degrees"])
    @pytest.mark.parametrize("translate", _EXHAUSTIVE_TYPE_TRANSFORM_AFFINE_RANGES["translate"])
    @pytest.mark.parametrize("scale", _EXHAUSTIVE_TYPE_TRANSFORM_AFFINE_RANGES["scale"])
    @pytest.mark.parametrize("shear", _EXHAUSTIVE_TYPE_TRANSFORM_AFFINE_RANGES["shear"])
    @pytest.mark.parametrize("seed", list(range(10)))
    def test_transform_get_params_bounds(self, degrees, translate, scale, shear, seed):
        image = make_image()
        height, width = F.get_size(image)

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
    def test_kernel_image(self, dtype, device):
        check_kernel(F.vertical_flip_image, make_image(dtype=dtype, device=device))

    @pytest.mark.parametrize("format", list(tv_tensors.BoundingBoxFormat))
    @pytest.mark.parametrize("dtype", [torch.float32, torch.int64])
    @pytest.mark.parametrize("device", cpu_and_cuda())
    def test_kernel_bounding_boxes(self, format, dtype, device):
        bounding_boxes = make_bounding_boxes(format=format, dtype=dtype, device=device)
        check_kernel(
            F.vertical_flip_bounding_boxes,
            bounding_boxes,
            format=format,
            canvas_size=bounding_boxes.canvas_size,
        )

    @pytest.mark.parametrize("make_mask", [make_segmentation_mask, make_detection_masks])
    def test_kernel_mask(self, make_mask):
        check_kernel(F.vertical_flip_mask, make_mask())

    def test_kernel_video(self):
        check_kernel(F.vertical_flip_video, make_video())

    @pytest.mark.parametrize(
        "make_input",
        [make_image_tensor, make_image_pil, make_image, make_bounding_boxes, make_segmentation_mask, make_video],
    )
    def test_functional(self, make_input):
        check_functional(F.vertical_flip, make_input())

    @pytest.mark.parametrize(
        ("kernel", "input_type"),
        [
            (F.vertical_flip_image, torch.Tensor),
            (F._vertical_flip_image_pil, PIL.Image.Image),
            (F.vertical_flip_image, tv_tensors.Image),
            (F.vertical_flip_bounding_boxes, tv_tensors.BoundingBoxes),
            (F.vertical_flip_mask, tv_tensors.Mask),
            (F.vertical_flip_video, tv_tensors.Video),
        ],
    )
    def test_functional_signature(self, kernel, input_type):
        check_functional_kernel_signature_match(F.vertical_flip, kernel=kernel, input_type=input_type)

    @pytest.mark.parametrize(
        "make_input",
        [make_image_tensor, make_image_pil, make_image, make_bounding_boxes, make_segmentation_mask, make_video],
    )
    @pytest.mark.parametrize("device", cpu_and_cuda())
    def test_transform(self, make_input, device):
        check_transform(transforms.RandomVerticalFlip(p=1), make_input(device=device))

    @pytest.mark.parametrize("fn", [F.vertical_flip, transform_cls_to_functional(transforms.RandomVerticalFlip, p=1)])
    def test_image_correctness(self, fn):
        image = make_image(dtype=torch.uint8, device="cpu")

        actual = fn(image)
        expected = F.to_image(F.vertical_flip(F.to_pil_image(image)))

        torch.testing.assert_close(actual, expected)

    def _reference_vertical_flip_bounding_boxes(self, bounding_boxes):
        affine_matrix = np.array(
            [
                [1, 0, 0],
                [0, -1, bounding_boxes.canvas_size[0]],
            ],
        )

        return reference_affine_bounding_boxes_helper(bounding_boxes, affine_matrix=affine_matrix)

    @pytest.mark.parametrize("format", list(tv_tensors.BoundingBoxFormat))
    @pytest.mark.parametrize("fn", [F.vertical_flip, transform_cls_to_functional(transforms.RandomVerticalFlip, p=1)])
    def test_bounding_boxes_correctness(self, format, fn):
        bounding_boxes = make_bounding_boxes(format=format)

        actual = fn(bounding_boxes)
        expected = self._reference_vertical_flip_bounding_boxes(bounding_boxes)

        torch.testing.assert_close(actual, expected)

    @pytest.mark.parametrize(
        "make_input",
        [make_image_tensor, make_image_pil, make_image, make_bounding_boxes, make_segmentation_mask, make_video],
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
    def test_kernel_image(self, param, value, dtype, device):
        kwargs = {param: value}
        if param != "angle":
            kwargs["angle"] = self._MINIMAL_AFFINE_KWARGS["angle"]
        check_kernel(
            F.rotate_image,
            make_image(dtype=dtype, device=device),
            **kwargs,
            check_scripted_vs_eager=not (param == "fill" and isinstance(value, (int, float))),
        )

    @param_value_parametrization(
        angle=_EXHAUSTIVE_TYPE_AFFINE_KWARGS["angle"],
        expand=[False, True],
        center=_EXHAUSTIVE_TYPE_AFFINE_KWARGS["center"],
    )
    @pytest.mark.parametrize("format", list(tv_tensors.BoundingBoxFormat))
    @pytest.mark.parametrize("dtype", [torch.float32, torch.uint8])
    @pytest.mark.parametrize("device", cpu_and_cuda())
    def test_kernel_bounding_boxes(self, param, value, format, dtype, device):
        kwargs = {param: value}
        if param != "angle":
            kwargs["angle"] = self._MINIMAL_AFFINE_KWARGS["angle"]

        bounding_boxes = make_bounding_boxes(format=format, dtype=dtype, device=device)

        check_kernel(
            F.rotate_bounding_boxes,
            bounding_boxes,
            format=format,
            canvas_size=bounding_boxes.canvas_size,
            **kwargs,
        )

    @pytest.mark.parametrize("make_mask", [make_segmentation_mask, make_detection_masks])
    def test_kernel_mask(self, make_mask):
        check_kernel(F.rotate_mask, make_mask(), **self._MINIMAL_AFFINE_KWARGS)

    def test_kernel_video(self):
        check_kernel(F.rotate_video, make_video(), **self._MINIMAL_AFFINE_KWARGS)

    @pytest.mark.parametrize(
        "make_input",
        [make_image_tensor, make_image_pil, make_image, make_bounding_boxes, make_segmentation_mask, make_video],
    )
    def test_functional(self, make_input):
        check_functional(F.rotate, make_input(), **self._MINIMAL_AFFINE_KWARGS)

    @pytest.mark.parametrize(
        ("kernel", "input_type"),
        [
            (F.rotate_image, torch.Tensor),
            (F._rotate_image_pil, PIL.Image.Image),
            (F.rotate_image, tv_tensors.Image),
            (F.rotate_bounding_boxes, tv_tensors.BoundingBoxes),
            (F.rotate_mask, tv_tensors.Mask),
            (F.rotate_video, tv_tensors.Video),
        ],
    )
    def test_functional_signature(self, kernel, input_type):
        check_functional_kernel_signature_match(F.rotate, kernel=kernel, input_type=input_type)

    @pytest.mark.parametrize(
        "make_input",
        [make_image_tensor, make_image_pil, make_image, make_bounding_boxes, make_segmentation_mask, make_video],
    )
    @pytest.mark.parametrize("device", cpu_and_cuda())
    def test_transform(self, make_input, device):
        check_transform(
            transforms.RandomRotation(**self._CORRECTNESS_TRANSFORM_AFFINE_RANGES), make_input(device=device)
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
        expected = F.to_image(
            F.rotate(
                F.to_pil_image(image), angle=angle, center=center, interpolation=interpolation, expand=expand, fill=fill
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
        expected = F.to_image(transform(F.to_pil_image(image)))

        mae = (actual.float() - expected.float()).abs().mean()
        assert mae < 1 if interpolation is transforms.InterpolationMode.NEAREST else 6

    def _compute_output_canvas_size(self, *, expand, canvas_size, affine_matrix):
        if not expand:
            return canvas_size, (0.0, 0.0)

        input_height, input_width = canvas_size

        input_image_frame = np.array(
            [
                [0.0, 0.0, 1.0],
                [0.0, input_height, 1.0],
                [input_width, input_height, 1.0],
                [input_width, 0.0, 1.0],
            ],
            dtype=np.float64,
        )
        output_image_frame = np.matmul(input_image_frame, affine_matrix.astype(input_image_frame.dtype).T)

        recenter_x = float(np.min(output_image_frame[:, 0]))
        recenter_y = float(np.min(output_image_frame[:, 1]))

        output_width = int(np.max(output_image_frame[:, 0]) - recenter_x)
        output_height = int(np.max(output_image_frame[:, 1]) - recenter_y)

        return (output_height, output_width), (recenter_x, recenter_y)

    def _recenter_bounding_boxes_after_expand(self, bounding_boxes, *, recenter_xy):
        x, y = recenter_xy
        if bounding_boxes.format is tv_tensors.BoundingBoxFormat.XYXY:
            translate = [x, y, x, y]
        else:
            translate = [x, y, 0.0, 0.0]
        return tv_tensors.wrap(
            (bounding_boxes.to(torch.float64) - torch.tensor(translate)).to(bounding_boxes.dtype), like=bounding_boxes
        )

    def _reference_rotate_bounding_boxes(self, bounding_boxes, *, angle, expand, center):
        if center is None:
            center = [s * 0.5 for s in bounding_boxes.canvas_size[::-1]]
        cx, cy = center

        a = np.cos(angle * np.pi / 180.0)
        b = np.sin(angle * np.pi / 180.0)
        affine_matrix = np.array(
            [
                [a, b, cx - cx * a - b * cy],
                [-b, a, cy + cx * b - a * cy],
            ],
        )

        new_canvas_size, recenter_xy = self._compute_output_canvas_size(
            expand=expand, canvas_size=bounding_boxes.canvas_size, affine_matrix=affine_matrix
        )

        output = reference_affine_bounding_boxes_helper(
            bounding_boxes,
            affine_matrix=affine_matrix,
            new_canvas_size=new_canvas_size,
            clamp=False,
        )

        return F.clamp_bounding_boxes(self._recenter_bounding_boxes_after_expand(output, recenter_xy=recenter_xy)).to(
            bounding_boxes
        )

    @pytest.mark.parametrize("format", list(tv_tensors.BoundingBoxFormat))
    @pytest.mark.parametrize("angle", _CORRECTNESS_AFFINE_KWARGS["angle"])
    @pytest.mark.parametrize("expand", [False, True])
    @pytest.mark.parametrize("center", _CORRECTNESS_AFFINE_KWARGS["center"])
    def test_functional_bounding_boxes_correctness(self, format, angle, expand, center):
        bounding_boxes = make_bounding_boxes(format=format)

        actual = F.rotate(bounding_boxes, angle=angle, expand=expand, center=center)
        expected = self._reference_rotate_bounding_boxes(bounding_boxes, angle=angle, expand=expand, center=center)

        torch.testing.assert_close(actual, expected)
        torch.testing.assert_close(F.get_size(actual), F.get_size(expected), atol=2 if expand else 0, rtol=0)

    @pytest.mark.parametrize("format", list(tv_tensors.BoundingBoxFormat))
    @pytest.mark.parametrize("expand", [False, True])
    @pytest.mark.parametrize("center", _CORRECTNESS_AFFINE_KWARGS["center"])
    @pytest.mark.parametrize("seed", list(range(5)))
    def test_transform_bounding_boxes_correctness(self, format, expand, center, seed):
        bounding_boxes = make_bounding_boxes(format=format)

        transform = transforms.RandomRotation(**self._CORRECTNESS_TRANSFORM_AFFINE_RANGES, expand=expand, center=center)

        torch.manual_seed(seed)
        params = transform._get_params([bounding_boxes])

        torch.manual_seed(seed)
        actual = transform(bounding_boxes)

        expected = self._reference_rotate_bounding_boxes(bounding_boxes, **params, expand=expand, center=center)

        torch.testing.assert_close(actual, expected)
        torch.testing.assert_close(F.get_size(actual), F.get_size(expected), atol=2 if expand else 0, rtol=0)

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


class TestContainerTransforms:
    class BuiltinTransform(transforms.Transform):
        def _transform(self, inpt, params):
            return inpt

    class PackedInputTransform(nn.Module):
        def forward(self, sample):
            assert len(sample) == 2
            return sample

    class UnpackedInputTransform(nn.Module):
        def forward(self, image, label):
            return image, label

    @pytest.mark.parametrize(
        "transform_cls", [transforms.Compose, functools.partial(transforms.RandomApply, p=1), transforms.RandomOrder]
    )
    @pytest.mark.parametrize(
        "wrapped_transform_clss",
        [
            [BuiltinTransform],
            [PackedInputTransform],
            [UnpackedInputTransform],
            [BuiltinTransform, BuiltinTransform],
            [PackedInputTransform, PackedInputTransform],
            [UnpackedInputTransform, UnpackedInputTransform],
            [BuiltinTransform, PackedInputTransform, BuiltinTransform],
            [BuiltinTransform, UnpackedInputTransform, BuiltinTransform],
            [PackedInputTransform, BuiltinTransform, PackedInputTransform],
            [UnpackedInputTransform, BuiltinTransform, UnpackedInputTransform],
        ],
    )
    @pytest.mark.parametrize("unpack", [True, False])
    def test_packed_unpacked(self, transform_cls, wrapped_transform_clss, unpack):
        needs_packed_inputs = any(issubclass(cls, self.PackedInputTransform) for cls in wrapped_transform_clss)
        needs_unpacked_inputs = any(issubclass(cls, self.UnpackedInputTransform) for cls in wrapped_transform_clss)
        assert not (needs_packed_inputs and needs_unpacked_inputs)

        transform = transform_cls([cls() for cls in wrapped_transform_clss])

        image = make_image()
        label = 3
        packed_input = (image, label)

        def call_transform():
            if unpack:
                return transform(*packed_input)
            else:
                return transform(packed_input)

        if needs_unpacked_inputs and not unpack:
            with pytest.raises(TypeError, match="missing 1 required positional argument"):
                call_transform()
        elif needs_packed_inputs and unpack:
            with pytest.raises(TypeError, match="takes 2 positional arguments but 3 were given"):
                call_transform()
        else:
            output = call_transform()

            assert isinstance(output, tuple) and len(output) == 2
            assert output[0] is image
            assert output[1] is label

    def test_compose(self):
        transform = transforms.Compose(
            [
                transforms.RandomHorizontalFlip(p=1),
                transforms.RandomVerticalFlip(p=1),
            ]
        )

        input = make_image()

        actual = check_transform(transform, input)
        expected = F.vertical_flip(F.horizontal_flip(input))

        assert_equal(actual, expected)

    @pytest.mark.parametrize("p", [0.0, 1.0])
    @pytest.mark.parametrize("sequence_type", [list, nn.ModuleList])
    def test_random_apply(self, p, sequence_type):
        transform = transforms.RandomApply(
            sequence_type(
                [
                    transforms.RandomHorizontalFlip(p=1),
                    transforms.RandomVerticalFlip(p=1),
                ]
            ),
            p=p,
        )

        # This needs to be a pure tensor (or a PIL image), because otherwise check_transforms skips the v1 compatibility
        # check
        input = make_image_tensor()
        output = check_transform(transform, input, check_v1_compatibility=issubclass(sequence_type, nn.ModuleList))

        if p == 1:
            assert_equal(output, F.vertical_flip(F.horizontal_flip(input)))
        else:
            assert output is input

    @pytest.mark.parametrize("p", [(0, 1), (1, 0)])
    def test_random_choice(self, p):
        transform = transforms.RandomChoice(
            [
                transforms.RandomHorizontalFlip(p=1),
                transforms.RandomVerticalFlip(p=1),
            ],
            p=p,
        )

        input = make_image()
        output = check_transform(transform, input)

        p_horz, p_vert = p
        if p_horz:
            assert_equal(output, F.horizontal_flip(input))
        else:
            assert_equal(output, F.vertical_flip(input))

    def test_random_order(self):
        transform = transforms.Compose(
            [
                transforms.RandomHorizontalFlip(p=1),
                transforms.RandomVerticalFlip(p=1),
            ]
        )

        input = make_image()

        actual = check_transform(transform, input)
        # We can't really check whether the transforms are actually applied in random order. However, horizontal and
        # vertical flip are commutative. Meaning, even under the assumption that the transform applies them in random
        # order, we can use a fixed order to compute the expected value.
        expected = F.vertical_flip(F.horizontal_flip(input))

        assert_equal(actual, expected)

    def test_errors(self):
        for cls in [transforms.Compose, transforms.RandomChoice, transforms.RandomOrder]:
            with pytest.raises(TypeError, match="Argument transforms should be a sequence of callables"):
                cls(lambda x: x)

        with pytest.raises(ValueError, match="at least one transform"):
            transforms.Compose([])

        for p in [-1, 2]:
            with pytest.raises(ValueError, match=re.escape("value in the interval [0.0, 1.0]")):
                transforms.RandomApply([lambda x: x], p=p)

        for transforms_, p in [([lambda x: x], []), ([], [1.0])]:
            with pytest.raises(ValueError, match="Length of p doesn't match the number of transforms"):
                transforms.RandomChoice(transforms_, p=p)


class TestToDtype:
    @pytest.mark.parametrize(
        ("kernel", "make_input"),
        [
            (F.to_dtype_image, make_image_tensor),
            (F.to_dtype_image, make_image),
            (F.to_dtype_video, make_video),
        ],
    )
    @pytest.mark.parametrize("input_dtype", [torch.float32, torch.float64, torch.uint8])
    @pytest.mark.parametrize("output_dtype", [torch.float32, torch.float64, torch.uint8])
    @pytest.mark.parametrize("device", cpu_and_cuda())
    @pytest.mark.parametrize("scale", (True, False))
    def test_kernel(self, kernel, make_input, input_dtype, output_dtype, device, scale):
        check_kernel(
            kernel,
            make_input(dtype=input_dtype, device=device),
            dtype=output_dtype,
            scale=scale,
        )

    @pytest.mark.parametrize("make_input", [make_image_tensor, make_image, make_video])
    @pytest.mark.parametrize("input_dtype", [torch.float32, torch.float64, torch.uint8])
    @pytest.mark.parametrize("output_dtype", [torch.float32, torch.float64, torch.uint8])
    @pytest.mark.parametrize("device", cpu_and_cuda())
    @pytest.mark.parametrize("scale", (True, False))
    def test_functional(self, make_input, input_dtype, output_dtype, device, scale):
        check_functional(
            F.to_dtype,
            make_input(dtype=input_dtype, device=device),
            dtype=output_dtype,
            scale=scale,
        )

    @pytest.mark.parametrize(
        "make_input",
        [make_image_tensor, make_image, make_bounding_boxes, make_segmentation_mask, make_video],
    )
    @pytest.mark.parametrize("input_dtype", [torch.float32, torch.float64, torch.uint8])
    @pytest.mark.parametrize("output_dtype", [torch.float32, torch.float64, torch.uint8])
    @pytest.mark.parametrize("device", cpu_and_cuda())
    @pytest.mark.parametrize("scale", (True, False))
    @pytest.mark.parametrize("as_dict", (True, False))
    def test_transform(self, make_input, input_dtype, output_dtype, device, scale, as_dict):
        input = make_input(dtype=input_dtype, device=device)
        if as_dict:
            output_dtype = {type(input): output_dtype}
        check_transform(transforms.ToDtype(dtype=output_dtype, scale=scale), input, check_sample_input=not as_dict)

    def reference_convert_dtype_image_tensor(self, image, dtype=torch.float, scale=False):
        input_dtype = image.dtype
        output_dtype = dtype

        if not scale:
            return image.to(dtype)

        if output_dtype == input_dtype:
            return image

        def fn(value):
            if input_dtype.is_floating_point:
                if output_dtype.is_floating_point:
                    return value
                else:
                    return round(decimal.Decimal(value) * torch.iinfo(output_dtype).max)
            else:
                input_max_value = torch.iinfo(input_dtype).max

                if output_dtype.is_floating_point:
                    return float(decimal.Decimal(value) / input_max_value)
                else:
                    output_max_value = torch.iinfo(output_dtype).max

                    if input_max_value > output_max_value:
                        factor = (input_max_value + 1) // (output_max_value + 1)
                        return value / factor
                    else:
                        factor = (output_max_value + 1) // (input_max_value + 1)
                        return value * factor

        return torch.tensor(tree_map(fn, image.tolist()), dtype=dtype, device=image.device)

    @pytest.mark.parametrize("input_dtype", [torch.float32, torch.float64, torch.uint8])
    @pytest.mark.parametrize("output_dtype", [torch.float32, torch.float64, torch.uint8])
    @pytest.mark.parametrize("device", cpu_and_cuda())
    @pytest.mark.parametrize("scale", (True, False))
    def test_image_correctness(self, input_dtype, output_dtype, device, scale):
        if input_dtype.is_floating_point and output_dtype == torch.int64:
            pytest.xfail("float to int64 conversion is not supported")

        input = make_image(dtype=input_dtype, device=device)

        out = F.to_dtype(input, dtype=output_dtype, scale=scale)
        expected = self.reference_convert_dtype_image_tensor(input, dtype=output_dtype, scale=scale)

        if input_dtype.is_floating_point and not output_dtype.is_floating_point and scale:
            torch.testing.assert_close(out, expected, atol=1, rtol=0)
        else:
            torch.testing.assert_close(out, expected)

    def was_scaled(self, inpt):
        # this assumes the target dtype is float
        return inpt.max() <= 1

    def make_inpt_with_bbox_and_mask(self, make_input):
        H, W = 10, 10
        inpt_dtype = torch.uint8
        bbox_dtype = torch.float32
        mask_dtype = torch.bool
        sample = {
            "inpt": make_input(size=(H, W), dtype=inpt_dtype),
            "bbox": make_bounding_boxes(canvas_size=(H, W), dtype=bbox_dtype),
            "mask": make_detection_masks(size=(H, W), dtype=mask_dtype),
        }

        return sample, inpt_dtype, bbox_dtype, mask_dtype

    @pytest.mark.parametrize("make_input", (make_image_tensor, make_image, make_video))
    @pytest.mark.parametrize("scale", (True, False))
    def test_dtype_not_a_dict(self, make_input, scale):
        # assert only inpt gets transformed when dtype isn't a dict

        sample, inpt_dtype, bbox_dtype, mask_dtype = self.make_inpt_with_bbox_and_mask(make_input)
        out = transforms.ToDtype(dtype=torch.float32, scale=scale)(sample)

        assert out["inpt"].dtype != inpt_dtype
        assert out["inpt"].dtype == torch.float32
        if scale:
            assert self.was_scaled(out["inpt"])
        else:
            assert not self.was_scaled(out["inpt"])
        assert out["bbox"].dtype == bbox_dtype
        assert out["mask"].dtype == mask_dtype

    @pytest.mark.parametrize("make_input", (make_image_tensor, make_image, make_video))
    def test_others_catch_all_and_none(self, make_input):
        # make sure "others" works as a catch-all and that None means no conversion

        sample, inpt_dtype, bbox_dtype, mask_dtype = self.make_inpt_with_bbox_and_mask(make_input)
        out = transforms.ToDtype(dtype={tv_tensors.Mask: torch.int64, "others": None})(sample)
        assert out["inpt"].dtype == inpt_dtype
        assert out["bbox"].dtype == bbox_dtype
        assert out["mask"].dtype != mask_dtype
        assert out["mask"].dtype == torch.int64

    @pytest.mark.parametrize("make_input", (make_image_tensor, make_image, make_video))
    def test_typical_use_case(self, make_input):
        # Typical use-case: want to convert dtype and scale for inpt and just dtype for masks.
        # This just makes sure we now have a decent API for this

        sample, inpt_dtype, bbox_dtype, mask_dtype = self.make_inpt_with_bbox_and_mask(make_input)
        out = transforms.ToDtype(
            dtype={type(sample["inpt"]): torch.float32, tv_tensors.Mask: torch.int64, "others": None}, scale=True
        )(sample)
        assert out["inpt"].dtype != inpt_dtype
        assert out["inpt"].dtype == torch.float32
        assert self.was_scaled(out["inpt"])
        assert out["bbox"].dtype == bbox_dtype
        assert out["mask"].dtype != mask_dtype
        assert out["mask"].dtype == torch.int64

    @pytest.mark.parametrize("make_input", (make_image_tensor, make_image, make_video))
    def test_errors_warnings(self, make_input):
        sample, inpt_dtype, bbox_dtype, mask_dtype = self.make_inpt_with_bbox_and_mask(make_input)

        with pytest.raises(ValueError, match="No dtype was specified for"):
            out = transforms.ToDtype(dtype={tv_tensors.Mask: torch.float32})(sample)
        with pytest.warns(UserWarning, match=re.escape("plain `torch.Tensor` will *not* be transformed")):
            transforms.ToDtype(dtype={torch.Tensor: torch.float32, tv_tensors.Image: torch.float32})
        with pytest.warns(UserWarning, match="no scaling will be done"):
            out = transforms.ToDtype(dtype={"others": None}, scale=True)(sample)
        assert out["inpt"].dtype == inpt_dtype
        assert out["bbox"].dtype == bbox_dtype
        assert out["mask"].dtype == mask_dtype


class TestAdjustBrightness:
    _CORRECTNESS_BRIGHTNESS_FACTORS = [0.5, 0.0, 1.0, 5.0]
    _DEFAULT_BRIGHTNESS_FACTOR = _CORRECTNESS_BRIGHTNESS_FACTORS[0]

    @pytest.mark.parametrize(
        ("kernel", "make_input"),
        [
            (F.adjust_brightness_image, make_image),
            (F.adjust_brightness_video, make_video),
        ],
    )
    @pytest.mark.parametrize("dtype", [torch.float32, torch.uint8])
    @pytest.mark.parametrize("device", cpu_and_cuda())
    def test_kernel(self, kernel, make_input, dtype, device):
        check_kernel(kernel, make_input(dtype=dtype, device=device), brightness_factor=self._DEFAULT_BRIGHTNESS_FACTOR)

    @pytest.mark.parametrize("make_input", [make_image_tensor, make_image_pil, make_image, make_video])
    def test_functional(self, make_input):
        check_functional(F.adjust_brightness, make_input(), brightness_factor=self._DEFAULT_BRIGHTNESS_FACTOR)

    @pytest.mark.parametrize(
        ("kernel", "input_type"),
        [
            (F.adjust_brightness_image, torch.Tensor),
            (F._adjust_brightness_image_pil, PIL.Image.Image),
            (F.adjust_brightness_image, tv_tensors.Image),
            (F.adjust_brightness_video, tv_tensors.Video),
        ],
    )
    def test_functional_signature(self, kernel, input_type):
        check_functional_kernel_signature_match(F.adjust_brightness, kernel=kernel, input_type=input_type)

    @pytest.mark.parametrize("brightness_factor", _CORRECTNESS_BRIGHTNESS_FACTORS)
    def test_image_correctness(self, brightness_factor):
        image = make_image(dtype=torch.uint8, device="cpu")

        actual = F.adjust_brightness(image, brightness_factor=brightness_factor)
        expected = F.to_image(F.adjust_brightness(F.to_pil_image(image), brightness_factor=brightness_factor))

        torch.testing.assert_close(actual, expected)


class TestCutMixMixUp:
    class DummyDataset:
        def __init__(self, size, num_classes):
            self.size = size
            self.num_classes = num_classes
            assert size < num_classes

        def __getitem__(self, idx):
            img = torch.rand(3, 100, 100)
            label = idx  # This ensures all labels in a batch are unique and makes testing easier
            return img, label

        def __len__(self):
            return self.size

    @pytest.mark.parametrize("T", [transforms.CutMix, transforms.MixUp])
    def test_supported_input_structure(self, T):

        batch_size = 32
        num_classes = 100

        dataset = self.DummyDataset(size=batch_size, num_classes=num_classes)

        cutmix_mixup = T(num_classes=num_classes)

        dl = DataLoader(dataset, batch_size=batch_size)

        # Input sanity checks
        img, target = next(iter(dl))
        input_img_size = img.shape[-3:]
        assert isinstance(img, torch.Tensor) and isinstance(target, torch.Tensor)
        assert target.shape == (batch_size,)

        def check_output(img, target):
            assert img.shape == (batch_size, *input_img_size)
            assert target.shape == (batch_size, num_classes)
            torch.testing.assert_close(target.sum(axis=-1), torch.ones(batch_size))
            num_non_zero_labels = (target != 0).sum(axis=-1)
            assert (num_non_zero_labels == 2).all()

        # After Dataloader, as unpacked input
        img, target = next(iter(dl))
        assert target.shape == (batch_size,)
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
    @pytest.mark.parametrize("T", [transforms.CutMix, transforms.MixUp])
    def test_cpu_vs_gpu(self, T):
        num_classes = 10
        batch_size = 3
        H, W = 12, 12

        imgs = torch.rand(batch_size, 3, H, W)
        labels = torch.randint(0, num_classes, (batch_size,))
        cutmix_mixup = T(alpha=0.5, num_classes=num_classes)

        _check_kernel_cuda_vs_cpu(cutmix_mixup, imgs, labels, rtol=None, atol=None)

    @pytest.mark.parametrize("T", [transforms.CutMix, transforms.MixUp])
    def test_error(self, T):

        num_classes = 10
        batch_size = 9

        imgs = torch.rand(batch_size, 3, 12, 12)
        cutmix_mixup = T(alpha=0.5, num_classes=num_classes)

        for input_with_bad_type in (
            F.to_pil_image(imgs[0]),
            tv_tensors.Mask(torch.rand(12, 12)),
            tv_tensors.BoundingBoxes(torch.rand(2, 4), format="XYXY", canvas_size=12),
        ):
            with pytest.raises(ValueError, match="does not support PIL images, "):
                cutmix_mixup(input_with_bad_type)

        with pytest.raises(ValueError, match="Could not infer where the labels are"):
            cutmix_mixup({"img": imgs, "Nothing_else": 3})

        with pytest.raises(ValueError, match="labels tensor should be of shape"):
            # Note: the error message isn't ideal, but that's because the label heuristic found the img as the label
            # It's OK, it's an edge-case. The important thing is that this fails loudly instead of passing silently
            cutmix_mixup(imgs)

        with pytest.raises(ValueError, match="When using the default labels_getter"):
            cutmix_mixup(imgs, "not_a_tensor")

        with pytest.raises(ValueError, match="labels tensor should be of shape"):
            cutmix_mixup(imgs, torch.randint(0, 2, size=(2, 3)))

        with pytest.raises(ValueError, match="Expected a batched input with 4 dims"):
            cutmix_mixup(imgs[None, None], torch.randint(0, num_classes, size=(batch_size,)))

        with pytest.raises(ValueError, match="does not match the batch size of the labels"):
            cutmix_mixup(imgs, torch.randint(0, num_classes, size=(batch_size + 1,)))

        with pytest.raises(ValueError, match="labels tensor should be of shape"):
            # The purpose of this check is more about documenting the current
            # behaviour of what happens on a Compose(), rather than actually
            # asserting the expected behaviour. We may support Compose() in the
            # future, e.g. for 2 consecutive CutMix?
            labels = torch.randint(0, num_classes, size=(batch_size,))
            transforms.Compose([cutmix_mixup, cutmix_mixup])(imgs, labels)


@pytest.mark.parametrize("key", ("labels", "LABELS", "LaBeL", "SOME_WEIRD_KEY_THAT_HAS_LABeL_IN_IT"))
@pytest.mark.parametrize("sample_type", (tuple, list, dict))
def test_labels_getter_default_heuristic(key, sample_type):
    labels = torch.arange(10)
    sample = {key: labels, "another_key": "whatever"}
    if sample_type is not dict:
        sample = sample_type((None, sample, "whatever_again"))
    assert transforms._utils._find_labels_default_heuristic(sample) is labels

    if key.lower() != "labels":
        # If "labels" is in the dict (case-insensitive),
        # it takes precedence over other keys which would otherwise be a match
        d = {key: "something_else", "labels": labels}
        assert transforms._utils._find_labels_default_heuristic(d) is labels


class TestShapeGetters:
    @pytest.mark.parametrize(
        ("kernel", "make_input"),
        [
            (F.get_dimensions_image, make_image_tensor),
            (F._get_dimensions_image_pil, make_image_pil),
            (F.get_dimensions_image, make_image),
            (F.get_dimensions_video, make_video),
        ],
    )
    def test_get_dimensions(self, kernel, make_input):
        size = (10, 10)
        color_space, num_channels = "RGB", 3

        input = make_input(size, color_space=color_space)

        assert kernel(input) == F.get_dimensions(input) == [num_channels, *size]

    @pytest.mark.parametrize(
        ("kernel", "make_input"),
        [
            (F.get_num_channels_image, make_image_tensor),
            (F._get_num_channels_image_pil, make_image_pil),
            (F.get_num_channels_image, make_image),
            (F.get_num_channels_video, make_video),
        ],
    )
    def test_get_num_channels(self, kernel, make_input):
        color_space, num_channels = "RGB", 3

        input = make_input(color_space=color_space)

        assert kernel(input) == F.get_num_channels(input) == num_channels

    @pytest.mark.parametrize(
        ("kernel", "make_input"),
        [
            (F.get_size_image, make_image_tensor),
            (F._get_size_image_pil, make_image_pil),
            (F.get_size_image, make_image),
            (F.get_size_bounding_boxes, make_bounding_boxes),
            (F.get_size_mask, make_detection_masks),
            (F.get_size_mask, make_segmentation_mask),
            (F.get_size_video, make_video),
        ],
    )
    def test_get_size(self, kernel, make_input):
        size = (10, 10)

        input = make_input(size)

        assert kernel(input) == F.get_size(input) == list(size)

    @pytest.mark.parametrize(
        ("kernel", "make_input"),
        [
            (F.get_num_frames_video, make_video_tensor),
            (F.get_num_frames_video, make_video),
        ],
    )
    def test_get_num_frames(self, kernel, make_input):
        num_frames = 4

        input = make_input(num_frames=num_frames)

        assert kernel(input) == F.get_num_frames(input) == num_frames

    @pytest.mark.parametrize(
        ("functional", "make_input"),
        [
            (F.get_dimensions, make_bounding_boxes),
            (F.get_dimensions, make_detection_masks),
            (F.get_dimensions, make_segmentation_mask),
            (F.get_num_channels, make_bounding_boxes),
            (F.get_num_channels, make_detection_masks),
            (F.get_num_channels, make_segmentation_mask),
            (F.get_num_frames, make_image_pil),
            (F.get_num_frames, make_image),
            (F.get_num_frames, make_bounding_boxes),
            (F.get_num_frames, make_detection_masks),
            (F.get_num_frames, make_segmentation_mask),
        ],
    )
    def test_unsupported_types(self, functional, make_input):
        input = make_input()

        with pytest.raises(TypeError, match=re.escape(str(type(input)))):
            functional(input)


class TestRegisterKernel:
    @pytest.mark.parametrize("functional", (F.resize, "resize"))
    def test_register_kernel(self, functional):
        class CustomTVTensor(tv_tensors.TVTensor):
            pass

        kernel_was_called = False

        @F.register_kernel(functional, CustomTVTensor)
        def new_resize(dp, *args, **kwargs):
            nonlocal kernel_was_called
            kernel_was_called = True
            return dp

        t = transforms.Resize(size=(224, 224), antialias=True)

        my_dp = CustomTVTensor(torch.rand(3, 10, 10))
        out = t(my_dp)
        assert out is my_dp
        assert kernel_was_called

        # Sanity check to make sure we didn't override the kernel of other types
        t(torch.rand(3, 10, 10)).shape == (3, 224, 224)
        t(tv_tensors.Image(torch.rand(3, 10, 10))).shape == (3, 224, 224)

    def test_errors(self):
        with pytest.raises(ValueError, match="Could not find functional with name"):
            F.register_kernel("bad_name", tv_tensors.Image)

        with pytest.raises(ValueError, match="Kernels can only be registered on functionals"):
            F.register_kernel(tv_tensors.Image, F.resize)

        with pytest.raises(ValueError, match="Kernels can only be registered for subclasses"):
            F.register_kernel(F.resize, object)

        with pytest.raises(ValueError, match="cannot be registered for the builtin tv_tensor classes"):
            F.register_kernel(F.resize, tv_tensors.Image)(F.resize_image)

        class CustomTVTensor(tv_tensors.TVTensor):
            pass

        def resize_custom_tv_tensor():
            pass

        F.register_kernel(F.resize, CustomTVTensor)(resize_custom_tv_tensor)

        with pytest.raises(ValueError, match="already has a kernel registered for type"):
            F.register_kernel(F.resize, CustomTVTensor)(resize_custom_tv_tensor)


class TestGetKernel:
    # We are using F.resize as functional and the kernels below as proxy. Any other functional / kernels combination
    # would also be fine
    KERNELS = {
        torch.Tensor: F.resize_image,
        PIL.Image.Image: F._resize_image_pil,
        tv_tensors.Image: F.resize_image,
        tv_tensors.BoundingBoxes: F.resize_bounding_boxes,
        tv_tensors.Mask: F.resize_mask,
        tv_tensors.Video: F.resize_video,
    }

    @pytest.mark.parametrize("input_type", [str, int, object])
    def test_unsupported_types(self, input_type):
        with pytest.raises(TypeError, match="supports inputs of type"):
            _get_kernel(F.resize, input_type)

    def test_exact_match(self):
        # We cannot use F.resize together with self.KERNELS mapping here directly here, since this is only the
        # ideal wrapping. Practically, we have an intermediate wrapper layer. Thus, we create a new resize functional
        # here, register the kernels without wrapper, and check the exact matching afterwards.
        def resize_with_pure_kernels():
            pass

        for input_type, kernel in self.KERNELS.items():
            _register_kernel_internal(resize_with_pure_kernels, input_type, tv_tensor_wrapper=False)(kernel)

            assert _get_kernel(resize_with_pure_kernels, input_type) is kernel

    def test_builtin_tv_tensor_subclass(self):
        # We cannot use F.resize together with self.KERNELS mapping here directly here, since this is only the
        # ideal wrapping. Practically, we have an intermediate wrapper layer. Thus, we create a new resize functional
        # here, register the kernels without wrapper, and check if subclasses of our builtin tv_tensors get dispatched
        # to the kernel of the corresponding superclass
        def resize_with_pure_kernels():
            pass

        class MyImage(tv_tensors.Image):
            pass

        class MyBoundingBoxes(tv_tensors.BoundingBoxes):
            pass

        class MyMask(tv_tensors.Mask):
            pass

        class MyVideo(tv_tensors.Video):
            pass

        for custom_tv_tensor_subclass in [
            MyImage,
            MyBoundingBoxes,
            MyMask,
            MyVideo,
        ]:
            builtin_tv_tensor_class = custom_tv_tensor_subclass.__mro__[1]
            builtin_tv_tensor_kernel = self.KERNELS[builtin_tv_tensor_class]
            _register_kernel_internal(resize_with_pure_kernels, builtin_tv_tensor_class, tv_tensor_wrapper=False)(
                builtin_tv_tensor_kernel
            )

            assert _get_kernel(resize_with_pure_kernels, custom_tv_tensor_subclass) is builtin_tv_tensor_kernel

    def test_tv_tensor_subclass(self):
        class MyTVTensor(tv_tensors.TVTensor):
            pass

        with pytest.raises(TypeError, match="supports inputs of type"):
            _get_kernel(F.resize, MyTVTensor)

        def resize_my_tv_tensor():
            pass

        _register_kernel_internal(F.resize, MyTVTensor, tv_tensor_wrapper=False)(resize_my_tv_tensor)

        assert _get_kernel(F.resize, MyTVTensor) is resize_my_tv_tensor

    def test_pil_image_subclass(self):
        opened_image = PIL.Image.open(Path(__file__).parent / "assets" / "encode_jpeg" / "grace_hopper_517x606.jpg")
        loaded_image = opened_image.convert("RGB")

        # check the assumptions
        assert isinstance(opened_image, PIL.Image.Image)
        assert type(opened_image) is not PIL.Image.Image

        assert type(loaded_image) is PIL.Image.Image

        size = [17, 11]
        for image in [opened_image, loaded_image]:
            kernel = _get_kernel(F.resize, type(image))

            output = kernel(image, size=size)

            assert F.get_size(output) == size


class TestPermuteChannels:
    _DEFAULT_PERMUTATION = [2, 0, 1]

    @pytest.mark.parametrize(
        ("kernel", "make_input"),
        [
            (F.permute_channels_image, make_image_tensor),
            # FIXME
            # check_kernel does not support PIL kernel, but it should
            (F.permute_channels_image, make_image),
            (F.permute_channels_video, make_video),
        ],
    )
    @pytest.mark.parametrize("dtype", [torch.float32, torch.uint8])
    @pytest.mark.parametrize("device", cpu_and_cuda())
    def test_kernel(self, kernel, make_input, dtype, device):
        check_kernel(kernel, make_input(dtype=dtype, device=device), permutation=self._DEFAULT_PERMUTATION)

    @pytest.mark.parametrize("make_input", [make_image_tensor, make_image_pil, make_image, make_video])
    def test_functional(self, make_input):
        check_functional(F.permute_channels, make_input(), permutation=self._DEFAULT_PERMUTATION)

    @pytest.mark.parametrize(
        ("kernel", "input_type"),
        [
            (F.permute_channels_image, torch.Tensor),
            (F._permute_channels_image_pil, PIL.Image.Image),
            (F.permute_channels_image, tv_tensors.Image),
            (F.permute_channels_video, tv_tensors.Video),
        ],
    )
    def test_functional_signature(self, kernel, input_type):
        check_functional_kernel_signature_match(F.permute_channels, kernel=kernel, input_type=input_type)

    def reference_image_correctness(self, image, permutation):
        channel_images = image.split(1, dim=-3)
        permuted_channel_images = [channel_images[channel_idx] for channel_idx in permutation]
        return tv_tensors.Image(torch.concat(permuted_channel_images, dim=-3))

    @pytest.mark.parametrize("permutation", [[2, 0, 1], [1, 2, 0], [2, 0, 1], [0, 1, 2]])
    @pytest.mark.parametrize("batch_dims", [(), (2,), (2, 1)])
    def test_image_correctness(self, permutation, batch_dims):
        image = make_image(batch_dims=batch_dims)

        actual = F.permute_channels(image, permutation=permutation)
        expected = self.reference_image_correctness(image, permutation=permutation)

        torch.testing.assert_close(actual, expected)


class TestElastic:
    def _make_displacement(self, inpt):
        return torch.rand(
            1,
            *F.get_size(inpt),
            2,
            dtype=torch.float32,
            device=inpt.device if isinstance(inpt, torch.Tensor) else "cpu",
        )

    @param_value_parametrization(
        interpolation=[transforms.InterpolationMode.NEAREST, transforms.InterpolationMode.BILINEAR],
        fill=EXHAUSTIVE_TYPE_FILLS,
    )
    @pytest.mark.parametrize("dtype", [torch.float32, torch.uint8, torch.float16])
    @pytest.mark.parametrize("device", cpu_and_cuda())
    def test_kernel_image(self, param, value, dtype, device):
        image = make_image_tensor(dtype=dtype, device=device)

        check_kernel(
            F.elastic_image,
            image,
            displacement=self._make_displacement(image),
            **{param: value},
            check_scripted_vs_eager=not (param == "fill" and isinstance(value, (int, float))),
            check_cuda_vs_cpu=dtype is not torch.float16,
        )

    @pytest.mark.parametrize("format", list(tv_tensors.BoundingBoxFormat))
    @pytest.mark.parametrize("dtype", [torch.float32, torch.int64])
    @pytest.mark.parametrize("device", cpu_and_cuda())
    def test_kernel_bounding_boxes(self, format, dtype, device):
        bounding_boxes = make_bounding_boxes(format=format, dtype=dtype, device=device)

        check_kernel(
            F.elastic_bounding_boxes,
            bounding_boxes,
            format=bounding_boxes.format,
            canvas_size=bounding_boxes.canvas_size,
            displacement=self._make_displacement(bounding_boxes),
        )

    @pytest.mark.parametrize("make_mask", [make_segmentation_mask, make_detection_masks])
    def test_kernel_mask(self, make_mask):
        mask = make_mask()
        check_kernel(F.elastic_mask, mask, displacement=self._make_displacement(mask))

    def test_kernel_video(self):
        video = make_video()
        check_kernel(F.elastic_video, video, displacement=self._make_displacement(video))

    @pytest.mark.parametrize(
        "make_input",
        [make_image_tensor, make_image_pil, make_image, make_bounding_boxes, make_segmentation_mask, make_video],
    )
    def test_functional(self, make_input):
        input = make_input()
        check_functional(F.elastic, input, displacement=self._make_displacement(input))

    @pytest.mark.parametrize(
        ("kernel", "input_type"),
        [
            (F.elastic_image, torch.Tensor),
            (F._elastic_image_pil, PIL.Image.Image),
            (F.elastic_image, tv_tensors.Image),
            (F.elastic_bounding_boxes, tv_tensors.BoundingBoxes),
            (F.elastic_mask, tv_tensors.Mask),
            (F.elastic_video, tv_tensors.Video),
        ],
    )
    def test_functional_signature(self, kernel, input_type):
        check_functional_kernel_signature_match(F.elastic, kernel=kernel, input_type=input_type)

    @pytest.mark.parametrize(
        "make_input",
        [make_image_tensor, make_image_pil, make_image, make_bounding_boxes, make_segmentation_mask, make_video],
    )
    def test_displacement_error(self, make_input):
        input = make_input()

        with pytest.raises(TypeError, match="displacement should be a Tensor"):
            F.elastic(input, displacement=None)

        with pytest.raises(ValueError, match="displacement shape should be"):
            F.elastic(input, displacement=torch.rand(F.get_size(input)))

    @pytest.mark.parametrize(
        "make_input",
        [make_image_tensor, make_image_pil, make_image, make_bounding_boxes, make_segmentation_mask, make_video],
    )
    # ElasticTransform needs larger images to avoid the needed internal padding being larger than the actual image
    @pytest.mark.parametrize("size", [(163, 163), (72, 333), (313, 95)])
    @pytest.mark.parametrize("device", cpu_and_cuda())
    def test_transform(self, make_input, size, device):
        # We have to skip that test on M1 because it's flaky: Mismatched elements: 35 / 89205 (0.0%)
        # See https://github.com/pytorch/vision/issues/8154
        # All other platforms are fine, so the differences do not come from something we own in torchvision
        check_v1_compatibility = False if sys.platform == "darwin" else dict(rtol=0, atol=1)

        check_transform(
            transforms.ElasticTransform(),
            make_input(size, device=device),
            check_v1_compatibility=check_v1_compatibility,
        )


class TestToPureTensor:
    def test_correctness(self):
        input = {
            "img": make_image(),
            "img_tensor": make_image_tensor(),
            "img_pil": make_image_pil(),
            "mask": make_detection_masks(),
            "video": make_video(),
            "bbox": make_bounding_boxes(),
            "str": "str",
        }

        out = transforms.ToPureTensor()(input)

        for input_value, out_value in zip(input.values(), out.values()):
            if isinstance(input_value, tv_tensors.TVTensor):
                assert isinstance(out_value, torch.Tensor) and not isinstance(out_value, tv_tensors.TVTensor)
            else:
                assert isinstance(out_value, type(input_value))


class TestCrop:
    INPUT_SIZE = (21, 11)

    CORRECTNESS_CROP_KWARGS = [
        # center
        dict(top=5, left=5, height=10, width=5),
        # larger than input, i.e. pad
        dict(top=-5, left=-5, height=30, width=20),
        # sides: left, right, top, bottom
        dict(top=-5, left=-5, height=30, width=10),
        dict(top=-5, left=5, height=30, width=10),
        dict(top=-5, left=-5, height=20, width=20),
        dict(top=5, left=-5, height=20, width=20),
        # corners: top-left, top-right, bottom-left, bottom-right
        dict(top=-5, left=-5, height=20, width=10),
        dict(top=-5, left=5, height=20, width=10),
        dict(top=5, left=-5, height=20, width=10),
        dict(top=5, left=5, height=20, width=10),
    ]
    MINIMAL_CROP_KWARGS = CORRECTNESS_CROP_KWARGS[0]

    @pytest.mark.parametrize("kwargs", CORRECTNESS_CROP_KWARGS)
    @pytest.mark.parametrize("dtype", [torch.uint8, torch.float32])
    @pytest.mark.parametrize("device", cpu_and_cuda())
    def test_kernel_image(self, kwargs, dtype, device):
        check_kernel(F.crop_image, make_image(self.INPUT_SIZE, dtype=dtype, device=device), **kwargs)

    @pytest.mark.parametrize("kwargs", CORRECTNESS_CROP_KWARGS)
    @pytest.mark.parametrize("format", list(tv_tensors.BoundingBoxFormat))
    @pytest.mark.parametrize("dtype", [torch.float32, torch.int64])
    @pytest.mark.parametrize("device", cpu_and_cuda())
    def test_kernel_bounding_box(self, kwargs, format, dtype, device):
        bounding_boxes = make_bounding_boxes(self.INPUT_SIZE, format=format, dtype=dtype, device=device)
        check_kernel(F.crop_bounding_boxes, bounding_boxes, format=format, **kwargs)

    @pytest.mark.parametrize("make_mask", [make_segmentation_mask, make_detection_masks])
    def test_kernel_mask(self, make_mask):
        check_kernel(F.crop_mask, make_mask(self.INPUT_SIZE), **self.MINIMAL_CROP_KWARGS)

    def test_kernel_video(self):
        check_kernel(F.crop_video, make_video(self.INPUT_SIZE), **self.MINIMAL_CROP_KWARGS)

    @pytest.mark.parametrize(
        "make_input",
        [make_image_tensor, make_image_pil, make_image, make_bounding_boxes, make_segmentation_mask, make_video],
    )
    def test_functional(self, make_input):
        check_functional(F.crop, make_input(self.INPUT_SIZE), **self.MINIMAL_CROP_KWARGS)

    @pytest.mark.parametrize(
        ("kernel", "input_type"),
        [
            (F.crop_image, torch.Tensor),
            (F._crop_image_pil, PIL.Image.Image),
            (F.crop_image, tv_tensors.Image),
            (F.crop_bounding_boxes, tv_tensors.BoundingBoxes),
            (F.crop_mask, tv_tensors.Mask),
            (F.crop_video, tv_tensors.Video),
        ],
    )
    def test_functional_signature(self, kernel, input_type):
        check_functional_kernel_signature_match(F.crop, kernel=kernel, input_type=input_type)

    @pytest.mark.parametrize("kwargs", CORRECTNESS_CROP_KWARGS)
    def test_functional_image_correctness(self, kwargs):
        image = make_image(self.INPUT_SIZE, dtype=torch.uint8, device="cpu")

        actual = F.crop(image, **kwargs)
        expected = F.to_image(F.crop(F.to_pil_image(image), **kwargs))

        assert_equal(actual, expected)

    @param_value_parametrization(
        size=[(10, 5), (25, 15), (25, 5), (10, 15)],
        fill=EXHAUSTIVE_TYPE_FILLS,
    )
    @pytest.mark.parametrize(
        "make_input",
        [make_image_tensor, make_image_pil, make_image, make_bounding_boxes, make_segmentation_mask, make_video],
    )
    def test_transform(self, param, value, make_input):
        input = make_input(self.INPUT_SIZE)

        check_sample_input = True
        if param == "fill":
            if isinstance(value, (tuple, list)):
                if isinstance(input, tv_tensors.Mask):
                    pytest.skip("F.pad_mask doesn't support non-scalar fill.")
                else:
                    check_sample_input = False

            kwargs = dict(
                # 1. size is required
                # 2. the fill parameter only has an affect if we need padding
                size=[s + 4 for s in self.INPUT_SIZE],
                fill=adapt_fill(value, dtype=input.dtype if isinstance(input, torch.Tensor) else torch.uint8),
            )
        else:
            kwargs = {param: value}

        check_transform(
            transforms.RandomCrop(**kwargs, pad_if_needed=True),
            input,
            check_v1_compatibility=param != "fill" or isinstance(value, (int, float)),
            check_sample_input=check_sample_input,
        )

    @pytest.mark.parametrize("padding", [1, (1, 1), (1, 1, 1, 1)])
    def test_transform_padding(self, padding):
        inpt = make_image(self.INPUT_SIZE)

        output_size = [s + 2 for s in F.get_size(inpt)]
        transform = transforms.RandomCrop(output_size, padding=padding)

        output = transform(inpt)

        assert F.get_size(output) == output_size

    @pytest.mark.parametrize("padding", [None, 1, (1, 1), (1, 1, 1, 1)])
    def test_transform_insufficient_padding(self, padding):
        inpt = make_image(self.INPUT_SIZE)

        output_size = [s + 3 for s in F.get_size(inpt)]
        transform = transforms.RandomCrop(output_size, padding=padding)

        with pytest.raises(ValueError, match="larger than (padded )?input image size"):
            transform(inpt)

    def test_transform_pad_if_needed(self):
        inpt = make_image(self.INPUT_SIZE)

        output_size = [s * 2 for s in F.get_size(inpt)]
        transform = transforms.RandomCrop(output_size, pad_if_needed=True)

        output = transform(inpt)

        assert F.get_size(output) == output_size

    @param_value_parametrization(
        size=[(10, 5), (25, 15), (25, 5), (10, 15)],
        fill=CORRECTNESS_FILLS,
        padding_mode=["constant", "edge", "reflect", "symmetric"],
    )
    @pytest.mark.parametrize("seed", list(range(5)))
    def test_transform_image_correctness(self, param, value, seed):
        kwargs = {param: value}
        if param != "size":
            # 1. size is required
            # 2. the fill / padding_mode parameters only have an affect if we need padding
            kwargs["size"] = [s + 4 for s in self.INPUT_SIZE]
        if param == "fill":
            kwargs["fill"] = adapt_fill(kwargs["fill"], dtype=torch.uint8)

        transform = transforms.RandomCrop(pad_if_needed=True, **kwargs)

        image = make_image(self.INPUT_SIZE)

        with freeze_rng_state():
            torch.manual_seed(seed)
            actual = transform(image)

            torch.manual_seed(seed)
            expected = F.to_image(transform(F.to_pil_image(image)))

        assert_equal(actual, expected)

    def _reference_crop_bounding_boxes(self, bounding_boxes, *, top, left, height, width):
        affine_matrix = np.array(
            [
                [1, 0, -left],
                [0, 1, -top],
            ],
        )
        return reference_affine_bounding_boxes_helper(
            bounding_boxes, affine_matrix=affine_matrix, new_canvas_size=(height, width)
        )

    @pytest.mark.parametrize("kwargs", CORRECTNESS_CROP_KWARGS)
    @pytest.mark.parametrize("format", list(tv_tensors.BoundingBoxFormat))
    @pytest.mark.parametrize("dtype", [torch.float32, torch.int64])
    @pytest.mark.parametrize("device", cpu_and_cuda())
    def test_functional_bounding_box_correctness(self, kwargs, format, dtype, device):
        bounding_boxes = make_bounding_boxes(self.INPUT_SIZE, format=format, dtype=dtype, device=device)

        actual = F.crop(bounding_boxes, **kwargs)
        expected = self._reference_crop_bounding_boxes(bounding_boxes, **kwargs)

        assert_equal(actual, expected, atol=1, rtol=0)
        assert_equal(F.get_size(actual), F.get_size(expected))

    @pytest.mark.parametrize("output_size", [(17, 11), (11, 17), (11, 11)])
    @pytest.mark.parametrize("format", list(tv_tensors.BoundingBoxFormat))
    @pytest.mark.parametrize("dtype", [torch.float32, torch.int64])
    @pytest.mark.parametrize("device", cpu_and_cuda())
    @pytest.mark.parametrize("seed", list(range(5)))
    def test_transform_bounding_boxes_correctness(self, output_size, format, dtype, device, seed):
        input_size = [s * 2 for s in output_size]
        bounding_boxes = make_bounding_boxes(input_size, format=format, dtype=dtype, device=device)

        transform = transforms.RandomCrop(output_size)

        with freeze_rng_state():
            torch.manual_seed(seed)
            params = transform._get_params([bounding_boxes])
            assert not params.pop("needs_pad")
            del params["padding"]
            assert params.pop("needs_crop")

            torch.manual_seed(seed)
            actual = transform(bounding_boxes)

        expected = self._reference_crop_bounding_boxes(bounding_boxes, **params)

        assert_equal(actual, expected)
        assert_equal(F.get_size(actual), F.get_size(expected))

    def test_errors(self):
        with pytest.raises(ValueError, match="Please provide only two dimensions"):
            transforms.RandomCrop([10, 12, 14])

        with pytest.raises(TypeError, match="Got inappropriate padding arg"):
            transforms.RandomCrop([10, 12], padding="abc")

        with pytest.raises(ValueError, match="Padding must be an int or a 1, 2, or 4"):
            transforms.RandomCrop([10, 12], padding=[-0.7, 0, 0.7])

        with pytest.raises(TypeError, match="Got inappropriate fill arg"):
            transforms.RandomCrop([10, 12], padding=1, fill="abc")

        with pytest.raises(ValueError, match="Padding mode should be either"):
            transforms.RandomCrop([10, 12], padding=1, padding_mode="abc")


class TestErase:
    INPUT_SIZE = (17, 11)
    FUNCTIONAL_KWARGS = dict(
        zip("ijhwv", [2, 2, 10, 8, torch.tensor(0.0, dtype=torch.float32, device="cpu").reshape(-1, 1, 1)])
    )

    @pytest.mark.parametrize("dtype", [torch.float32, torch.uint8])
    @pytest.mark.parametrize("device", cpu_and_cuda())
    def test_kernel_image(self, dtype, device):
        check_kernel(F.erase_image, make_image(self.INPUT_SIZE, dtype=dtype, device=device), **self.FUNCTIONAL_KWARGS)

    @pytest.mark.parametrize("dtype", [torch.float32, torch.uint8])
    @pytest.mark.parametrize("device", cpu_and_cuda())
    def test_kernel_image_inplace(self, dtype, device):
        input = make_image(self.INPUT_SIZE, dtype=dtype, device=device)
        input_version = input._version

        output_out_of_place = F.erase_image(input, **self.FUNCTIONAL_KWARGS)
        assert output_out_of_place.data_ptr() != input.data_ptr()
        assert output_out_of_place is not input

        output_inplace = F.erase_image(input, **self.FUNCTIONAL_KWARGS, inplace=True)
        assert output_inplace.data_ptr() == input.data_ptr()
        assert output_inplace._version > input_version
        assert output_inplace is input

        assert_equal(output_inplace, output_out_of_place)

    def test_kernel_video(self):
        check_kernel(F.erase_video, make_video(self.INPUT_SIZE), **self.FUNCTIONAL_KWARGS)

    @pytest.mark.parametrize(
        "make_input",
        [make_image_tensor, make_image_pil, make_image, make_video],
    )
    def test_functional(self, make_input):
        check_functional(F.erase, make_input(), **self.FUNCTIONAL_KWARGS)

    @pytest.mark.parametrize(
        ("kernel", "input_type"),
        [
            (F.erase_image, torch.Tensor),
            (F._erase_image_pil, PIL.Image.Image),
            (F.erase_image, tv_tensors.Image),
            (F.erase_video, tv_tensors.Video),
        ],
    )
    def test_functional_signature(self, kernel, input_type):
        check_functional_kernel_signature_match(F.erase, kernel=kernel, input_type=input_type)

    @pytest.mark.parametrize(
        "make_input",
        [make_image_tensor, make_image_pil, make_image, make_video],
    )
    @pytest.mark.parametrize("device", cpu_and_cuda())
    def test_transform(self, make_input, device):
        input = make_input(device=device)

        with pytest.warns(UserWarning, match="currently passing through inputs of type"):
            check_transform(
                transforms.RandomErasing(p=1),
                input,
                check_v1_compatibility=not isinstance(input, PIL.Image.Image),
            )

    def _reference_erase_image(self, image, *, i, j, h, w, v):
        mask = torch.zeros_like(image, dtype=torch.bool)
        mask[..., i : i + h, j : j + w] = True

        # The broadcasting and type casting logic is handled automagically in the kernel through indexing
        value = torch.broadcast_to(v, (*image.shape[:-2], h, w)).to(image)

        erased_image = torch.empty_like(image)
        erased_image[mask] = value.flatten()
        erased_image[~mask] = image[~mask]

        return erased_image

    @pytest.mark.parametrize("dtype", [torch.float32, torch.uint8])
    @pytest.mark.parametrize("device", cpu_and_cuda())
    def test_functional_image_correctness(self, dtype, device):
        image = make_image(dtype=dtype, device=device)

        actual = F.erase(image, **self.FUNCTIONAL_KWARGS)
        expected = self._reference_erase_image(image, **self.FUNCTIONAL_KWARGS)

        assert_equal(actual, expected)

    @param_value_parametrization(
        scale=[(0.1, 0.2), [0.0, 1.0]],
        ratio=[(0.3, 0.7), [0.1, 5.0]],
        value=[0, 0.5, (0, 1, 0), [-0.2, 0.0, 1.3], "random"],
    )
    @pytest.mark.parametrize("dtype", [torch.float32, torch.uint8])
    @pytest.mark.parametrize("device", cpu_and_cuda())
    @pytest.mark.parametrize("seed", list(range(5)))
    def test_transform_image_correctness(self, param, value, dtype, device, seed):
        transform = transforms.RandomErasing(**{param: value}, p=1)

        image = make_image(dtype=dtype, device=device)

        with freeze_rng_state():
            torch.manual_seed(seed)
            # This emulates the random apply check that happens before _get_params is called
            torch.rand(1)
            params = transform._get_params([image])

            torch.manual_seed(seed)
            actual = transform(image)

        expected = self._reference_erase_image(image, **params)

        assert_equal(actual, expected)

    def test_transform_errors(self):
        with pytest.raises(TypeError, match="Argument value should be either a number or str or a sequence"):
            transforms.RandomErasing(value={})

        with pytest.raises(ValueError, match="If value is str, it should be 'random'"):
            transforms.RandomErasing(value="abc")

        with pytest.raises(TypeError, match="Scale should be a sequence"):
            transforms.RandomErasing(scale=123)

        with pytest.raises(TypeError, match="Ratio should be a sequence"):
            transforms.RandomErasing(ratio=123)

        with pytest.raises(ValueError, match="Scale should be between 0 and 1"):
            transforms.RandomErasing(scale=[-1, 2])

        transform = transforms.RandomErasing(value=[1, 2, 3, 4])

        with pytest.raises(ValueError, match="If value is a sequence, it should have either a single value"):
            transform._get_params([make_image()])


class TestGaussianBlur:
    @pytest.mark.parametrize("kernel_size", [1, 3, (3, 1), [3, 5]])
    @pytest.mark.parametrize("sigma", [None, 1.0, 1, (0.5,), [0.3], (0.3, 0.7), [0.9, 0.2]])
    def test_kernel_image(self, kernel_size, sigma):
        check_kernel(
            F.gaussian_blur_image,
            make_image(),
            kernel_size=kernel_size,
            sigma=sigma,
            check_scripted_vs_eager=not (isinstance(kernel_size, int) or isinstance(sigma, (float, int))),
        )

    def test_kernel_image_errors(self):
        image = make_image_tensor()

        with pytest.raises(ValueError, match="kernel_size is a sequence its length should be 2"):
            F.gaussian_blur_image(image, kernel_size=[1, 2, 3])

        for kernel_size in [2, -1]:
            with pytest.raises(ValueError, match="kernel_size should have odd and positive integers"):
                F.gaussian_blur_image(image, kernel_size=kernel_size)

        with pytest.raises(ValueError, match="sigma is a sequence, its length should be 2"):
            F.gaussian_blur_image(image, kernel_size=1, sigma=[1, 2, 3])

        with pytest.raises(TypeError, match="sigma should be either float or sequence of floats"):
            F.gaussian_blur_image(image, kernel_size=1, sigma=object())

        with pytest.raises(ValueError, match="sigma should have positive values"):
            F.gaussian_blur_image(image, kernel_size=1, sigma=-1)

    def test_kernel_video(self):
        check_kernel(F.gaussian_blur_video, make_video(), kernel_size=(3, 3))

    @pytest.mark.parametrize(
        "make_input",
        [make_image_tensor, make_image_pil, make_image, make_video],
    )
    def test_functional(self, make_input):
        check_functional(F.gaussian_blur, make_input(), kernel_size=(3, 3))

    @pytest.mark.parametrize(
        ("kernel", "input_type"),
        [
            (F.gaussian_blur_image, torch.Tensor),
            (F._gaussian_blur_image_pil, PIL.Image.Image),
            (F.gaussian_blur_image, tv_tensors.Image),
            (F.gaussian_blur_video, tv_tensors.Video),
        ],
    )
    def test_functional_signature(self, kernel, input_type):
        check_functional_kernel_signature_match(F.gaussian_blur, kernel=kernel, input_type=input_type)

    @pytest.mark.parametrize(
        "make_input",
        [make_image_tensor, make_image_pil, make_image, make_bounding_boxes, make_segmentation_mask, make_video],
    )
    @pytest.mark.parametrize("device", cpu_and_cuda())
    @pytest.mark.parametrize("sigma", [5, 2.0, (0.5, 2), [1.3, 2.7]])
    def test_transform(self, make_input, device, sigma):
        check_transform(transforms.GaussianBlur(kernel_size=3, sigma=sigma), make_input(device=device))

    def test_assertions(self):
        with pytest.raises(ValueError, match="Kernel size should be a tuple/list of two integers"):
            transforms.GaussianBlur([10, 12, 14])

        with pytest.raises(ValueError, match="Kernel size value should be an odd and positive number"):
            transforms.GaussianBlur(4)

        with pytest.raises(ValueError, match="If sigma is a sequence its length should be 1 or 2. Got 3"):
            transforms.GaussianBlur(3, sigma=[1, 2, 3])

        with pytest.raises(ValueError, match="sigma values should be positive and of the form"):
            transforms.GaussianBlur(3, sigma=-1.0)

        with pytest.raises(ValueError, match="sigma values should be positive and of the form"):
            transforms.GaussianBlur(3, sigma=[2.0, 1.0])

        with pytest.raises(TypeError, match="sigma should be a number or a sequence of numbers"):
            transforms.GaussianBlur(3, sigma={})

    @pytest.mark.parametrize("sigma", [10.0, [10.0, 12.0], (10, 12.0), [10]])
    def test__get_params(self, sigma):
        transform = transforms.GaussianBlur(3, sigma=sigma)
        params = transform._get_params([])

        if isinstance(sigma, float):
            assert params["sigma"][0] == params["sigma"][1] == sigma
        elif isinstance(sigma, list) and len(sigma) == 1:
            assert params["sigma"][0] == params["sigma"][1] == sigma[0]
        else:
            assert sigma[0] <= params["sigma"][0] <= sigma[1]
            assert sigma[0] <= params["sigma"][1] <= sigma[1]

    # np_img = np.arange(3 * 10 * 12, dtype="uint8").reshape((10, 12, 3))
    # np_img2 = np.arange(26 * 28, dtype="uint8").reshape((26, 28))
    # {
    #     "10_12_3__3_3_0.8": cv2.GaussianBlur(np_img, ksize=(3, 3), sigmaX=0.8),
    #     "10_12_3__3_3_0.5": cv2.GaussianBlur(np_img, ksize=(3, 3), sigmaX=0.5),
    #     "10_12_3__3_5_0.8": cv2.GaussianBlur(np_img, ksize=(3, 5), sigmaX=0.8),
    #     "10_12_3__3_5_0.5": cv2.GaussianBlur(np_img, ksize=(3, 5), sigmaX=0.5),
    #     "26_28_1__23_23_1.7": cv2.GaussianBlur(np_img2, ksize=(23, 23), sigmaX=1.7),
    # }
    REFERENCE_GAUSSIAN_BLUR_IMAGE_RESULTS = torch.load(
        Path(__file__).parent / "assets" / "gaussian_blur_opencv_results.pt"
    )

    @pytest.mark.parametrize(
        ("dimensions", "kernel_size", "sigma"),
        [
            ((3, 10, 12), (3, 3), 0.8),
            ((3, 10, 12), (3, 3), 0.5),
            ((3, 10, 12), (3, 5), 0.8),
            ((3, 10, 12), (3, 5), 0.5),
            ((1, 26, 28), (23, 23), 1.7),
        ],
    )
    @pytest.mark.parametrize("dtype", [torch.float32, torch.float64, torch.float16])
    @pytest.mark.parametrize("device", cpu_and_cuda())
    def test_functional_image_correctness(self, dimensions, kernel_size, sigma, dtype, device):
        if dtype is torch.float16 and device == "cpu":
            pytest.skip("The CPU implementation of float16 on CPU differs from opencv")

        num_channels, height, width = dimensions

        reference_results_key = f"{height}_{width}_{num_channels}__{kernel_size[0]}_{kernel_size[1]}_{sigma}"
        expected = (
            torch.tensor(self.REFERENCE_GAUSSIAN_BLUR_IMAGE_RESULTS[reference_results_key])
            .reshape(height, width, num_channels)
            .permute(2, 0, 1)
            .to(dtype=dtype, device=device)
        )

        image = tv_tensors.Image(
            torch.arange(num_channels * height * width, dtype=torch.uint8)
            .reshape(height, width, num_channels)
            .permute(2, 0, 1),
            dtype=dtype,
            device=device,
        )

        actual = F.gaussian_blur_image(image, kernel_size=kernel_size, sigma=sigma)

        torch.testing.assert_close(actual, expected, rtol=0, atol=1)


class TestAutoAugmentTransforms:
    # These transforms have a lot of branches in their `forward()` passes which are conditioned on random sampling.
    # It's typically very hard to test the effect on some parameters without heavy mocking logic.
    # This class adds correctness tests for the kernels that are specific to those transforms. The rest of kernels, e.g.
    # rotate, are tested in their respective classes. The rest of the tests here are mostly smoke tests.

    def _reference_shear_translate(self, image, *, transform_id, magnitude, interpolation, fill):
        if isinstance(image, PIL.Image.Image):
            input = image
        else:
            input = F.to_pil_image(image)

        matrix = {
            "ShearX": (1, magnitude, 0, 0, 1, 0),
            "ShearY": (1, 0, 0, magnitude, 1, 0),
            "TranslateX": (1, 0, -int(magnitude), 0, 1, 0),
            "TranslateY": (1, 0, 0, 0, 1, -int(magnitude)),
        }[transform_id]

        output = input.transform(
            input.size, PIL.Image.AFFINE, matrix, resample=pil_modes_mapping[interpolation], fill=fill
        )

        if isinstance(image, PIL.Image.Image):
            return output
        else:
            return F.to_image(output)

    @pytest.mark.parametrize("transform_id", ["ShearX", "ShearY", "TranslateX", "TranslateY"])
    @pytest.mark.parametrize("magnitude", [0.3, -0.2, 0.0])
    @pytest.mark.parametrize(
        "interpolation", [transforms.InterpolationMode.NEAREST, transforms.InterpolationMode.BILINEAR]
    )
    @pytest.mark.parametrize("fill", CORRECTNESS_FILLS)
    @pytest.mark.parametrize("input_type", ["Tensor", "PIL"])
    def test_correctness_shear_translate(self, transform_id, magnitude, interpolation, fill, input_type):
        # ShearX/Y and TranslateX/Y are the only ops that are native to the AA transforms. They are modeled after the
        # reference implementation:
        # https://github.com/tensorflow/models/blob/885fda091c46c59d6c7bb5c7e760935eacc229da/research/autoaugment/augmentation_transforms.py#L273-L362
        # All other ops are checked in their respective dedicated tests.

        image = make_image(dtype=torch.uint8, device="cpu")
        if input_type == "PIL":
            image = F.to_pil_image(image)

        if "Translate" in transform_id:
            # For TranslateX/Y magnitude is a value in pixels
            magnitude *= min(F.get_size(image))

        actual = transforms.AutoAugment()._apply_image_or_video_transform(
            image,
            transform_id=transform_id,
            magnitude=magnitude,
            interpolation=interpolation,
            fill={type(image): fill},
        )
        expected = self._reference_shear_translate(
            image, transform_id=transform_id, magnitude=magnitude, interpolation=interpolation, fill=fill
        )

        if input_type == "PIL":
            actual, expected = F.to_image(actual), F.to_image(expected)

        if "Shear" in transform_id and input_type == "Tensor":
            mae = (actual.float() - expected.float()).abs().mean()
            assert mae < (12 if interpolation is transforms.InterpolationMode.NEAREST else 5)
        else:
            assert_close(actual, expected, rtol=0, atol=1)

    def _sample_input_adapter(self, transform, input, device):
        adapted_input = {}
        image_or_video_found = False
        for key, value in input.items():
            if isinstance(value, (tv_tensors.BoundingBoxes, tv_tensors.Mask)):
                # AA transforms don't support bounding boxes or masks
                continue
            elif check_type(value, (tv_tensors.Image, tv_tensors.Video, is_pure_tensor, PIL.Image.Image)):
                if image_or_video_found:
                    # AA transforms only support a single image or video
                    continue
                image_or_video_found = True
            adapted_input[key] = value
        return adapted_input

    @pytest.mark.parametrize(
        "transform",
        [transforms.AutoAugment(), transforms.RandAugment(), transforms.TrivialAugmentWide(), transforms.AugMix()],
    )
    @pytest.mark.parametrize("make_input", [make_image_tensor, make_image_pil, make_image, make_video])
    @pytest.mark.parametrize("dtype", [torch.uint8, torch.float32])
    @pytest.mark.parametrize("device", cpu_and_cuda())
    def test_transform_smoke(self, transform, make_input, dtype, device):
        if make_input is make_image_pil and not (dtype is torch.uint8 and device == "cpu"):
            pytest.skip(
                "PIL image tests with parametrization other than dtype=torch.uint8 and device='cpu' "
                "will degenerate to that anyway."
            )
        input = make_input(dtype=dtype, device=device)

        with freeze_rng_state():
            # By default every test starts from the same random seed. This leads to minimal coverage of the sampling
            # that happens inside forward(). To avoid calling the transform multiple times to achieve higher coverage,
            # we build a reproducible random seed from the input type, dtype, and device.
            torch.manual_seed(hash((make_input, dtype, device)))

            # For v2, we changed the random sampling of the AA transforms. This makes it impossible to compare the v1
            # and v2 outputs without complicated mocking and monkeypatching. Thus, we skip the v1 compatibility checks
            # here and only check if we can script the v2 transform and subsequently call the result.
            check_transform(
                transform, input, check_v1_compatibility=False, check_sample_input=self._sample_input_adapter
            )

            if type(input) is torch.Tensor and dtype is torch.uint8:
                _script(transform)(input)

    def test_auto_augment_policy_error(self):
        with pytest.raises(ValueError, match="provided policy"):
            transforms.AutoAugment(policy=None)

    @pytest.mark.parametrize("severity", [0, 11])
    def test_aug_mix_severity_error(self, severity):
        with pytest.raises(ValueError, match="severity must be between"):
            transforms.AugMix(severity=severity)


class TestConvertBoundingBoxFormat:
    old_new_formats = list(itertools.permutations(iter(tv_tensors.BoundingBoxFormat), 2))

    @pytest.mark.parametrize(("old_format", "new_format"), old_new_formats)
    def test_kernel(self, old_format, new_format):
        check_kernel(
            F.convert_bounding_box_format,
            make_bounding_boxes(format=old_format),
            new_format=new_format,
            old_format=old_format,
        )

    @pytest.mark.parametrize("format", list(tv_tensors.BoundingBoxFormat))
    @pytest.mark.parametrize("inplace", [False, True])
    def test_kernel_noop(self, format, inplace):
        input = make_bounding_boxes(format=format).as_subclass(torch.Tensor)
        input_version = input._version

        output = F.convert_bounding_box_format(input, old_format=format, new_format=format, inplace=inplace)

        assert output is input
        assert output.data_ptr() == input.data_ptr()
        assert output._version == input_version

    @pytest.mark.parametrize(("old_format", "new_format"), old_new_formats)
    def test_kernel_inplace(self, old_format, new_format):
        input = make_bounding_boxes(format=old_format).as_subclass(torch.Tensor)
        input_version = input._version

        output_out_of_place = F.convert_bounding_box_format(input, old_format=old_format, new_format=new_format)
        assert output_out_of_place.data_ptr() != input.data_ptr()
        assert output_out_of_place is not input

        output_inplace = F.convert_bounding_box_format(
            input, old_format=old_format, new_format=new_format, inplace=True
        )
        assert output_inplace.data_ptr() == input.data_ptr()
        assert output_inplace._version > input_version
        assert output_inplace is input

        assert_equal(output_inplace, output_out_of_place)

    @pytest.mark.parametrize(("old_format", "new_format"), old_new_formats)
    def test_functional(self, old_format, new_format):
        check_functional(F.convert_bounding_box_format, make_bounding_boxes(format=old_format), new_format=new_format)

    @pytest.mark.parametrize(("old_format", "new_format"), old_new_formats)
    @pytest.mark.parametrize("format_type", ["enum", "str"])
    def test_transform(self, old_format, new_format, format_type):
        check_transform(
            transforms.ConvertBoundingBoxFormat(new_format.name if format_type == "str" else new_format),
            make_bounding_boxes(format=old_format),
        )

    @pytest.mark.parametrize(("old_format", "new_format"), old_new_formats)
    def test_strings(self, old_format, new_format):
        # Non-regression test for https://github.com/pytorch/vision/issues/8258
        input = tv_tensors.BoundingBoxes(torch.tensor([[10, 10, 20, 20]]), format=old_format, canvas_size=(50, 50))
        expected = self._reference_convert_bounding_box_format(input, new_format)

        old_format = old_format.name
        new_format = new_format.name

        out_functional = F.convert_bounding_box_format(input, new_format=new_format)
        out_functional_tensor = F.convert_bounding_box_format(
            input.as_subclass(torch.Tensor), old_format=old_format, new_format=new_format
        )
        out_transform = transforms.ConvertBoundingBoxFormat(new_format)(input)
        for out in (out_functional, out_functional_tensor, out_transform):
            assert_equal(out, expected)

    def _reference_convert_bounding_box_format(self, bounding_boxes, new_format):
        return tv_tensors.wrap(
            torchvision.ops.box_convert(
                bounding_boxes.as_subclass(torch.Tensor),
                in_fmt=bounding_boxes.format.name.lower(),
                out_fmt=new_format.name.lower(),
            ).to(bounding_boxes.dtype),
            like=bounding_boxes,
            format=new_format,
        )

    @pytest.mark.parametrize(("old_format", "new_format"), old_new_formats)
    @pytest.mark.parametrize("dtype", [torch.int64, torch.float32])
    @pytest.mark.parametrize("device", cpu_and_cuda())
    @pytest.mark.parametrize("fn_type", ["functional", "transform"])
    def test_correctness(self, old_format, new_format, dtype, device, fn_type):
        bounding_boxes = make_bounding_boxes(format=old_format, dtype=dtype, device=device)

        if fn_type == "functional":
            fn = functools.partial(F.convert_bounding_box_format, new_format=new_format)
        else:
            fn = transforms.ConvertBoundingBoxFormat(format=new_format)

        actual = fn(bounding_boxes)
        expected = self._reference_convert_bounding_box_format(bounding_boxes, new_format)

        assert_equal(actual, expected)

    def test_errors(self):
        input_tv_tensor = make_bounding_boxes()
        input_pure_tensor = input_tv_tensor.as_subclass(torch.Tensor)

        for input in [input_tv_tensor, input_pure_tensor]:
            with pytest.raises(TypeError, match="missing 1 required argument: 'new_format'"):
                F.convert_bounding_box_format(input)

        with pytest.raises(ValueError, match="`old_format` has to be passed"):
            F.convert_bounding_box_format(input_pure_tensor, new_format=input_tv_tensor.format)

        with pytest.raises(ValueError, match="`old_format` must not be passed"):
            F.convert_bounding_box_format(
                input_tv_tensor, old_format=input_tv_tensor.format, new_format=input_tv_tensor.format
            )


class TestResizedCrop:
    INPUT_SIZE = (17, 11)
    CROP_KWARGS = dict(top=2, left=2, height=5, width=7)
    OUTPUT_SIZE = (19, 32)

    @pytest.mark.parametrize(
        ("kernel", "make_input"),
        [
            (F.resized_crop_image, make_image),
            (F.resized_crop_bounding_boxes, make_bounding_boxes),
            (F.resized_crop_mask, make_segmentation_mask),
            (F.resized_crop_mask, make_detection_masks),
            (F.resized_crop_video, make_video),
        ],
    )
    def test_kernel(self, kernel, make_input):
        input = make_input(self.INPUT_SIZE)
        if isinstance(input, tv_tensors.BoundingBoxes):
            extra_kwargs = dict(format=input.format)
        elif isinstance(input, tv_tensors.Mask):
            extra_kwargs = dict()
        else:
            extra_kwargs = dict(antialias=True)

        check_kernel(kernel, input, **self.CROP_KWARGS, size=self.OUTPUT_SIZE, **extra_kwargs)

    @pytest.mark.parametrize(
        "make_input",
        [make_image_tensor, make_image_pil, make_image, make_bounding_boxes, make_segmentation_mask, make_video],
    )
    def test_functional(self, make_input):
        check_functional(
            F.resized_crop, make_input(self.INPUT_SIZE), **self.CROP_KWARGS, size=self.OUTPUT_SIZE, antialias=True
        )

    @pytest.mark.parametrize(
        ("kernel", "input_type"),
        [
            (F.resized_crop_image, torch.Tensor),
            (F._resized_crop_image_pil, PIL.Image.Image),
            (F.resized_crop_image, tv_tensors.Image),
            (F.resized_crop_bounding_boxes, tv_tensors.BoundingBoxes),
            (F.resized_crop_mask, tv_tensors.Mask),
            (F.resized_crop_video, tv_tensors.Video),
        ],
    )
    def test_functional_signature(self, kernel, input_type):
        check_functional_kernel_signature_match(F.resized_crop, kernel=kernel, input_type=input_type)

    @param_value_parametrization(
        scale=[(0.1, 0.2), [0.0, 1.0]],
        ratio=[(0.3, 0.7), [0.1, 5.0]],
    )
    @pytest.mark.parametrize(
        "make_input",
        [make_image_tensor, make_image_pil, make_image, make_bounding_boxes, make_segmentation_mask, make_video],
    )
    def test_transform(self, param, value, make_input):
        check_transform(
            transforms.RandomResizedCrop(size=self.OUTPUT_SIZE, **{param: value}, antialias=True),
            make_input(self.INPUT_SIZE),
            check_v1_compatibility=dict(rtol=0, atol=1),
        )

    # `InterpolationMode.NEAREST` is modeled after the buggy `INTER_NEAREST` interpolation of CV2.
    # The PIL equivalent of `InterpolationMode.NEAREST` is `InterpolationMode.NEAREST_EXACT`
    @pytest.mark.parametrize("interpolation", set(INTERPOLATION_MODES) - {transforms.InterpolationMode.NEAREST})
    def test_functional_image_correctness(self, interpolation):
        image = make_image(self.INPUT_SIZE, dtype=torch.uint8)

        actual = F.resized_crop(
            image, **self.CROP_KWARGS, size=self.OUTPUT_SIZE, interpolation=interpolation, antialias=True
        )
        expected = F.to_image(
            F.resized_crop(
                F.to_pil_image(image), **self.CROP_KWARGS, size=self.OUTPUT_SIZE, interpolation=interpolation
            )
        )

        torch.testing.assert_close(actual, expected, atol=1, rtol=0)

    def _reference_resized_crop_bounding_boxes(self, bounding_boxes, *, top, left, height, width, size):
        new_height, new_width = size

        crop_affine_matrix = np.array(
            [
                [1, 0, -left],
                [0, 1, -top],
                [0, 0, 1],
            ],
        )
        resize_affine_matrix = np.array(
            [
                [new_width / width, 0, 0],
                [0, new_height / height, 0],
                [0, 0, 1],
            ],
        )
        affine_matrix = (resize_affine_matrix @ crop_affine_matrix)[:2, :]

        return reference_affine_bounding_boxes_helper(
            bounding_boxes,
            affine_matrix=affine_matrix,
            new_canvas_size=size,
        )

    @pytest.mark.parametrize("format", list(tv_tensors.BoundingBoxFormat))
    def test_functional_bounding_boxes_correctness(self, format):
        bounding_boxes = make_bounding_boxes(self.INPUT_SIZE, format=format)

        actual = F.resized_crop(bounding_boxes, **self.CROP_KWARGS, size=self.OUTPUT_SIZE)
        expected = self._reference_resized_crop_bounding_boxes(
            bounding_boxes, **self.CROP_KWARGS, size=self.OUTPUT_SIZE
        )

        assert_equal(actual, expected)
        assert_equal(F.get_size(actual), F.get_size(expected))

    def test_transform_errors_warnings(self):
        with pytest.raises(ValueError, match="provide only two dimensions"):
            transforms.RandomResizedCrop(size=(1, 2, 3))

        with pytest.raises(TypeError, match="Scale should be a sequence"):
            transforms.RandomResizedCrop(size=self.INPUT_SIZE, scale=123)

        with pytest.raises(TypeError, match="Ratio should be a sequence"):
            transforms.RandomResizedCrop(size=self.INPUT_SIZE, ratio=123)

        for param in ["scale", "ratio"]:
            with pytest.warns(match="Scale and ratio should be of kind"):
                transforms.RandomResizedCrop(size=self.INPUT_SIZE, **{param: [1, 0]})


class TestPad:
    EXHAUSTIVE_TYPE_PADDINGS = [1, (1,), (1, 2), (1, 2, 3, 4), [1], [1, 2], [1, 2, 3, 4]]
    CORRECTNESS_PADDINGS = [
        padding
        for padding in EXHAUSTIVE_TYPE_PADDINGS
        if isinstance(padding, int) or isinstance(padding, list) and len(padding) > 1
    ]
    PADDING_MODES = ["constant", "symmetric", "edge", "reflect"]

    @param_value_parametrization(
        padding=EXHAUSTIVE_TYPE_PADDINGS,
        fill=EXHAUSTIVE_TYPE_FILLS,
        padding_mode=PADDING_MODES,
    )
    @pytest.mark.parametrize("dtype", [torch.uint8, torch.float32])
    @pytest.mark.parametrize("device", cpu_and_cuda())
    def test_kernel_image(self, param, value, dtype, device):
        if param == "fill":
            value = adapt_fill(value, dtype=dtype)
        kwargs = {param: value}
        if param != "padding":
            kwargs["padding"] = [1]

        image = make_image(dtype=dtype, device=device)

        check_kernel(
            F.pad_image,
            image,
            **kwargs,
            check_scripted_vs_eager=not (
                (param == "padding" and isinstance(value, int))
                # See https://github.com/pytorch/vision/pull/7252#issue-1585585521 for details
                or (
                    param == "fill"
                    and (
                        isinstance(value, tuple) or (isinstance(value, list) and any(isinstance(v, int) for v in value))
                    )
                )
            ),
        )

    @pytest.mark.parametrize("format", list(tv_tensors.BoundingBoxFormat))
    def test_kernel_bounding_boxes(self, format):
        bounding_boxes = make_bounding_boxes(format=format)
        check_kernel(
            F.pad_bounding_boxes,
            bounding_boxes,
            format=bounding_boxes.format,
            canvas_size=bounding_boxes.canvas_size,
            padding=[1],
        )

    @pytest.mark.parametrize("padding_mode", ["symmetric", "edge", "reflect"])
    def test_kernel_bounding_boxes_errors(self, padding_mode):
        bounding_boxes = make_bounding_boxes()
        with pytest.raises(ValueError, match=f"'{padding_mode}' is not supported"):
            F.pad_bounding_boxes(
                bounding_boxes,
                format=bounding_boxes.format,
                canvas_size=bounding_boxes.canvas_size,
                padding=[1],
                padding_mode=padding_mode,
            )

    @pytest.mark.parametrize("make_mask", [make_segmentation_mask, make_detection_masks])
    def test_kernel_mask(self, make_mask):
        check_kernel(F.pad_mask, make_mask(), padding=[1])

    @pytest.mark.parametrize("fill", [[1], (0,), [1, 0, 1], (0, 1, 0)])
    def test_kernel_mask_errors(self, fill):
        with pytest.raises(ValueError, match="Non-scalar fill value is not supported"):
            F.pad_mask(make_segmentation_mask(), padding=[1], fill=fill)

    def test_kernel_video(self):
        check_kernel(F.pad_video, make_video(), padding=[1])

    @pytest.mark.parametrize(
        "make_input",
        [make_image_tensor, make_image_pil, make_image, make_bounding_boxes, make_segmentation_mask, make_video],
    )
    def test_functional(self, make_input):
        check_functional(F.pad, make_input(), padding=[1])

    @pytest.mark.parametrize(
        ("kernel", "input_type"),
        [
            (F.pad_image, torch.Tensor),
            # The PIL kernel uses fill=0 as default rather than fill=None as all others.
            # Since the whole fill story is already really inconsistent, we won't introduce yet another case to allow
            # for this test to pass.
            # See https://github.com/pytorch/vision/issues/6623 for a discussion.
            # (F._pad_image_pil, PIL.Image.Image),
            (F.pad_image, tv_tensors.Image),
            (F.pad_bounding_boxes, tv_tensors.BoundingBoxes),
            (F.pad_mask, tv_tensors.Mask),
            (F.pad_video, tv_tensors.Video),
        ],
    )
    def test_functional_signature(self, kernel, input_type):
        check_functional_kernel_signature_match(F.pad, kernel=kernel, input_type=input_type)

    @pytest.mark.parametrize(
        "make_input",
        [make_image_tensor, make_image_pil, make_image, make_bounding_boxes, make_segmentation_mask, make_video],
    )
    def test_transform(self, make_input):
        check_transform(transforms.Pad(padding=[1]), make_input())

    def test_transform_errors(self):
        with pytest.raises(TypeError, match="Got inappropriate padding arg"):
            transforms.Pad("abc")

        with pytest.raises(ValueError, match="Padding must be an int or a 1, 2, or 4"):
            transforms.Pad([-0.7, 0, 0.7])

        with pytest.raises(TypeError, match="Got inappropriate fill arg"):
            transforms.Pad(12, fill="abc")

        with pytest.raises(ValueError, match="Padding mode should be either"):
            transforms.Pad(12, padding_mode="abc")

    @pytest.mark.parametrize("padding", CORRECTNESS_PADDINGS)
    @pytest.mark.parametrize(
        ("padding_mode", "fill"),
        [
            *[("constant", fill) for fill in CORRECTNESS_FILLS],
            *[(padding_mode, None) for padding_mode in ["symmetric", "edge", "reflect"]],
        ],
    )
    @pytest.mark.parametrize("fn", [F.pad, transform_cls_to_functional(transforms.Pad)])
    def test_image_correctness(self, padding, padding_mode, fill, fn):
        image = make_image(dtype=torch.uint8, device="cpu")

        fill = adapt_fill(fill, dtype=torch.uint8)

        actual = fn(image, padding=padding, padding_mode=padding_mode, fill=fill)
        expected = F.to_image(F.pad(F.to_pil_image(image), padding=padding, padding_mode=padding_mode, fill=fill))

        assert_equal(actual, expected)

    def _reference_pad_bounding_boxes(self, bounding_boxes, *, padding):
        if isinstance(padding, int):
            padding = [padding]
        left, top, right, bottom = padding * (4 // len(padding))

        affine_matrix = np.array(
            [
                [1, 0, left],
                [0, 1, top],
            ],
        )

        height = bounding_boxes.canvas_size[0] + top + bottom
        width = bounding_boxes.canvas_size[1] + left + right

        return reference_affine_bounding_boxes_helper(
            bounding_boxes, affine_matrix=affine_matrix, new_canvas_size=(height, width)
        )

    @pytest.mark.parametrize("padding", CORRECTNESS_PADDINGS)
    @pytest.mark.parametrize("format", list(tv_tensors.BoundingBoxFormat))
    @pytest.mark.parametrize("dtype", [torch.int64, torch.float32])
    @pytest.mark.parametrize("device", cpu_and_cuda())
    @pytest.mark.parametrize("fn", [F.pad, transform_cls_to_functional(transforms.Pad)])
    def test_bounding_boxes_correctness(self, padding, format, dtype, device, fn):
        bounding_boxes = make_bounding_boxes(format=format, dtype=dtype, device=device)

        actual = fn(bounding_boxes, padding=padding)
        expected = self._reference_pad_bounding_boxes(bounding_boxes, padding=padding)

        assert_equal(actual, expected)


class TestCenterCrop:
    INPUT_SIZE = (17, 11)
    OUTPUT_SIZES = [(3, 5), (5, 3), (4, 4), (21, 9), (13, 15), (19, 14), 3, (4,), [5], INPUT_SIZE]

    @pytest.mark.parametrize("output_size", OUTPUT_SIZES)
    @pytest.mark.parametrize("dtype", [torch.int64, torch.float32])
    @pytest.mark.parametrize("device", cpu_and_cuda())
    def test_kernel_image(self, output_size, dtype, device):
        check_kernel(
            F.center_crop_image,
            make_image(self.INPUT_SIZE, dtype=dtype, device=device),
            output_size=output_size,
            check_scripted_vs_eager=not isinstance(output_size, int),
        )

    @pytest.mark.parametrize("output_size", OUTPUT_SIZES)
    @pytest.mark.parametrize("format", list(tv_tensors.BoundingBoxFormat))
    def test_kernel_bounding_boxes(self, output_size, format):
        bounding_boxes = make_bounding_boxes(self.INPUT_SIZE, format=format)
        check_kernel(
            F.center_crop_bounding_boxes,
            bounding_boxes,
            format=bounding_boxes.format,
            canvas_size=bounding_boxes.canvas_size,
            output_size=output_size,
            check_scripted_vs_eager=not isinstance(output_size, int),
        )

    @pytest.mark.parametrize("make_mask", [make_segmentation_mask, make_detection_masks])
    def test_kernel_mask(self, make_mask):
        check_kernel(F.center_crop_mask, make_mask(), output_size=self.OUTPUT_SIZES[0])

    def test_kernel_video(self):
        check_kernel(F.center_crop_video, make_video(self.INPUT_SIZE), output_size=self.OUTPUT_SIZES[0])

    @pytest.mark.parametrize(
        "make_input",
        [make_image_tensor, make_image_pil, make_image, make_bounding_boxes, make_segmentation_mask, make_video],
    )
    def test_functional(self, make_input):
        check_functional(F.center_crop, make_input(self.INPUT_SIZE), output_size=self.OUTPUT_SIZES[0])

    @pytest.mark.parametrize(
        ("kernel", "input_type"),
        [
            (F.center_crop_image, torch.Tensor),
            (F._center_crop_image_pil, PIL.Image.Image),
            (F.center_crop_image, tv_tensors.Image),
            (F.center_crop_bounding_boxes, tv_tensors.BoundingBoxes),
            (F.center_crop_mask, tv_tensors.Mask),
            (F.center_crop_video, tv_tensors.Video),
        ],
    )
    def test_functional_signature(self, kernel, input_type):
        check_functional_kernel_signature_match(F.center_crop, kernel=kernel, input_type=input_type)

    @pytest.mark.parametrize(
        "make_input",
        [make_image_tensor, make_image_pil, make_image, make_bounding_boxes, make_segmentation_mask, make_video],
    )
    def test_transform(self, make_input):
        check_transform(transforms.CenterCrop(self.OUTPUT_SIZES[0]), make_input(self.INPUT_SIZE))

    @pytest.mark.parametrize("output_size", OUTPUT_SIZES)
    @pytest.mark.parametrize("fn", [F.center_crop, transform_cls_to_functional(transforms.CenterCrop)])
    def test_image_correctness(self, output_size, fn):
        image = make_image(self.INPUT_SIZE, dtype=torch.uint8, device="cpu")

        actual = fn(image, output_size)
        expected = F.to_image(F.center_crop(F.to_pil_image(image), output_size=output_size))

        assert_equal(actual, expected)

    def _reference_center_crop_bounding_boxes(self, bounding_boxes, output_size):
        image_height, image_width = bounding_boxes.canvas_size
        if isinstance(output_size, int):
            output_size = (output_size, output_size)
        elif len(output_size) == 1:
            output_size *= 2
        crop_height, crop_width = output_size

        top = int(round((image_height - crop_height) / 2))
        left = int(round((image_width - crop_width) / 2))

        affine_matrix = np.array(
            [
                [1, 0, -left],
                [0, 1, -top],
            ],
        )
        return reference_affine_bounding_boxes_helper(
            bounding_boxes, affine_matrix=affine_matrix, new_canvas_size=output_size
        )

    @pytest.mark.parametrize("output_size", OUTPUT_SIZES)
    @pytest.mark.parametrize("format", list(tv_tensors.BoundingBoxFormat))
    @pytest.mark.parametrize("dtype", [torch.int64, torch.float32])
    @pytest.mark.parametrize("device", cpu_and_cuda())
    @pytest.mark.parametrize("fn", [F.center_crop, transform_cls_to_functional(transforms.CenterCrop)])
    def test_bounding_boxes_correctness(self, output_size, format, dtype, device, fn):
        bounding_boxes = make_bounding_boxes(self.INPUT_SIZE, format=format, dtype=dtype, device=device)

        actual = fn(bounding_boxes, output_size)
        expected = self._reference_center_crop_bounding_boxes(bounding_boxes, output_size)

        assert_equal(actual, expected)


class TestPerspective:
    COEFFICIENTS = [
        [1.2405, 0.1772, -6.9113, 0.0463, 1.251, -5.235, 0.00013, 0.0018],
        [0.7366, -0.11724, 1.45775, -0.15012, 0.73406, 2.6019, -0.0072, -0.0063],
    ]
    START_END_POINTS = [
        ([[0, 0], [33, 0], [33, 25], [0, 25]], [[3, 2], [32, 3], [30, 24], [2, 25]]),
        ([[3, 2], [32, 3], [30, 24], [2, 25]], [[0, 0], [33, 0], [33, 25], [0, 25]]),
        ([[3, 2], [32, 3], [30, 24], [2, 25]], [[5, 5], [30, 3], [33, 19], [4, 25]]),
    ]
    MINIMAL_KWARGS = dict(startpoints=None, endpoints=None, coefficients=COEFFICIENTS[0])

    @param_value_parametrization(
        coefficients=COEFFICIENTS,
        start_end_points=START_END_POINTS,
        fill=EXHAUSTIVE_TYPE_FILLS,
    )
    @pytest.mark.parametrize("dtype", [torch.uint8, torch.float32])
    @pytest.mark.parametrize("device", cpu_and_cuda())
    def test_kernel_image(self, param, value, dtype, device):
        if param == "start_end_points":
            kwargs = dict(zip(["startpoints", "endpoints"], value))
        else:
            kwargs = {"startpoints": None, "endpoints": None, param: value}
        if param == "fill":
            kwargs["coefficients"] = self.COEFFICIENTS[0]

        check_kernel(
            F.perspective_image,
            make_image(dtype=dtype, device=device),
            **kwargs,
            check_scripted_vs_eager=not (param == "fill" and isinstance(value, (int, float))),
        )

    def test_kernel_image_error(self):
        image = make_image_tensor()

        with pytest.raises(ValueError, match="startpoints/endpoints or the coefficients must have non `None` values"):
            F.perspective_image(image, startpoints=None, endpoints=None)

        with pytest.raises(
            ValueError, match="startpoints/endpoints and the coefficients shouldn't be defined concurrently"
        ):
            startpoints, endpoints = self.START_END_POINTS[0]
            coefficients = self.COEFFICIENTS[0]
            F.perspective_image(image, startpoints=startpoints, endpoints=endpoints, coefficients=coefficients)

        with pytest.raises(ValueError, match="coefficients should have 8 float values"):
            F.perspective_image(image, startpoints=None, endpoints=None, coefficients=list(range(7)))

    @param_value_parametrization(
        coefficients=COEFFICIENTS,
        start_end_points=START_END_POINTS,
    )
    @pytest.mark.parametrize("format", list(tv_tensors.BoundingBoxFormat))
    def test_kernel_bounding_boxes(self, param, value, format):
        if param == "start_end_points":
            kwargs = dict(zip(["startpoints", "endpoints"], value))
        else:
            kwargs = {"startpoints": None, "endpoints": None, param: value}

        bounding_boxes = make_bounding_boxes(format=format)

        check_kernel(
            F.perspective_bounding_boxes,
            bounding_boxes,
            format=bounding_boxes.format,
            canvas_size=bounding_boxes.canvas_size,
            **kwargs,
        )

    def test_kernel_bounding_boxes_error(self):
        bounding_boxes = make_bounding_boxes()
        format, canvas_size = bounding_boxes.format, bounding_boxes.canvas_size
        bounding_boxes = bounding_boxes.as_subclass(torch.Tensor)

        with pytest.raises(RuntimeError, match="Denominator is zero"):
            F.perspective_bounding_boxes(
                bounding_boxes,
                format=format,
                canvas_size=canvas_size,
                startpoints=None,
                endpoints=None,
                coefficients=[0.0] * 8,
            )

    @pytest.mark.parametrize("make_mask", [make_segmentation_mask, make_detection_masks])
    def test_kernel_mask(self, make_mask):
        check_kernel(F.perspective_mask, make_mask(), **self.MINIMAL_KWARGS)

    def test_kernel_video(self):
        check_kernel(F.perspective_video, make_video(), **self.MINIMAL_KWARGS)

    @pytest.mark.parametrize(
        "make_input",
        [make_image_tensor, make_image_pil, make_image, make_bounding_boxes, make_segmentation_mask, make_video],
    )
    def test_functional(self, make_input):
        check_functional(F.perspective, make_input(), **self.MINIMAL_KWARGS)

    @pytest.mark.parametrize(
        ("kernel", "input_type"),
        [
            (F.perspective_image, torch.Tensor),
            (F._perspective_image_pil, PIL.Image.Image),
            (F.perspective_image, tv_tensors.Image),
            (F.perspective_bounding_boxes, tv_tensors.BoundingBoxes),
            (F.perspective_mask, tv_tensors.Mask),
            (F.perspective_video, tv_tensors.Video),
        ],
    )
    def test_functional_signature(self, kernel, input_type):
        check_functional_kernel_signature_match(F.perspective, kernel=kernel, input_type=input_type)

    @pytest.mark.parametrize("distortion_scale", [0.5, 0.0, 1.0])
    @pytest.mark.parametrize(
        "make_input",
        [make_image_tensor, make_image_pil, make_image, make_bounding_boxes, make_segmentation_mask, make_video],
    )
    def test_transform(self, distortion_scale, make_input):
        check_transform(transforms.RandomPerspective(distortion_scale=distortion_scale, p=1), make_input())

    @pytest.mark.parametrize("distortion_scale", [-1, 2])
    def test_transform_error(self, distortion_scale):
        with pytest.raises(ValueError, match="distortion_scale value should be between 0 and 1"):
            transforms.RandomPerspective(distortion_scale=distortion_scale)

    @pytest.mark.parametrize("coefficients", COEFFICIENTS)
    @pytest.mark.parametrize(
        "interpolation", [transforms.InterpolationMode.NEAREST, transforms.InterpolationMode.BILINEAR]
    )
    @pytest.mark.parametrize("fill", CORRECTNESS_FILLS)
    def test_image_functional_correctness(self, coefficients, interpolation, fill):
        image = make_image(dtype=torch.uint8, device="cpu")

        actual = F.perspective(
            image, startpoints=None, endpoints=None, coefficients=coefficients, interpolation=interpolation, fill=fill
        )
        expected = F.to_image(
            F.perspective(
                F.to_pil_image(image),
                startpoints=None,
                endpoints=None,
                coefficients=coefficients,
                interpolation=interpolation,
                fill=fill,
            )
        )

        if interpolation is transforms.InterpolationMode.BILINEAR:
            abs_diff = (actual.float() - expected.float()).abs()
            assert (abs_diff > 1).float().mean() < 7e-2
            mae = abs_diff.mean()
            assert mae < 3
        else:
            assert_equal(actual, expected)

    def _reference_perspective_bounding_boxes(self, bounding_boxes, *, startpoints, endpoints):
        format = bounding_boxes.format
        canvas_size = bounding_boxes.canvas_size
        dtype = bounding_boxes.dtype
        device = bounding_boxes.device

        coefficients = _get_perspective_coeffs(endpoints, startpoints)

        def perspective_bounding_boxes(bounding_boxes):
            m1 = np.array(
                [
                    [coefficients[0], coefficients[1], coefficients[2]],
                    [coefficients[3], coefficients[4], coefficients[5]],
                ]
            )
            m2 = np.array(
                [
                    [coefficients[6], coefficients[7], 1.0],
                    [coefficients[6], coefficients[7], 1.0],
                ]
            )

            # Go to float before converting to prevent precision loss in case of CXCYWH -> XYXY and W or H is 1
            input_xyxy = F.convert_bounding_box_format(
                bounding_boxes.to(dtype=torch.float64, device="cpu", copy=True),
                old_format=format,
                new_format=tv_tensors.BoundingBoxFormat.XYXY,
                inplace=True,
            )
            x1, y1, x2, y2 = input_xyxy.squeeze(0).tolist()

            points = np.array(
                [
                    [x1, y1, 1.0],
                    [x2, y1, 1.0],
                    [x1, y2, 1.0],
                    [x2, y2, 1.0],
                ]
            )

            numerator = points @ m1.T
            denominator = points @ m2.T
            transformed_points = numerator / denominator

            output_xyxy = torch.Tensor(
                [
                    float(np.min(transformed_points[:, 0])),
                    float(np.min(transformed_points[:, 1])),
                    float(np.max(transformed_points[:, 0])),
                    float(np.max(transformed_points[:, 1])),
                ]
            )

            output = F.convert_bounding_box_format(
                output_xyxy, old_format=tv_tensors.BoundingBoxFormat.XYXY, new_format=format
            )

            # It is important to clamp before casting, especially for CXCYWH format, dtype=int64
            return F.clamp_bounding_boxes(
                output,
                format=format,
                canvas_size=canvas_size,
            ).to(dtype=dtype, device=device)

        return tv_tensors.BoundingBoxes(
            torch.cat([perspective_bounding_boxes(b) for b in bounding_boxes.reshape(-1, 4).unbind()], dim=0).reshape(
                bounding_boxes.shape
            ),
            format=format,
            canvas_size=canvas_size,
        )

    @pytest.mark.parametrize(("startpoints", "endpoints"), START_END_POINTS)
    @pytest.mark.parametrize("format", list(tv_tensors.BoundingBoxFormat))
    @pytest.mark.parametrize("dtype", [torch.int64, torch.float32])
    @pytest.mark.parametrize("device", cpu_and_cuda())
    def test_correctness_perspective_bounding_boxes(self, startpoints, endpoints, format, dtype, device):
        bounding_boxes = make_bounding_boxes(format=format, dtype=dtype, device=device)

        actual = F.perspective(bounding_boxes, startpoints=startpoints, endpoints=endpoints)
        expected = self._reference_perspective_bounding_boxes(
            bounding_boxes, startpoints=startpoints, endpoints=endpoints
        )

        assert_close(actual, expected, rtol=0, atol=1)


class TestEqualize:
    @pytest.mark.parametrize("dtype", [torch.uint8, torch.float32])
    @pytest.mark.parametrize("device", cpu_and_cuda())
    def test_kernel_image(self, dtype, device):
        check_kernel(F.equalize_image, make_image(dtype=dtype, device=device))

    def test_kernel_video(self):
        check_kernel(F.equalize_image, make_video())

    @pytest.mark.parametrize("make_input", [make_image_tensor, make_image_pil, make_image, make_video])
    def test_functional(self, make_input):
        check_functional(F.equalize, make_input())

    @pytest.mark.parametrize(
        ("kernel", "input_type"),
        [
            (F.equalize_image, torch.Tensor),
            (F._equalize_image_pil, PIL.Image.Image),
            (F.equalize_image, tv_tensors.Image),
            (F.equalize_video, tv_tensors.Video),
        ],
    )
    def test_functional_signature(self, kernel, input_type):
        check_functional_kernel_signature_match(F.equalize, kernel=kernel, input_type=input_type)

    @pytest.mark.parametrize(
        "make_input",
        [make_image_tensor, make_image_pil, make_image, make_video],
    )
    def test_transform(self, make_input):
        check_transform(transforms.RandomEqualize(p=1), make_input())

    @pytest.mark.parametrize(("low", "high"), [(0, 64), (64, 192), (192, 256), (0, 1), (127, 128), (255, 256)])
    @pytest.mark.parametrize("fn", [F.equalize, transform_cls_to_functional(transforms.RandomEqualize, p=1)])
    def test_image_correctness(self, low, high, fn):
        # We are not using the default `make_image` here since that uniformly samples the values over the whole value
        # range. Since the whole point of F.equalize is to transform an arbitrary distribution of values into a uniform
        # one over the full range, the information gain is low if we already provide something really close to the
        # expected value.
        image = tv_tensors.Image(
            torch.testing.make_tensor((3, 117, 253), dtype=torch.uint8, device="cpu", low=low, high=high)
        )

        actual = fn(image)
        expected = F.to_image(F.equalize(F.to_pil_image(image)))

        assert_equal(actual, expected)


class TestUniformTemporalSubsample:
    def test_kernel_video(self):
        check_kernel(F.uniform_temporal_subsample_video, make_video(), num_samples=2)

    @pytest.mark.parametrize("make_input", [make_video_tensor, make_video])
    def test_functional(self, make_input):
        check_functional(F.uniform_temporal_subsample, make_input(), num_samples=2)

    @pytest.mark.parametrize(
        ("kernel", "input_type"),
        [
            (F.uniform_temporal_subsample_video, torch.Tensor),
            (F.uniform_temporal_subsample_video, tv_tensors.Video),
        ],
    )
    def test_functional_signature(self, kernel, input_type):
        check_functional_kernel_signature_match(F.uniform_temporal_subsample, kernel=kernel, input_type=input_type)

    @pytest.mark.parametrize("make_input", [make_video_tensor, make_video])
    def test_transform(self, make_input):
        check_transform(transforms.UniformTemporalSubsample(num_samples=2), make_input())

    def _reference_uniform_temporal_subsample_video(self, video, *, num_samples):
        # Adapted from
        # https://github.com/facebookresearch/pytorchvideo/blob/c8d23d8b7e597586a9e2d18f6ed31ad8aa379a7a/pytorchvideo/transforms/functional.py#L19
        t = video.shape[-4]
        assert num_samples > 0 and t > 0
        # Sample by nearest neighbor interpolation if num_samples > t.
        indices = torch.linspace(0, t - 1, num_samples, device=video.device)
        indices = torch.clamp(indices, 0, t - 1).long()
        return tv_tensors.Video(torch.index_select(video, -4, indices))

    CORRECTNESS_NUM_FRAMES = 5

    @pytest.mark.parametrize("num_samples", list(range(1, CORRECTNESS_NUM_FRAMES + 1)))
    @pytest.mark.parametrize("dtype", [torch.uint8, torch.float32])
    @pytest.mark.parametrize("device", cpu_and_cuda())
    @pytest.mark.parametrize(
        "fn", [F.uniform_temporal_subsample, transform_cls_to_functional(transforms.UniformTemporalSubsample)]
    )
    def test_video_correctness(self, num_samples, dtype, device, fn):
        video = make_video(num_frames=self.CORRECTNESS_NUM_FRAMES, dtype=dtype, device=device)

        actual = fn(video, num_samples=num_samples)
        expected = self._reference_uniform_temporal_subsample_video(video, num_samples=num_samples)

        assert_equal(actual, expected)


class TestNormalize:
    MEANS_STDS = [
        ((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        ([0.0, 0.0, 0.0], [1.0, 1.0, 1.0]),
    ]
    MEAN, STD = MEANS_STDS[0]

    @pytest.mark.parametrize(("mean", "std"), [*MEANS_STDS, (0.5, 2.0)])
    @pytest.mark.parametrize("device", cpu_and_cuda())
    def test_kernel_image(self, mean, std, device):
        check_kernel(F.normalize_image, make_image(dtype=torch.float32, device=device), mean=self.MEAN, std=self.STD)

    @pytest.mark.parametrize("device", cpu_and_cuda())
    def test_kernel_image_inplace(self, device):
        input = make_image_tensor(dtype=torch.float32, device=device)
        input_version = input._version

        output_out_of_place = F.normalize_image(input, mean=self.MEAN, std=self.STD)
        assert output_out_of_place.data_ptr() != input.data_ptr()
        assert output_out_of_place is not input

        output_inplace = F.normalize_image(input, mean=self.MEAN, std=self.STD, inplace=True)
        assert output_inplace.data_ptr() == input.data_ptr()
        assert output_inplace._version > input_version
        assert output_inplace is input

        assert_equal(output_inplace, output_out_of_place)

    def test_kernel_video(self):
        check_kernel(F.normalize_video, make_video(dtype=torch.float32), mean=self.MEAN, std=self.STD)

    @pytest.mark.parametrize("make_input", [make_image_tensor, make_image, make_video])
    def test_functional(self, make_input):
        check_functional(F.normalize, make_input(dtype=torch.float32), mean=self.MEAN, std=self.STD)

    @pytest.mark.parametrize(
        ("kernel", "input_type"),
        [
            (F.normalize_image, torch.Tensor),
            (F.normalize_image, tv_tensors.Image),
            (F.normalize_video, tv_tensors.Video),
        ],
    )
    def test_functional_signature(self, kernel, input_type):
        check_functional_kernel_signature_match(F.normalize, kernel=kernel, input_type=input_type)

    def test_functional_error(self):
        with pytest.raises(TypeError, match="should be a float tensor"):
            F.normalize_image(make_image(dtype=torch.uint8), mean=self.MEAN, std=self.STD)

        with pytest.raises(ValueError, match="tensor image of size"):
            F.normalize_image(torch.rand(16, 16, dtype=torch.float32), mean=self.MEAN, std=self.STD)

        for std in [0, [0, 0, 0], [0, 1, 1]]:
            with pytest.raises(ValueError, match="std evaluated to zero, leading to division by zero"):
                F.normalize_image(make_image(dtype=torch.float32), mean=self.MEAN, std=std)

    def _sample_input_adapter(self, transform, input, device):
        adapted_input = {}
        for key, value in input.items():
            if isinstance(value, PIL.Image.Image):
                # normalize doesn't support PIL images
                continue
            elif check_type(value, (is_pure_tensor, tv_tensors.Image, tv_tensors.Video)):
                # normalize doesn't support integer images
                value = F.to_dtype(value, torch.float32, scale=True)
            adapted_input[key] = value
        return adapted_input

    @pytest.mark.parametrize("make_input", [make_image_tensor, make_image, make_video])
    def test_transform(self, make_input):
        check_transform(
            transforms.Normalize(mean=self.MEAN, std=self.STD),
            make_input(dtype=torch.float32),
            check_sample_input=self._sample_input_adapter,
        )

    def _reference_normalize_image(self, image, *, mean, std):
        image = image.numpy()
        mean, std = [np.array(stat, dtype=image.dtype).reshape((-1, 1, 1)) for stat in [mean, std]]
        return tv_tensors.Image((image - mean) / std)

    @pytest.mark.parametrize(("mean", "std"), MEANS_STDS)
    @pytest.mark.parametrize("dtype", [torch.float16, torch.float32, torch.float64])
    @pytest.mark.parametrize("fn", [F.normalize, transform_cls_to_functional(transforms.Normalize)])
    def test_correctness_image(self, mean, std, dtype, fn):
        image = make_image(dtype=dtype)

        actual = fn(image, mean=mean, std=std)
        expected = self._reference_normalize_image(image, mean=mean, std=std)

        assert_equal(actual, expected)


class TestClampBoundingBoxes:
    @pytest.mark.parametrize("format", list(tv_tensors.BoundingBoxFormat))
    @pytest.mark.parametrize("dtype", [torch.int64, torch.float32])
    @pytest.mark.parametrize("device", cpu_and_cuda())
    def test_kernel(self, format, dtype, device):
        bounding_boxes = make_bounding_boxes(format=format, dtype=dtype, device=device)
        check_kernel(
            F.clamp_bounding_boxes,
            bounding_boxes,
            format=bounding_boxes.format,
            canvas_size=bounding_boxes.canvas_size,
        )

    @pytest.mark.parametrize("format", list(tv_tensors.BoundingBoxFormat))
    def test_functional(self, format):
        check_functional(F.clamp_bounding_boxes, make_bounding_boxes(format=format))

    def test_errors(self):
        input_tv_tensor = make_bounding_boxes()
        input_pure_tensor = input_tv_tensor.as_subclass(torch.Tensor)
        format, canvas_size = input_tv_tensor.format, input_tv_tensor.canvas_size

        for format_, canvas_size_ in [(None, None), (format, None), (None, canvas_size)]:
            with pytest.raises(
                ValueError, match="For pure tensor inputs, `format` and `canvas_size` have to be passed."
            ):
                F.clamp_bounding_boxes(input_pure_tensor, format=format_, canvas_size=canvas_size_)

        for format_, canvas_size_ in [(format, canvas_size), (format, None), (None, canvas_size)]:
            with pytest.raises(
                ValueError, match="For bounding box tv_tensor inputs, `format` and `canvas_size` must not be passed."
            ):
                F.clamp_bounding_boxes(input_tv_tensor, format=format_, canvas_size=canvas_size_)

    def test_transform(self):
        check_transform(transforms.ClampBoundingBoxes(), make_bounding_boxes())


class TestInvert:
    @pytest.mark.parametrize("dtype", [torch.uint8, torch.int16, torch.float32])
    @pytest.mark.parametrize("device", cpu_and_cuda())
    def test_kernel_image(self, dtype, device):
        check_kernel(F.invert_image, make_image(dtype=dtype, device=device))

    def test_kernel_video(self):
        check_kernel(F.invert_video, make_video())

    @pytest.mark.parametrize("make_input", [make_image_tensor, make_image, make_image_pil, make_video])
    def test_functional(self, make_input):
        check_functional(F.invert, make_input())

    @pytest.mark.parametrize(
        ("kernel", "input_type"),
        [
            (F.invert_image, torch.Tensor),
            (F._invert_image_pil, PIL.Image.Image),
            (F.invert_image, tv_tensors.Image),
            (F.invert_video, tv_tensors.Video),
        ],
    )
    def test_functional_signature(self, kernel, input_type):
        check_functional_kernel_signature_match(F.invert, kernel=kernel, input_type=input_type)

    @pytest.mark.parametrize("make_input", [make_image_tensor, make_image_pil, make_image, make_video])
    def test_transform(self, make_input):
        check_transform(transforms.RandomInvert(p=1), make_input())

    @pytest.mark.parametrize("fn", [F.invert, transform_cls_to_functional(transforms.RandomInvert, p=1)])
    def test_correctness_image(self, fn):
        image = make_image(dtype=torch.uint8, device="cpu")

        actual = fn(image)
        expected = F.to_image(F.invert(F.to_pil_image(image)))

        assert_equal(actual, expected)


class TestPosterize:
    @pytest.mark.parametrize("dtype", [torch.uint8, torch.float32])
    @pytest.mark.parametrize("device", cpu_and_cuda())
    def test_kernel_image(self, dtype, device):
        check_kernel(F.posterize_image, make_image(dtype=dtype, device=device), bits=1)

    def test_kernel_video(self):
        check_kernel(F.posterize_video, make_video(), bits=1)

    @pytest.mark.parametrize("make_input", [make_image_tensor, make_image, make_image_pil, make_video])
    def test_functional(self, make_input):
        check_functional(F.posterize, make_input(), bits=1)

    @pytest.mark.parametrize(
        ("kernel", "input_type"),
        [
            (F.posterize_image, torch.Tensor),
            (F._posterize_image_pil, PIL.Image.Image),
            (F.posterize_image, tv_tensors.Image),
            (F.posterize_video, tv_tensors.Video),
        ],
    )
    def test_functional_signature(self, kernel, input_type):
        check_functional_kernel_signature_match(F.posterize, kernel=kernel, input_type=input_type)

    @pytest.mark.parametrize("make_input", [make_image_tensor, make_image_pil, make_image, make_video])
    def test_transform(self, make_input):
        check_transform(transforms.RandomPosterize(bits=1, p=1), make_input())

    @pytest.mark.parametrize("bits", [1, 4, 8])
    @pytest.mark.parametrize("fn", [F.posterize, transform_cls_to_functional(transforms.RandomPosterize, p=1)])
    def test_correctness_image(self, bits, fn):
        image = make_image(dtype=torch.uint8, device="cpu")

        actual = fn(image, bits=bits)
        expected = F.to_image(F.posterize(F.to_pil_image(image), bits=bits))

        assert_equal(actual, expected)


class TestSolarize:
    def _make_threshold(self, input, *, factor=0.5):
        dtype = input.dtype if isinstance(input, torch.Tensor) else torch.uint8
        return (float if dtype.is_floating_point else int)(get_max_value(dtype) * factor)

    @pytest.mark.parametrize("dtype", [torch.uint8, torch.float32])
    @pytest.mark.parametrize("device", cpu_and_cuda())
    def test_kernel_image(self, dtype, device):
        image = make_image(dtype=dtype, device=device)
        check_kernel(F.solarize_image, image, threshold=self._make_threshold(image))

    def test_kernel_video(self):
        video = make_video()
        check_kernel(F.solarize_video, video, threshold=self._make_threshold(video))

    @pytest.mark.parametrize("make_input", [make_image_tensor, make_image, make_image_pil, make_video])
    def test_functional(self, make_input):
        input = make_input()
        check_functional(F.solarize, input, threshold=self._make_threshold(input))

    @pytest.mark.parametrize(
        ("kernel", "input_type"),
        [
            (F.solarize_image, torch.Tensor),
            (F._solarize_image_pil, PIL.Image.Image),
            (F.solarize_image, tv_tensors.Image),
            (F.solarize_video, tv_tensors.Video),
        ],
    )
    def test_functional_signature(self, kernel, input_type):
        check_functional_kernel_signature_match(F.solarize, kernel=kernel, input_type=input_type)

    @pytest.mark.parametrize(("dtype", "threshold"), [(torch.uint8, 256), (torch.float, 1.5)])
    def test_functional_error(self, dtype, threshold):
        with pytest.raises(TypeError, match="Threshold should be less or equal the maximum value of the dtype"):
            F.solarize(make_image(dtype=dtype), threshold=threshold)

    @pytest.mark.parametrize("make_input", [make_image_tensor, make_image_pil, make_image, make_video])
    def test_transform(self, make_input):
        input = make_input()
        check_transform(transforms.RandomSolarize(threshold=self._make_threshold(input), p=1), input)

    @pytest.mark.parametrize("threshold_factor", [0.0, 0.1, 0.5, 0.9, 1.0])
    @pytest.mark.parametrize("fn", [F.solarize, transform_cls_to_functional(transforms.RandomSolarize, p=1)])
    def test_correctness_image(self, threshold_factor, fn):
        image = make_image(dtype=torch.uint8, device="cpu")
        threshold = self._make_threshold(image, factor=threshold_factor)

        actual = fn(image, threshold=threshold)
        expected = F.to_image(F.solarize(F.to_pil_image(image), threshold=threshold))

        assert_equal(actual, expected)


class TestAutocontrast:
    @pytest.mark.parametrize("dtype", [torch.uint8, torch.int16, torch.float32])
    @pytest.mark.parametrize("device", cpu_and_cuda())
    def test_kernel_image(self, dtype, device):
        check_kernel(F.autocontrast_image, make_image(dtype=dtype, device=device))

    def test_kernel_video(self):
        check_kernel(F.autocontrast_video, make_video())

    @pytest.mark.parametrize("make_input", [make_image_tensor, make_image, make_image_pil, make_video])
    def test_functional(self, make_input):
        check_functional(F.autocontrast, make_input())

    @pytest.mark.parametrize(
        ("kernel", "input_type"),
        [
            (F.autocontrast_image, torch.Tensor),
            (F._autocontrast_image_pil, PIL.Image.Image),
            (F.autocontrast_image, tv_tensors.Image),
            (F.autocontrast_video, tv_tensors.Video),
        ],
    )
    def test_functional_signature(self, kernel, input_type):
        check_functional_kernel_signature_match(F.autocontrast, kernel=kernel, input_type=input_type)

    @pytest.mark.parametrize("make_input", [make_image_tensor, make_image_pil, make_image, make_video])
    def test_transform(self, make_input):
        check_transform(transforms.RandomAutocontrast(p=1), make_input(), check_v1_compatibility=dict(rtol=0, atol=1))

    @pytest.mark.parametrize("fn", [F.autocontrast, transform_cls_to_functional(transforms.RandomAutocontrast, p=1)])
    def test_correctness_image(self, fn):
        image = make_image(dtype=torch.uint8, device="cpu")

        actual = fn(image)
        expected = F.to_image(F.autocontrast(F.to_pil_image(image)))

        assert_close(actual, expected, rtol=0, atol=1)


class TestAdjustSharpness:
    @pytest.mark.parametrize("dtype", [torch.uint8, torch.float32])
    @pytest.mark.parametrize("device", cpu_and_cuda())
    def test_kernel_image(self, dtype, device):
        check_kernel(F.adjust_sharpness_image, make_image(dtype=dtype, device=device), sharpness_factor=0.5)

    def test_kernel_video(self):
        check_kernel(F.adjust_sharpness_video, make_video(), sharpness_factor=0.5)

    @pytest.mark.parametrize("make_input", [make_image_tensor, make_image, make_image_pil, make_video])
    def test_functional(self, make_input):
        check_functional(F.adjust_sharpness, make_input(), sharpness_factor=0.5)

    @pytest.mark.parametrize(
        ("kernel", "input_type"),
        [
            (F.adjust_sharpness_image, torch.Tensor),
            (F._adjust_sharpness_image_pil, PIL.Image.Image),
            (F.adjust_sharpness_image, tv_tensors.Image),
            (F.adjust_sharpness_video, tv_tensors.Video),
        ],
    )
    def test_functional_signature(self, kernel, input_type):
        check_functional_kernel_signature_match(F.adjust_sharpness, kernel=kernel, input_type=input_type)

    @pytest.mark.parametrize("make_input", [make_image_tensor, make_image_pil, make_image, make_video])
    def test_transform(self, make_input):
        check_transform(transforms.RandomAdjustSharpness(sharpness_factor=0.5, p=1), make_input())

    def test_functional_error(self):
        with pytest.raises(TypeError, match="can have 1 or 3 channels"):
            F.adjust_sharpness(make_image(color_space="RGBA"), sharpness_factor=0.5)

        with pytest.raises(ValueError, match="is not non-negative"):
            F.adjust_sharpness(make_image(), sharpness_factor=-1)

    @pytest.mark.parametrize("sharpness_factor", [0.1, 0.5, 1.0])
    @pytest.mark.parametrize(
        "fn", [F.adjust_sharpness, transform_cls_to_functional(transforms.RandomAdjustSharpness, p=1)]
    )
    def test_correctness_image(self, sharpness_factor, fn):
        image = make_image(dtype=torch.uint8, device="cpu")

        actual = fn(image, sharpness_factor=sharpness_factor)
        expected = F.to_image(F.adjust_sharpness(F.to_pil_image(image), sharpness_factor=sharpness_factor))

        assert_equal(actual, expected)


class TestAdjustContrast:
    @pytest.mark.parametrize("dtype", [torch.uint8, torch.float32])
    @pytest.mark.parametrize("device", cpu_and_cuda())
    def test_kernel_image(self, dtype, device):
        check_kernel(F.adjust_contrast_image, make_image(dtype=dtype, device=device), contrast_factor=0.5)

    def test_kernel_video(self):
        check_kernel(F.adjust_contrast_video, make_video(), contrast_factor=0.5)

    @pytest.mark.parametrize("make_input", [make_image_tensor, make_image, make_image_pil, make_video])
    def test_functional(self, make_input):
        check_functional(F.adjust_contrast, make_input(), contrast_factor=0.5)

    @pytest.mark.parametrize(
        ("kernel", "input_type"),
        [
            (F.adjust_contrast_image, torch.Tensor),
            (F._adjust_contrast_image_pil, PIL.Image.Image),
            (F.adjust_contrast_image, tv_tensors.Image),
            (F.adjust_contrast_video, tv_tensors.Video),
        ],
    )
    def test_functional_signature(self, kernel, input_type):
        check_functional_kernel_signature_match(F.adjust_contrast, kernel=kernel, input_type=input_type)

    def test_functional_error(self):
        with pytest.raises(TypeError, match="permitted channel values are 1 or 3"):
            F.adjust_contrast(make_image(color_space="RGBA"), contrast_factor=0.5)

        with pytest.raises(ValueError, match="is not non-negative"):
            F.adjust_contrast(make_image(), contrast_factor=-1)

    @pytest.mark.parametrize("contrast_factor", [0.1, 0.5, 1.0])
    def test_correctness_image(self, contrast_factor):
        image = make_image(dtype=torch.uint8, device="cpu")

        actual = F.adjust_contrast(image, contrast_factor=contrast_factor)
        expected = F.to_image(F.adjust_contrast(F.to_pil_image(image), contrast_factor=contrast_factor))

        assert_close(actual, expected, rtol=0, atol=1)


class TestAdjustGamma:
    @pytest.mark.parametrize("dtype", [torch.uint8, torch.float32])
    @pytest.mark.parametrize("device", cpu_and_cuda())
    def test_kernel_image(self, dtype, device):
        check_kernel(F.adjust_gamma_image, make_image(dtype=dtype, device=device), gamma=0.5)

    def test_kernel_video(self):
        check_kernel(F.adjust_gamma_video, make_video(), gamma=0.5)

    @pytest.mark.parametrize("make_input", [make_image_tensor, make_image, make_image_pil, make_video])
    def test_functional(self, make_input):
        check_functional(F.adjust_gamma, make_input(), gamma=0.5)

    @pytest.mark.parametrize(
        ("kernel", "input_type"),
        [
            (F.adjust_gamma_image, torch.Tensor),
            (F._adjust_gamma_image_pil, PIL.Image.Image),
            (F.adjust_gamma_image, tv_tensors.Image),
            (F.adjust_gamma_video, tv_tensors.Video),
        ],
    )
    def test_functional_signature(self, kernel, input_type):
        check_functional_kernel_signature_match(F.adjust_gamma, kernel=kernel, input_type=input_type)

    def test_functional_error(self):
        with pytest.raises(ValueError, match="Gamma should be a non-negative real number"):
            F.adjust_gamma(make_image(), gamma=-1)

    @pytest.mark.parametrize("gamma", [0.1, 0.5, 1.0])
    @pytest.mark.parametrize("gain", [0.1, 1.0, 2.0])
    def test_correctness_image(self, gamma, gain):
        image = make_image(dtype=torch.uint8, device="cpu")

        actual = F.adjust_gamma(image, gamma=gamma, gain=gain)
        expected = F.to_image(F.adjust_gamma(F.to_pil_image(image), gamma=gamma, gain=gain))

        assert_equal(actual, expected)


class TestAdjustHue:
    @pytest.mark.parametrize("dtype", [torch.uint8, torch.float32])
    @pytest.mark.parametrize("device", cpu_and_cuda())
    def test_kernel_image(self, dtype, device):
        check_kernel(F.adjust_hue_image, make_image(dtype=dtype, device=device), hue_factor=0.25)

    def test_kernel_video(self):
        check_kernel(F.adjust_hue_video, make_video(), hue_factor=0.25)

    @pytest.mark.parametrize("make_input", [make_image_tensor, make_image, make_image_pil, make_video])
    def test_functional(self, make_input):
        check_functional(F.adjust_hue, make_input(), hue_factor=0.25)

    @pytest.mark.parametrize(
        ("kernel", "input_type"),
        [
            (F.adjust_hue_image, torch.Tensor),
            (F._adjust_hue_image_pil, PIL.Image.Image),
            (F.adjust_hue_image, tv_tensors.Image),
            (F.adjust_hue_video, tv_tensors.Video),
        ],
    )
    def test_functional_signature(self, kernel, input_type):
        check_functional_kernel_signature_match(F.adjust_hue, kernel=kernel, input_type=input_type)

    def test_functional_error(self):
        with pytest.raises(TypeError, match="permitted channel values are 1 or 3"):
            F.adjust_hue(make_image(color_space="RGBA"), hue_factor=0.25)

        for hue_factor in [-1, 1]:
            with pytest.raises(ValueError, match=re.escape("is not in [-0.5, 0.5]")):
                F.adjust_hue(make_image(), hue_factor=hue_factor)

    @pytest.mark.parametrize("hue_factor", [-0.5, -0.3, 0.0, 0.2, 0.5])
    def test_correctness_image(self, hue_factor):
        image = make_image(dtype=torch.uint8, device="cpu")

        actual = F.adjust_hue(image, hue_factor=hue_factor)
        expected = F.to_image(F.adjust_hue(F.to_pil_image(image), hue_factor=hue_factor))

        mae = (actual.float() - expected.float()).abs().mean()
        assert mae < 2


class TestAdjustSaturation:
    @pytest.mark.parametrize("dtype", [torch.uint8, torch.float32])
    @pytest.mark.parametrize("device", cpu_and_cuda())
    def test_kernel_image(self, dtype, device):
        check_kernel(F.adjust_saturation_image, make_image(dtype=dtype, device=device), saturation_factor=0.5)

    def test_kernel_video(self):
        check_kernel(F.adjust_saturation_video, make_video(), saturation_factor=0.5)

    @pytest.mark.parametrize("make_input", [make_image_tensor, make_image, make_image_pil, make_video])
    def test_functional(self, make_input):
        check_functional(F.adjust_saturation, make_input(), saturation_factor=0.5)

    @pytest.mark.parametrize(
        ("kernel", "input_type"),
        [
            (F.adjust_saturation_image, torch.Tensor),
            (F._adjust_saturation_image_pil, PIL.Image.Image),
            (F.adjust_saturation_image, tv_tensors.Image),
            (F.adjust_saturation_video, tv_tensors.Video),
        ],
    )
    def test_functional_signature(self, kernel, input_type):
        check_functional_kernel_signature_match(F.adjust_saturation, kernel=kernel, input_type=input_type)

    def test_functional_error(self):
        with pytest.raises(TypeError, match="permitted channel values are 1 or 3"):
            F.adjust_saturation(make_image(color_space="RGBA"), saturation_factor=0.5)

        with pytest.raises(ValueError, match="is not non-negative"):
            F.adjust_saturation(make_image(), saturation_factor=-1)

    @pytest.mark.parametrize("saturation_factor", [0.1, 0.5, 1.0])
    def test_correctness_image(self, saturation_factor):
        image = make_image(dtype=torch.uint8, device="cpu")

        actual = F.adjust_saturation(image, saturation_factor=saturation_factor)
        expected = F.to_image(F.adjust_saturation(F.to_pil_image(image), saturation_factor=saturation_factor))

        assert_close(actual, expected, rtol=0, atol=1)


class TestFiveTenCrop:
    INPUT_SIZE = (17, 11)
    OUTPUT_SIZE = (3, 5)

    @pytest.mark.parametrize("dtype", [torch.uint8, torch.float32])
    @pytest.mark.parametrize("device", cpu_and_cuda())
    @pytest.mark.parametrize("kernel", [F.five_crop_image, F.ten_crop_image])
    def test_kernel_image(self, dtype, device, kernel):
        check_kernel(
            kernel,
            make_image(self.INPUT_SIZE, dtype=dtype, device=device),
            size=self.OUTPUT_SIZE,
            check_batched_vs_unbatched=False,
        )

    @pytest.mark.parametrize("kernel", [F.five_crop_video, F.ten_crop_video])
    def test_kernel_video(self, kernel):
        check_kernel(kernel, make_video(self.INPUT_SIZE), size=self.OUTPUT_SIZE, check_batched_vs_unbatched=False)

    def _functional_wrapper(self, fn):
        # This wrapper is needed to make five_crop / ten_crop compatible with check_functional, since that requires a
        # single output rather than a sequence.
        @functools.wraps(fn)
        def wrapper(*args, **kwargs):
            outputs = fn(*args, **kwargs)
            return outputs[0]

        return wrapper

    @pytest.mark.parametrize(
        "make_input",
        [make_image_tensor, make_image_pil, make_image, make_video],
    )
    @pytest.mark.parametrize("functional", [F.five_crop, F.ten_crop])
    def test_functional(self, make_input, functional):
        check_functional(
            self._functional_wrapper(functional),
            make_input(self.INPUT_SIZE),
            size=self.OUTPUT_SIZE,
            check_scripted_smoke=False,
        )

    @pytest.mark.parametrize(
        ("functional", "kernel", "input_type"),
        [
            (F.five_crop, F.five_crop_image, torch.Tensor),
            (F.five_crop, F._five_crop_image_pil, PIL.Image.Image),
            (F.five_crop, F.five_crop_image, tv_tensors.Image),
            (F.five_crop, F.five_crop_video, tv_tensors.Video),
            (F.ten_crop, F.ten_crop_image, torch.Tensor),
            (F.ten_crop, F._ten_crop_image_pil, PIL.Image.Image),
            (F.ten_crop, F.ten_crop_image, tv_tensors.Image),
            (F.ten_crop, F.ten_crop_video, tv_tensors.Video),
        ],
    )
    def test_functional_signature(self, functional, kernel, input_type):
        check_functional_kernel_signature_match(functional, kernel=kernel, input_type=input_type)

    class _TransformWrapper(nn.Module):
        # This wrapper is needed to make FiveCrop / TenCrop compatible with check_transform, since that requires a
        # single output rather than a sequence.
        _v1_transform_cls = None

        def _extract_params_for_v1_transform(self):
            return dict(five_ten_crop_transform=self.five_ten_crop_transform)

        def __init__(self, five_ten_crop_transform):
            super().__init__()
            type(self)._v1_transform_cls = type(self)
            self.five_ten_crop_transform = five_ten_crop_transform

        def forward(self, input: torch.Tensor) -> torch.Tensor:
            outputs = self.five_ten_crop_transform(input)
            return outputs[0]

    @pytest.mark.parametrize(
        "make_input",
        [make_image_tensor, make_image_pil, make_image, make_video],
    )
    @pytest.mark.parametrize("transform_cls", [transforms.FiveCrop, transforms.TenCrop])
    def test_transform(self, make_input, transform_cls):
        check_transform(
            self._TransformWrapper(transform_cls(size=self.OUTPUT_SIZE)),
            make_input(self.INPUT_SIZE),
            check_sample_input=False,
        )

    @pytest.mark.parametrize("make_input", [make_bounding_boxes, make_detection_masks])
    @pytest.mark.parametrize("transform_cls", [transforms.FiveCrop, transforms.TenCrop])
    def test_transform_error(self, make_input, transform_cls):
        transform = transform_cls(size=self.OUTPUT_SIZE)

        with pytest.raises(TypeError, match="not supported"):
            transform(make_input(self.INPUT_SIZE))

    @pytest.mark.parametrize("fn", [F.five_crop, transform_cls_to_functional(transforms.FiveCrop)])
    def test_correctness_image_five_crop(self, fn):
        image = make_image(self.INPUT_SIZE, dtype=torch.uint8, device="cpu")

        actual = fn(image, size=self.OUTPUT_SIZE)
        expected = F.five_crop(F.to_pil_image(image), size=self.OUTPUT_SIZE)

        assert isinstance(actual, tuple)
        assert_equal(actual, [F.to_image(e) for e in expected])

    @pytest.mark.parametrize("fn_or_class", [F.ten_crop, transforms.TenCrop])
    @pytest.mark.parametrize("vertical_flip", [False, True])
    def test_correctness_image_ten_crop(self, fn_or_class, vertical_flip):
        if fn_or_class is transforms.TenCrop:
            fn = transform_cls_to_functional(fn_or_class, size=self.OUTPUT_SIZE, vertical_flip=vertical_flip)
            kwargs = dict()
        else:
            fn = fn_or_class
            kwargs = dict(size=self.OUTPUT_SIZE, vertical_flip=vertical_flip)

        image = make_image(self.INPUT_SIZE, dtype=torch.uint8, device="cpu")

        actual = fn(image, **kwargs)
        expected = F.ten_crop(F.to_pil_image(image), size=self.OUTPUT_SIZE, vertical_flip=vertical_flip)

        assert isinstance(actual, tuple)
        assert_equal(actual, [F.to_image(e) for e in expected])


class TestColorJitter:
    @pytest.mark.parametrize(
        "make_input",
        [make_image_tensor, make_image_pil, make_image, make_video],
    )
    @pytest.mark.parametrize("dtype", [torch.uint8, torch.float32])
    @pytest.mark.parametrize("device", cpu_and_cuda())
    def test_transform(self, make_input, dtype, device):
        if make_input is make_image_pil and not (dtype is torch.uint8 and device == "cpu"):
            pytest.skip(
                "PIL image tests with parametrization other than dtype=torch.uint8 and device='cpu' "
                "will degenerate to that anyway."
            )

        check_transform(
            transforms.ColorJitter(brightness=0.5, contrast=0.5, saturation=0.5, hue=0.25),
            make_input(dtype=dtype, device=device),
        )

    def test_transform_noop(self):
        input = make_image()
        input_version = input._version

        transform = transforms.ColorJitter()
        output = transform(input)

        assert output is input
        assert output.data_ptr() == input.data_ptr()
        assert output._version == input_version

    def test_transform_error(self):
        with pytest.raises(ValueError, match="must be non negative"):
            transforms.ColorJitter(brightness=-1)

        for brightness in [object(), [1, 2, 3]]:
            with pytest.raises(TypeError, match="single number or a sequence with length 2"):
                transforms.ColorJitter(brightness=brightness)

        with pytest.raises(ValueError, match="values should be between"):
            transforms.ColorJitter(brightness=(-1, 0.5))

        with pytest.raises(ValueError, match="values should be between"):
            transforms.ColorJitter(hue=1)

    @pytest.mark.parametrize("brightness", [None, 0.1, (0.2, 0.3)])
    @pytest.mark.parametrize("contrast", [None, 0.4, (0.5, 0.6)])
    @pytest.mark.parametrize("saturation", [None, 0.7, (0.8, 0.9)])
    @pytest.mark.parametrize("hue", [None, 0.3, (-0.1, 0.2)])
    def test_transform_correctness(self, brightness, contrast, saturation, hue):
        image = make_image(dtype=torch.uint8, device="cpu")

        transform = transforms.ColorJitter(brightness=brightness, contrast=contrast, saturation=saturation, hue=hue)

        with freeze_rng_state():
            torch.manual_seed(0)
            actual = transform(image)

            torch.manual_seed(0)
            expected = F.to_image(transform(F.to_pil_image(image)))

        mae = (actual.float() - expected.float()).abs().mean()
        assert mae < 2


class TestRgbToGrayscale:
    @pytest.mark.parametrize("dtype", [torch.uint8, torch.float32])
    @pytest.mark.parametrize("device", cpu_and_cuda())
    def test_kernel_image(self, dtype, device):
        check_kernel(F.rgb_to_grayscale_image, make_image(dtype=dtype, device=device))

    @pytest.mark.parametrize("make_input", [make_image_tensor, make_image_pil, make_image])
    def test_functional(self, make_input):
        check_functional(F.rgb_to_grayscale, make_input())

    @pytest.mark.parametrize(
        ("kernel", "input_type"),
        [
            (F.rgb_to_grayscale_image, torch.Tensor),
            (F._rgb_to_grayscale_image_pil, PIL.Image.Image),
            (F.rgb_to_grayscale_image, tv_tensors.Image),
        ],
    )
    def test_functional_signature(self, kernel, input_type):
        check_functional_kernel_signature_match(F.rgb_to_grayscale, kernel=kernel, input_type=input_type)

    @pytest.mark.parametrize("transform", [transforms.Grayscale(), transforms.RandomGrayscale(p=1)])
    @pytest.mark.parametrize("make_input", [make_image_tensor, make_image_pil, make_image])
    def test_transform(self, transform, make_input):
        check_transform(transform, make_input())

    @pytest.mark.parametrize("num_output_channels", [1, 3])
    @pytest.mark.parametrize("fn", [F.rgb_to_grayscale, transform_cls_to_functional(transforms.Grayscale)])
    def test_image_correctness(self, num_output_channels, fn):
        image = make_image(dtype=torch.uint8, device="cpu")

        actual = fn(image, num_output_channels=num_output_channels)
        expected = F.to_image(F.rgb_to_grayscale(F.to_pil_image(image), num_output_channels=num_output_channels))

        assert_equal(actual, expected, rtol=0, atol=1)

    @pytest.mark.parametrize("num_input_channels", [1, 3])
    def test_random_transform_correctness(self, num_input_channels):
        image = make_image(
            color_space={
                1: "GRAY",
                3: "RGB",
            }[num_input_channels],
            dtype=torch.uint8,
            device="cpu",
        )

        transform = transforms.RandomGrayscale(p=1)

        actual = transform(image)
        expected = F.to_image(F.rgb_to_grayscale(F.to_pil_image(image), num_output_channels=num_input_channels))

        assert_equal(actual, expected, rtol=0, atol=1)


class TestRandomZoomOut:
    # Tests are light because this largely relies on the already tested `pad` kernels.

    @pytest.mark.parametrize(
        "make_input",
        [
            make_image_tensor,
            make_image_pil,
            make_image,
            make_bounding_boxes,
            make_segmentation_mask,
            make_detection_masks,
            make_video,
        ],
    )
    def test_transform(self, make_input):
        check_transform(transforms.RandomZoomOut(p=1), make_input())

    def test_transform_error(self):
        for side_range in [None, 1, [1, 2, 3]]:
            with pytest.raises(
                ValueError if isinstance(side_range, list) else TypeError, match="should be a sequence of length 2"
            ):
                transforms.RandomZoomOut(side_range=side_range)

        for side_range in [[0.5, 1.5], [2.0, 1.0]]:
            with pytest.raises(ValueError, match="Invalid side range"):
                transforms.RandomZoomOut(side_range=side_range)

    @pytest.mark.parametrize("side_range", [(1.0, 4.0), [2.0, 5.0]])
    @pytest.mark.parametrize(
        "make_input",
        [
            make_image_tensor,
            make_image_pil,
            make_image,
            make_bounding_boxes,
            make_segmentation_mask,
            make_detection_masks,
            make_video,
        ],
    )
    @pytest.mark.parametrize("device", cpu_and_cuda())
    def test_transform_params_correctness(self, side_range, make_input, device):
        if make_input is make_image_pil and device != "cpu":
            pytest.skip("PIL image tests with parametrization device!='cpu' will degenerate to that anyway.")

        transform = transforms.RandomZoomOut(side_range=side_range)

        input = make_input()
        height, width = F.get_size(input)

        params = transform._get_params([input])
        assert "padding" in params

        padding = params["padding"]
        assert len(padding) == 4

        assert 0 <= padding[0] <= (side_range[1] - 1) * width
        assert 0 <= padding[1] <= (side_range[1] - 1) * height
        assert 0 <= padding[2] <= (side_range[1] - 1) * width
        assert 0 <= padding[3] <= (side_range[1] - 1) * height


class TestRandomPhotometricDistort:
    # Tests are light because this largely relies on the already tested
    # `adjust_{brightness,contrast,saturation,hue}` and `permute_channels` kernels.

    @pytest.mark.parametrize(
        "make_input",
        [make_image_tensor, make_image_pil, make_image, make_video],
    )
    @pytest.mark.parametrize("dtype", [torch.uint8, torch.float32])
    @pytest.mark.parametrize("device", cpu_and_cuda())
    def test_transform(self, make_input, dtype, device):
        if make_input is make_image_pil and not (dtype is torch.uint8 and device == "cpu"):
            pytest.skip(
                "PIL image tests with parametrization other than dtype=torch.uint8 and device='cpu' "
                "will degenerate to that anyway."
            )

        check_transform(
            transforms.RandomPhotometricDistort(
                brightness=(0.3, 0.4), contrast=(0.5, 0.6), saturation=(0.7, 0.8), hue=(-0.1, 0.2), p=1
            ),
            make_input(dtype=dtype, device=device),
        )


class TestScaleJitter:
    # Tests are light because this largely relies on the already tested `resize` kernels.

    INPUT_SIZE = (17, 11)
    TARGET_SIZE = (12, 13)

    @pytest.mark.parametrize(
        "make_input",
        [make_image_tensor, make_image_pil, make_image, make_bounding_boxes, make_segmentation_mask, make_video],
    )
    @pytest.mark.parametrize("device", cpu_and_cuda())
    def test_transform(self, make_input, device):
        if make_input is make_image_pil and device != "cpu":
            pytest.skip("PIL image tests with parametrization device!='cpu' will degenerate to that anyway.")

        check_transform(transforms.ScaleJitter(self.TARGET_SIZE), make_input(self.INPUT_SIZE, device=device))

    def test__get_params(self):
        input_size = self.INPUT_SIZE
        target_size = self.TARGET_SIZE
        scale_range = (0.5, 1.5)

        transform = transforms.ScaleJitter(target_size=target_size, scale_range=scale_range)
        params = transform._get_params([make_image(input_size)])

        assert "size" in params
        size = params["size"]

        assert isinstance(size, tuple) and len(size) == 2
        height, width = size

        r_min = min(target_size[1] / input_size[0], target_size[0] / input_size[1]) * scale_range[0]
        r_max = min(target_size[1] / input_size[0], target_size[0] / input_size[1]) * scale_range[1]

        assert int(input_size[0] * r_min) <= height <= int(input_size[0] * r_max)
        assert int(input_size[1] * r_min) <= width <= int(input_size[1] * r_max)


class TestLinearTransform:
    def _make_matrix_and_vector(self, input, *, device=None):
        device = device or input.device
        numel = math.prod(F.get_dimensions(input))
        transformation_matrix = torch.randn((numel, numel), device=device)
        mean_vector = torch.randn((numel,), device=device)
        return transformation_matrix, mean_vector

    def _sample_input_adapter(self, transform, input, device):
        return {key: value for key, value in input.items() if not isinstance(value, PIL.Image.Image)}

    @pytest.mark.parametrize("make_input", [make_image_tensor, make_image, make_video])
    @pytest.mark.parametrize("dtype", [torch.uint8, torch.float32])
    @pytest.mark.parametrize("device", cpu_and_cuda())
    def test_transform(self, make_input, dtype, device):
        input = make_input(dtype=dtype, device=device)
        check_transform(
            transforms.LinearTransformation(*self._make_matrix_and_vector(input)),
            input,
            check_sample_input=self._sample_input_adapter,
        )

    def test_transform_error(self):
        with pytest.raises(ValueError, match="transformation_matrix should be square"):
            transforms.LinearTransformation(transformation_matrix=torch.rand(2, 3), mean_vector=torch.rand(2))

        with pytest.raises(ValueError, match="mean_vector should have the same length"):
            transforms.LinearTransformation(transformation_matrix=torch.rand(2, 2), mean_vector=torch.rand(1))

        for matrix_dtype, vector_dtype in [(torch.float32, torch.float64), (torch.float64, torch.float32)]:
            with pytest.raises(ValueError, match="Input tensors should have the same dtype"):
                transforms.LinearTransformation(
                    transformation_matrix=torch.rand(2, 2, dtype=matrix_dtype),
                    mean_vector=torch.rand(2, dtype=vector_dtype),
                )

        image = make_image()
        transform = transforms.LinearTransformation(transformation_matrix=torch.rand(2, 2), mean_vector=torch.rand(2))
        with pytest.raises(ValueError, match="Input tensor and transformation matrix have incompatible shape"):
            transform(image)

        transform = transforms.LinearTransformation(*self._make_matrix_and_vector(image))
        with pytest.raises(TypeError, match="does not support PIL images"):
            transform(F.to_pil_image(image))

    @needs_cuda
    def test_transform_error_cuda(self):
        for matrix_device, vector_device in [("cuda", "cpu"), ("cpu", "cuda")]:
            with pytest.raises(ValueError, match="Input tensors should be on the same device"):
                transforms.LinearTransformation(
                    transformation_matrix=torch.rand(2, 2, device=matrix_device),
                    mean_vector=torch.rand(2, device=vector_device),
                )

        for input_device, param_device in [("cuda", "cpu"), ("cpu", "cuda")]:
            input = make_image(device=input_device)
            transform = transforms.LinearTransformation(*self._make_matrix_and_vector(input, device=param_device))
            with pytest.raises(
                ValueError, match="Input tensor should be on the same device as transformation matrix and mean vector"
            ):
                transform(input)


def make_image_numpy(*args, **kwargs):
    image = make_image_tensor(*args, **kwargs)
    return image.permute((1, 2, 0)).numpy()


class TestToImage:
    @pytest.mark.parametrize("make_input", [make_image_tensor, make_image_pil, make_image, make_image_numpy])
    @pytest.mark.parametrize("fn", [F.to_image, transform_cls_to_functional(transforms.ToImage)])
    def test_functional_and_transform(self, make_input, fn):
        input = make_input()
        output = fn(input)

        assert isinstance(output, tv_tensors.Image)

        input_size = list(input.shape[:2]) if isinstance(input, np.ndarray) else F.get_size(input)
        assert F.get_size(output) == input_size

        if isinstance(input, torch.Tensor):
            assert output.data_ptr() == input.data_ptr()

    def test_functional_error(self):
        with pytest.raises(TypeError, match="Input can either be a pure Tensor, a numpy array, or a PIL image"):
            F.to_image(object())


class TestToPILImage:
    @pytest.mark.parametrize("make_input", [make_image_tensor, make_image, make_image_numpy])
    @pytest.mark.parametrize("color_space", ["RGB", "GRAY"])
    @pytest.mark.parametrize("fn", [F.to_pil_image, transform_cls_to_functional(transforms.ToPILImage)])
    def test_functional_and_transform(self, make_input, color_space, fn):
        input = make_input(color_space=color_space)
        output = fn(input)

        assert isinstance(output, PIL.Image.Image)

        input_size = list(input.shape[:2]) if isinstance(input, np.ndarray) else F.get_size(input)
        assert F.get_size(output) == input_size

    def test_functional_error(self):
        with pytest.raises(TypeError, match="pic should be Tensor or ndarray"):
            F.to_pil_image(object())

        for ndim in [1, 4]:
            with pytest.raises(ValueError, match="pic should be 2/3 dimensional"):
                F.to_pil_image(torch.empty(*[1] * ndim))

        with pytest.raises(ValueError, match="pic should not have > 4 channels"):
            num_channels = 5
            F.to_pil_image(torch.empty(num_channels, 1, 1))


class TestToTensor:
    @pytest.mark.parametrize("make_input", [make_image_tensor, make_image_pil, make_image, make_image_numpy])
    def test_smoke(self, make_input):
        with pytest.warns(UserWarning, match="deprecated and will be removed"):
            transform = transforms.ToTensor()

        input = make_input()
        output = transform(input)

        input_size = list(input.shape[:2]) if isinstance(input, np.ndarray) else F.get_size(input)
        assert F.get_size(output) == input_size


class TestPILToTensor:
    @pytest.mark.parametrize("color_space", ["RGB", "GRAY"])
    @pytest.mark.parametrize("fn", [F.pil_to_tensor, transform_cls_to_functional(transforms.PILToTensor)])
    def test_functional_and_transform(self, color_space, fn):
        input = make_image_pil(color_space=color_space)
        output = fn(input)

        assert isinstance(output, torch.Tensor) and not isinstance(output, tv_tensors.TVTensor)
        assert F.get_size(output) == F.get_size(input)

    def test_functional_error(self):
        with pytest.raises(TypeError, match="pic should be PIL Image"):
            F.pil_to_tensor(object())


class TestLambda:
    @pytest.mark.parametrize("input", [object(), torch.empty(()), np.empty(()), "string", 1, 0.0])
    @pytest.mark.parametrize("types", [(), (torch.Tensor, np.ndarray)])
    def test_transform(self, input, types):
        was_applied = False

        def was_applied_fn(input):
            nonlocal was_applied
            was_applied = True
            return input

        transform = transforms.Lambda(was_applied_fn, *types)
        output = transform(input)

        assert output is input
        assert was_applied is (not types or isinstance(input, types))


@pytest.mark.parametrize(
    ("alias", "target"),
    [
        pytest.param(alias, target, id=alias.__name__)
        for alias, target in [
            (F.hflip, F.horizontal_flip),
            (F.vflip, F.vertical_flip),
            (F.get_image_num_channels, F.get_num_channels),
            (F.to_pil_image, F.to_pil_image),
            (F.elastic_transform, F.elastic),
            (F.to_grayscale, F.rgb_to_grayscale),
        ]
    ],
)
def test_alias(alias, target):
    assert alias is target


@pytest.mark.parametrize(
    "make_inputs",
    itertools.permutations(
        [
            make_image_tensor,
            make_image_tensor,
            make_image_pil,
            make_image,
            make_video,
        ],
        3,
    ),
)
def test_pure_tensor_heuristic(make_inputs):
    flat_inputs = [make_input() for make_input in make_inputs]

    def split_on_pure_tensor(to_split):
        # This takes a sequence that is structurally aligned with `flat_inputs` and splits its items into three parts:
        # 1. The first pure tensor. If none is present, this will be `None`
        # 2. A list of the remaining pure tensors
        # 3. A list of all other items
        pure_tensors = []
        others = []
        # Splitting always happens on the original `flat_inputs` to avoid any erroneous type changes by the transform to
        # affect the splitting.
        for item, inpt in zip(to_split, flat_inputs):
            (pure_tensors if is_pure_tensor(inpt) else others).append(item)
        return pure_tensors[0] if pure_tensors else None, pure_tensors[1:], others

    class CopyCloneTransform(transforms.Transform):
        def _transform(self, inpt, params):
            return inpt.clone() if isinstance(inpt, torch.Tensor) else inpt.copy()

        @staticmethod
        def was_applied(output, inpt):
            identity = output is inpt
            if identity:
                return False

            # Make sure nothing fishy is going on
            assert_equal(output, inpt)
            return True

    first_pure_tensor_input, other_pure_tensor_inputs, other_inputs = split_on_pure_tensor(flat_inputs)

    transform = CopyCloneTransform()
    transformed_sample = transform(flat_inputs)

    first_pure_tensor_output, other_pure_tensor_outputs, other_outputs = split_on_pure_tensor(transformed_sample)

    if first_pure_tensor_input is not None:
        if other_inputs:
            assert not transform.was_applied(first_pure_tensor_output, first_pure_tensor_input)
        else:
            assert transform.was_applied(first_pure_tensor_output, first_pure_tensor_input)

    for output, inpt in zip(other_pure_tensor_outputs, other_pure_tensor_inputs):
        assert not transform.was_applied(output, inpt)

    for input, output in zip(other_inputs, other_outputs):
        assert transform.was_applied(output, input)


class TestRandomIoUCrop:
    @pytest.mark.parametrize("device", cpu_and_cuda())
    @pytest.mark.parametrize("options", [[0.5, 0.9], [2.0]])
    def test__get_params(self, device, options):
        orig_h, orig_w = size = (24, 32)
        image = make_image(size)
        bboxes = tv_tensors.BoundingBoxes(
            torch.tensor([[1, 1, 10, 10], [20, 20, 23, 23], [1, 20, 10, 23], [20, 1, 23, 10]]),
            format="XYXY",
            canvas_size=size,
            device=device,
        )
        sample = [image, bboxes]

        transform = transforms.RandomIoUCrop(sampler_options=options)

        n_samples = 5
        for _ in range(n_samples):

            params = transform._get_params(sample)

            if options == [2.0]:
                assert len(params) == 0
                return

            assert len(params["is_within_crop_area"]) > 0
            assert params["is_within_crop_area"].dtype == torch.bool

            assert int(transform.min_scale * orig_h) <= params["height"] <= int(transform.max_scale * orig_h)
            assert int(transform.min_scale * orig_w) <= params["width"] <= int(transform.max_scale * orig_w)

            left, top = params["left"], params["top"]
            new_h, new_w = params["height"], params["width"]
            ious = box_iou(
                bboxes,
                torch.tensor([[left, top, left + new_w, top + new_h]], dtype=bboxes.dtype, device=bboxes.device),
            )
            assert ious.max() >= options[0] or ious.max() >= options[1], f"{ious} vs {options}"

    def test__transform_empty_params(self, mocker):
        transform = transforms.RandomIoUCrop(sampler_options=[2.0])
        image = tv_tensors.Image(torch.rand(1, 3, 4, 4))
        bboxes = tv_tensors.BoundingBoxes(torch.tensor([[1, 1, 2, 2]]), format="XYXY", canvas_size=(4, 4))
        label = torch.tensor([1])
        sample = [image, bboxes, label]
        # Let's mock transform._get_params to control the output:
        transform._get_params = mocker.MagicMock(return_value={})
        output = transform(sample)
        torch.testing.assert_close(output, sample)

    def test_forward_assertion(self):
        transform = transforms.RandomIoUCrop()
        with pytest.raises(
            TypeError,
            match="requires input sample to contain tensor or PIL images and bounding boxes",
        ):
            transform(torch.tensor(0))

    def test__transform(self, mocker):
        transform = transforms.RandomIoUCrop()

        size = (32, 24)
        image = make_image(size)
        bboxes = make_bounding_boxes(format="XYXY", canvas_size=size, num_boxes=6)
        masks = make_detection_masks(size, num_masks=6)

        sample = [image, bboxes, masks]

        is_within_crop_area = torch.tensor([0, 1, 0, 1, 0, 1], dtype=torch.bool)

        params = dict(top=1, left=2, height=12, width=12, is_within_crop_area=is_within_crop_area)
        transform._get_params = mocker.MagicMock(return_value=params)
        output = transform(sample)

        # check number of bboxes vs number of labels:
        output_bboxes = output[1]
        assert isinstance(output_bboxes, tv_tensors.BoundingBoxes)
        assert (output_bboxes[~is_within_crop_area] == 0).all()

        output_masks = output[2]
        assert isinstance(output_masks, tv_tensors.Mask)


class TestRandomShortestSize:
    @pytest.mark.parametrize("min_size,max_size", [([5, 9], 20), ([5, 9], None)])
    def test__get_params(self, min_size, max_size):
        canvas_size = (3, 10)

        transform = transforms.RandomShortestSize(min_size=min_size, max_size=max_size, antialias=True)

        sample = make_image(canvas_size)
        params = transform._get_params([sample])

        assert "size" in params
        size = params["size"]

        assert isinstance(size, tuple) and len(size) == 2

        longer = max(size)
        shorter = min(size)
        if max_size is not None:
            assert longer <= max_size
            assert shorter <= max_size
        else:
            assert shorter in min_size


class TestRandomResize:
    def test__get_params(self):
        min_size = 3
        max_size = 6

        transform = transforms.RandomResize(min_size=min_size, max_size=max_size, antialias=True)

        for _ in range(10):
            params = transform._get_params([])

            assert isinstance(params["size"], list) and len(params["size"]) == 1
            size = params["size"][0]

            assert min_size <= size < max_size


@pytest.mark.parametrize("image_type", (PIL.Image, torch.Tensor, tv_tensors.Image))
@pytest.mark.parametrize("label_type", (torch.Tensor, int))
@pytest.mark.parametrize("dataset_return_type", (dict, tuple))
@pytest.mark.parametrize("to_tensor", (transforms.ToTensor, transforms.ToImage))
def test_classification_preset(image_type, label_type, dataset_return_type, to_tensor):

    image = tv_tensors.Image(torch.randint(0, 256, size=(1, 3, 250, 250), dtype=torch.uint8))
    if image_type is PIL.Image:
        image = to_pil_image(image[0])
    elif image_type is torch.Tensor:
        image = image.as_subclass(torch.Tensor)
        assert is_pure_tensor(image)

    label = 1 if label_type is int else torch.tensor([1])

    if dataset_return_type is dict:
        sample = {
            "image": image,
            "label": label,
        }
    else:
        sample = image, label

    if to_tensor is transforms.ToTensor:
        with pytest.warns(UserWarning, match="deprecated and will be removed"):
            to_tensor = to_tensor()
    else:
        to_tensor = to_tensor()

    t = transforms.Compose(
        [
            transforms.RandomResizedCrop((224, 224), antialias=True),
            transforms.RandomHorizontalFlip(p=1),
            transforms.RandAugment(),
            transforms.TrivialAugmentWide(),
            transforms.AugMix(),
            transforms.AutoAugment(),
            to_tensor,
            # TODO: ConvertImageDtype is a pass-through on PIL images, is that
            # intended?  This results in a failure if we convert to tensor after
            # it, because the image would still be uint8 which make Normalize
            # fail.
            transforms.ConvertImageDtype(torch.float),
            transforms.Normalize(mean=[0, 0, 0], std=[1, 1, 1]),
            transforms.RandomErasing(p=1),
        ]
    )

    out = t(sample)

    assert type(out) == type(sample)

    if dataset_return_type is tuple:
        out_image, out_label = out
    else:
        assert out.keys() == sample.keys()
        out_image, out_label = out.values()

    assert out_image.shape[-2:] == (224, 224)
    assert out_label == label


@pytest.mark.parametrize("image_type", (PIL.Image, torch.Tensor, tv_tensors.Image))
@pytest.mark.parametrize("data_augmentation", ("hflip", "lsj", "multiscale", "ssd", "ssdlite"))
@pytest.mark.parametrize("to_tensor", (transforms.ToTensor, transforms.ToImage))
@pytest.mark.parametrize("sanitize", (True, False))
def test_detection_preset(image_type, data_augmentation, to_tensor, sanitize):
    torch.manual_seed(0)

    if to_tensor is transforms.ToTensor:
        with pytest.warns(UserWarning, match="deprecated and will be removed"):
            to_tensor = to_tensor()
    else:
        to_tensor = to_tensor()

    if data_augmentation == "hflip":
        t = [
            transforms.RandomHorizontalFlip(p=1),
            to_tensor,
            transforms.ConvertImageDtype(torch.float),
        ]
    elif data_augmentation == "lsj":
        t = [
            transforms.ScaleJitter(target_size=(1024, 1024), antialias=True),
            # Note: replaced FixedSizeCrop with RandomCrop, becuase we're
            # leaving FixedSizeCrop in prototype for now, and it expects Label
            # classes which we won't release yet.
            # transforms.FixedSizeCrop(
            #     size=(1024, 1024), fill=defaultdict(lambda: (123.0, 117.0, 104.0), {tv_tensors.Mask: 0})
            # ),
            transforms.RandomCrop((1024, 1024), pad_if_needed=True),
            transforms.RandomHorizontalFlip(p=1),
            to_tensor,
            transforms.ConvertImageDtype(torch.float),
        ]
    elif data_augmentation == "multiscale":
        t = [
            transforms.RandomShortestSize(
                min_size=(480, 512, 544, 576, 608, 640, 672, 704, 736, 768, 800), max_size=1333, antialias=True
            ),
            transforms.RandomHorizontalFlip(p=1),
            to_tensor,
            transforms.ConvertImageDtype(torch.float),
        ]
    elif data_augmentation == "ssd":
        t = [
            transforms.RandomPhotometricDistort(p=1),
            transforms.RandomZoomOut(fill={"others": (123.0, 117.0, 104.0), tv_tensors.Mask: 0}, p=1),
            transforms.RandomIoUCrop(),
            transforms.RandomHorizontalFlip(p=1),
            to_tensor,
            transforms.ConvertImageDtype(torch.float),
        ]
    elif data_augmentation == "ssdlite":
        t = [
            transforms.RandomIoUCrop(),
            transforms.RandomHorizontalFlip(p=1),
            to_tensor,
            transforms.ConvertImageDtype(torch.float),
        ]
    if sanitize:
        t += [transforms.SanitizeBoundingBoxes()]
    t = transforms.Compose(t)

    num_boxes = 5
    H = W = 250

    image = tv_tensors.Image(torch.randint(0, 256, size=(1, 3, H, W), dtype=torch.uint8))
    if image_type is PIL.Image:
        image = to_pil_image(image[0])
    elif image_type is torch.Tensor:
        image = image.as_subclass(torch.Tensor)
        assert is_pure_tensor(image)

    label = torch.randint(0, 10, size=(num_boxes,))

    boxes = torch.randint(0, min(H, W) // 2, size=(num_boxes, 4))
    boxes[:, 2:] += boxes[:, :2]
    boxes = boxes.clamp(min=0, max=min(H, W))
    boxes = tv_tensors.BoundingBoxes(boxes, format="XYXY", canvas_size=(H, W))

    masks = tv_tensors.Mask(torch.randint(0, 2, size=(num_boxes, H, W), dtype=torch.uint8))

    sample = {
        "image": image,
        "label": label,
        "boxes": boxes,
        "masks": masks,
    }

    out = t(sample)

    if isinstance(to_tensor, transforms.ToTensor) and image_type is not tv_tensors.Image:
        assert is_pure_tensor(out["image"])
    else:
        assert isinstance(out["image"], tv_tensors.Image)
    assert isinstance(out["label"], type(sample["label"]))

    num_boxes_expected = {
        # ssd and ssdlite contain RandomIoUCrop which may "remove" some bbox. It
        # doesn't remove them strictly speaking, it just marks some boxes as
        # degenerate and those boxes will be later removed by
        # SanitizeBoundingBoxes(), which we add to the pipelines if the sanitize
        # param is True.
        # Note that the values below are probably specific to the random seed
        # set above (which is fine).
        (True, "ssd"): 5,
        (True, "ssdlite"): 4,
    }.get((sanitize, data_augmentation), num_boxes)

    assert out["boxes"].shape[0] == out["masks"].shape[0] == out["label"].shape[0] == num_boxes_expected


class TestSanitizeBoundingBoxes:
    @pytest.mark.parametrize("min_size", (1, 10))
    @pytest.mark.parametrize("labels_getter", ("default", lambda inputs: inputs["labels"], None, lambda inputs: None))
    @pytest.mark.parametrize("sample_type", (tuple, dict))
    def test_transform(self, min_size, labels_getter, sample_type):

        if sample_type is tuple and not isinstance(labels_getter, str):
            # The "lambda inputs: inputs["labels"]" labels_getter used in this test
            # doesn't work if the input is a tuple.
            return

        H, W = 256, 128

        boxes_and_validity = [
            ([0, 1, 10, 1], False),  # Y1 == Y2
            ([0, 1, 0, 20], False),  # X1 == X2
            ([0, 0, min_size - 1, 10], False),  # H < min_size
            ([0, 0, 10, min_size - 1], False),  # W < min_size
            ([0, 0, 10, H + 1], False),  # Y2 > H
            ([0, 0, W + 1, 10], False),  # X2 > W
            ([-1, 1, 10, 20], False),  # any < 0
            ([0, 0, -1, 20], False),  # any < 0
            ([0, 0, -10, -1], False),  # any < 0
            ([0, 0, min_size, 10], True),  # H < min_size
            ([0, 0, 10, min_size], True),  # W < min_size
            ([0, 0, W, H], True),  # TODO: Is that actually OK?? Should it be -1?
            ([1, 1, 30, 20], True),
            ([0, 0, 10, 10], True),
            ([1, 1, 30, 20], True),
        ]

        random.shuffle(boxes_and_validity)  # For test robustness: mix order of wrong and correct cases
        boxes, is_valid_mask = zip(*boxes_and_validity)
        valid_indices = [i for (i, is_valid) in enumerate(is_valid_mask) if is_valid]

        boxes = torch.tensor(boxes)
        labels = torch.arange(boxes.shape[0])

        boxes = tv_tensors.BoundingBoxes(
            boxes,
            format=tv_tensors.BoundingBoxFormat.XYXY,
            canvas_size=(H, W),
        )

        masks = tv_tensors.Mask(torch.randint(0, 2, size=(boxes.shape[0], H, W)))
        whatever = torch.rand(10)
        input_img = torch.randint(0, 256, size=(1, 3, H, W), dtype=torch.uint8)
        sample = {
            "image": input_img,
            "labels": labels,
            "boxes": boxes,
            "whatever": whatever,
            "None": None,
            "masks": masks,
        }

        if sample_type is tuple:
            img = sample.pop("image")
            sample = (img, sample)

        out = transforms.SanitizeBoundingBoxes(min_size=min_size, labels_getter=labels_getter)(sample)

        if sample_type is tuple:
            out_image = out[0]
            out_labels = out[1]["labels"]
            out_boxes = out[1]["boxes"]
            out_masks = out[1]["masks"]
            out_whatever = out[1]["whatever"]
        else:
            out_image = out["image"]
            out_labels = out["labels"]
            out_boxes = out["boxes"]
            out_masks = out["masks"]
            out_whatever = out["whatever"]

        assert out_image is input_img
        assert out_whatever is whatever

        assert isinstance(out_boxes, tv_tensors.BoundingBoxes)
        assert isinstance(out_masks, tv_tensors.Mask)

        if labels_getter is None or (callable(labels_getter) and labels_getter({"labels": "blah"}) is None):
            assert out_labels is labels
        else:
            assert isinstance(out_labels, torch.Tensor)
            assert out_boxes.shape[0] == out_labels.shape[0] == out_masks.shape[0]
            # This works because we conveniently set labels to arange(num_boxes)
            assert out_labels.tolist() == valid_indices

    def test_no_label(self):
        # Non-regression test for https://github.com/pytorch/vision/issues/7878

        img = make_image()
        boxes = make_bounding_boxes()

        with pytest.raises(ValueError, match="or a two-tuple whose second item is a dict"):
            transforms.SanitizeBoundingBoxes()(img, boxes)

        out_img, out_boxes = transforms.SanitizeBoundingBoxes(labels_getter=None)(img, boxes)
        assert isinstance(out_img, tv_tensors.Image)
        assert isinstance(out_boxes, tv_tensors.BoundingBoxes)

    def test_errors(self):
        good_bbox = tv_tensors.BoundingBoxes(
            [[0, 0, 10, 10]],
            format=tv_tensors.BoundingBoxFormat.XYXY,
            canvas_size=(20, 20),
        )

        with pytest.raises(ValueError, match="min_size must be >= 1"):
            transforms.SanitizeBoundingBoxes(min_size=0)
        with pytest.raises(ValueError, match="labels_getter should either be 'default'"):
            transforms.SanitizeBoundingBoxes(labels_getter=12)

        with pytest.raises(ValueError, match="Could not infer where the labels are"):
            bad_labels_key = {"bbox": good_bbox, "BAD_KEY": torch.arange(good_bbox.shape[0])}
            transforms.SanitizeBoundingBoxes()(bad_labels_key)

        with pytest.raises(ValueError, match="must be a tensor"):
            not_a_tensor = {"bbox": good_bbox, "labels": torch.arange(good_bbox.shape[0]).tolist()}
            transforms.SanitizeBoundingBoxes()(not_a_tensor)

        with pytest.raises(ValueError, match="Number of boxes"):
            different_sizes = {"bbox": good_bbox, "labels": torch.arange(good_bbox.shape[0] + 3)}
            transforms.SanitizeBoundingBoxes()(different_sizes)
