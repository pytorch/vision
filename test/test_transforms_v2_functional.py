import inspect
import math
import os
import re

from typing import get_type_hints

import numpy as np
import PIL.Image
import pytest

import torch

from common_utils import (
    assert_close,
    cache,
    cpu_and_cuda,
    DEFAULT_SQUARE_SPATIAL_SIZE,
    make_bboxes,
    needs_cuda,
    parametrized_error_message,
    set_rng_seed,
)
from torch.utils._pytree import tree_map
from torchvision import datapoints
from torchvision.transforms.functional import _get_perspective_coeffs
from torchvision.transforms.v2 import functional as F
from torchvision.transforms.v2.functional._geometry import _center_crop_compute_padding
from torchvision.transforms.v2.functional._meta import clamp_bboxes, convert_format_bboxes
from torchvision.transforms.v2.utils import is_simple_tensor
from transforms_v2_dispatcher_infos import DISPATCHER_INFOS
from transforms_v2_kernel_infos import KERNEL_INFOS


KERNEL_INFOS_MAP = {info.kernel: info for info in KERNEL_INFOS}
DISPATCHER_INFOS_MAP = {info.dispatcher: info for info in DISPATCHER_INFOS}


@cache
def script(fn):
    try:
        return torch.jit.script(fn)
    except Exception as error:
        raise AssertionError(f"Trying to `torch.jit.script` '{fn.__name__}' raised the error above.") from error


# Scripting a function often triggers a warning like
# `UserWarning: operator() profile_node %$INT1 : int[] = prim::profile_ivalue($INT2) does not have profile information`
# with varying `INT1` and `INT2`. Since these are uninteresting for us and only clutter the test summary, we ignore
# them.
ignore_jit_warning_no_profile = pytest.mark.filterwarnings(
    f"ignore:{re.escape('operator() profile_node %')}:UserWarning"
)


def make_info_args_kwargs_params(info, *, args_kwargs_fn, test_id=None):
    args_kwargs = list(args_kwargs_fn(info))
    if not args_kwargs:
        raise pytest.UsageError(
            f"Couldn't collect a single `ArgsKwargs` for `{info.id}`{f' in {test_id}' if test_id else ''}"
        )
    idx_field_len = len(str(len(args_kwargs)))
    return [
        pytest.param(
            info,
            args_kwargs_,
            marks=info.get_marks(test_id, args_kwargs_) if test_id else [],
            id=f"{info.id}-{idx:0{idx_field_len}}",
        )
        for idx, args_kwargs_ in enumerate(args_kwargs)
    ]


def make_info_args_kwargs_parametrization(infos, *, args_kwargs_fn):
    def decorator(test_fn):
        parts = test_fn.__qualname__.split(".")
        if len(parts) == 1:
            test_class_name = None
            test_function_name = parts[0]
        elif len(parts) == 2:
            test_class_name, test_function_name = parts
        else:
            raise pytest.UsageError("Unable to parse the test class name and test function name from test function")
        test_id = (test_class_name, test_function_name)

        argnames = ("info", "args_kwargs")
        argvalues = []
        for info in infos:
            argvalues.extend(make_info_args_kwargs_params(info, args_kwargs_fn=args_kwargs_fn, test_id=test_id))

        return pytest.mark.parametrize(argnames, argvalues)(test_fn)

    return decorator


@pytest.fixture(autouse=True)
def fix_rng_seed():
    set_rng_seed(0)
    yield


@pytest.fixture()
def test_id(request):
    test_class_name = request.cls.__name__ if request.cls is not None else None
    test_function_name = request.node.originalname
    return test_class_name, test_function_name


class TestKernels:
    sample_inputs = make_info_args_kwargs_parametrization(
        KERNEL_INFOS,
        args_kwargs_fn=lambda kernel_info: kernel_info.sample_inputs_fn(),
    )
    reference_inputs = make_info_args_kwargs_parametrization(
        [info for info in KERNEL_INFOS if info.reference_fn is not None],
        args_kwargs_fn=lambda info: info.reference_inputs_fn(),
    )

    @make_info_args_kwargs_parametrization(
        [info for info in KERNEL_INFOS if info.logs_usage],
        args_kwargs_fn=lambda info: info.sample_inputs_fn(),
    )
    @pytest.mark.parametrize("device", cpu_and_cuda())
    def test_logging(self, spy_on, info, args_kwargs, device):
        spy = spy_on(torch._C._log_api_usage_once)

        (input, *other_args), kwargs = args_kwargs.load(device)
        info.kernel(input.as_subclass(torch.Tensor), *other_args, **kwargs)

        spy.assert_any_call(f"{info.kernel.__module__}.{info.id}")

    @ignore_jit_warning_no_profile
    @sample_inputs
    @pytest.mark.parametrize("device", cpu_and_cuda())
    def test_scripted_vs_eager(self, test_id, info, args_kwargs, device):
        kernel_eager = info.kernel
        kernel_scripted = script(kernel_eager)

        (input, *other_args), kwargs = args_kwargs.load(device)
        input = input.as_subclass(torch.Tensor)

        actual = kernel_scripted(input, *other_args, **kwargs)
        expected = kernel_eager(input, *other_args, **kwargs)

        assert_close(
            actual,
            expected,
            **info.get_closeness_kwargs(test_id, dtype=input.dtype, device=input.device),
            msg=parametrized_error_message(input, other_args, **kwargs),
        )

    def _unbatch(self, batch, *, data_dims):
        if isinstance(batch, torch.Tensor):
            batched_tensor = batch
            metadata = ()
        else:
            batched_tensor, *metadata = batch

        if batched_tensor.ndim == data_dims:
            return batch

        return [
            self._unbatch(unbatched, data_dims=data_dims)
            for unbatched in (
                batched_tensor.unbind(0) if not metadata else [(t, *metadata) for t in batched_tensor.unbind(0)]
            )
        ]

    @sample_inputs
    @pytest.mark.parametrize("device", cpu_and_cuda())
    def test_batched_vs_single(self, test_id, info, args_kwargs, device):
        (batched_input, *other_args), kwargs = args_kwargs.load(device)

        datapoint_type = datapoints.Image if is_simple_tensor(batched_input) else type(batched_input)
        # This dictionary contains the number of rightmost dimensions that contain the actual data.
        # Everything to the left is considered a batch dimension.
        data_dims = {
            datapoints.Image: 3,
            datapoints.BBoxes: 1,
            # `Mask`'s are special in the sense that the data dimensions depend on the type of mask. For detection masks
            # it is 3 `(*, N, H, W)`, but for segmentation masks it is 2 `(*, H, W)`. Since both a grouped under one
            # type all kernels should also work without differentiating between the two. Thus, we go with 2 here as
            # common ground.
            datapoints.Mask: 2,
            datapoints.Video: 4,
        }.get(datapoint_type)
        if data_dims is None:
            raise pytest.UsageError(
                f"The number of data dimensions cannot be determined for input of type {datapoint_type.__name__}."
            ) from None
        elif batched_input.ndim <= data_dims:
            pytest.skip("Input is not batched.")
        elif not all(batched_input.shape[:-data_dims]):
            pytest.skip("Input has a degenerate batch shape.")

        batched_input = batched_input.as_subclass(torch.Tensor)
        batched_output = info.kernel(batched_input, *other_args, **kwargs)
        actual = self._unbatch(batched_output, data_dims=data_dims)

        single_inputs = self._unbatch(batched_input, data_dims=data_dims)
        expected = tree_map(lambda single_input: info.kernel(single_input, *other_args, **kwargs), single_inputs)

        assert_close(
            actual,
            expected,
            **info.get_closeness_kwargs(test_id, dtype=batched_input.dtype, device=batched_input.device),
            msg=parametrized_error_message(batched_input, *other_args, **kwargs),
        )

    @sample_inputs
    @pytest.mark.parametrize("device", cpu_and_cuda())
    def test_no_inplace(self, info, args_kwargs, device):
        (input, *other_args), kwargs = args_kwargs.load(device)
        input = input.as_subclass(torch.Tensor)

        if input.numel() == 0:
            pytest.skip("The input has a degenerate shape.")

        input_version = input._version
        info.kernel(input, *other_args, **kwargs)

        assert input._version == input_version

    @sample_inputs
    @needs_cuda
    def test_cuda_vs_cpu(self, test_id, info, args_kwargs):
        (input_cpu, *other_args), kwargs = args_kwargs.load("cpu")
        input_cpu = input_cpu.as_subclass(torch.Tensor)
        input_cuda = input_cpu.to("cuda")

        output_cpu = info.kernel(input_cpu, *other_args, **kwargs)
        output_cuda = info.kernel(input_cuda, *other_args, **kwargs)

        assert_close(
            output_cuda,
            output_cpu,
            check_device=False,
            **info.get_closeness_kwargs(test_id, dtype=input_cuda.dtype, device=input_cuda.device),
            msg=parametrized_error_message(input_cpu, *other_args, **kwargs),
        )

    @sample_inputs
    @pytest.mark.parametrize("device", cpu_and_cuda())
    def test_dtype_and_device_consistency(self, info, args_kwargs, device):
        (input, *other_args), kwargs = args_kwargs.load(device)
        input = input.as_subclass(torch.Tensor)

        output = info.kernel(input, *other_args, **kwargs)
        # Most kernels just return a tensor, but some also return some additional metadata
        if not isinstance(output, torch.Tensor):
            output, *_ = output

        assert output.dtype == input.dtype
        assert output.device == input.device

    @reference_inputs
    def test_against_reference(self, test_id, info, args_kwargs):
        (input, *other_args), kwargs = args_kwargs.load("cpu")

        actual = info.kernel(input.as_subclass(torch.Tensor), *other_args, **kwargs)
        # We intnetionally don't unwrap the input of the reference function in order for it to have access to all
        # metadata regardless of whether the kernel takes it explicitly or not
        expected = info.reference_fn(input, *other_args, **kwargs)

        assert_close(
            actual,
            expected,
            **info.get_closeness_kwargs(test_id, dtype=input.dtype, device=input.device),
            msg=parametrized_error_message(input, *other_args, **kwargs),
        )

    @make_info_args_kwargs_parametrization(
        [info for info in KERNEL_INFOS if info.float32_vs_uint8],
        args_kwargs_fn=lambda info: info.reference_inputs_fn(),
    )
    def test_float32_vs_uint8(self, test_id, info, args_kwargs):
        (input, *other_args), kwargs = args_kwargs.load("cpu")
        input = input.as_subclass(torch.Tensor)

        if input.dtype != torch.uint8:
            pytest.skip(f"Input dtype is {input.dtype}.")

        adapted_other_args, adapted_kwargs = info.float32_vs_uint8(other_args, kwargs)

        actual = info.kernel(
            F.to_dtype_image_tensor(input, dtype=torch.float32, scale=True),
            *adapted_other_args,
            **adapted_kwargs,
        )

        expected = F.to_dtype_image_tensor(info.kernel(input, *other_args, **kwargs), dtype=torch.float32, scale=True)

        assert_close(
            actual,
            expected,
            **info.get_closeness_kwargs(test_id, dtype=torch.float32, device=input.device),
            msg=parametrized_error_message(input, *other_args, **kwargs),
        )


@pytest.fixture
def spy_on(mocker):
    def make_spy(fn, *, module=None, name=None):
        # TODO: we can probably get rid of the non-default modules and names if we eliminate aliasing
        module = module or fn.__module__
        name = name or fn.__name__
        spy = mocker.patch(f"{module}.{name}", wraps=fn)
        return spy

    return make_spy


class TestDispatchers:
    image_sample_inputs = make_info_args_kwargs_parametrization(
        [info for info in DISPATCHER_INFOS if datapoints.Image in info.kernels],
        args_kwargs_fn=lambda info: info.sample_inputs(datapoints.Image),
    )

    @make_info_args_kwargs_parametrization(
        DISPATCHER_INFOS,
        args_kwargs_fn=lambda info: info.sample_inputs(),
    )
    @pytest.mark.parametrize("device", cpu_and_cuda())
    def test_logging(self, spy_on, info, args_kwargs, device):
        spy = spy_on(torch._C._log_api_usage_once)

        args, kwargs = args_kwargs.load(device)
        info.dispatcher(*args, **kwargs)

        spy.assert_any_call(f"{info.dispatcher.__module__}.{info.id}")

    @ignore_jit_warning_no_profile
    @image_sample_inputs
    @pytest.mark.parametrize("device", cpu_and_cuda())
    def test_scripted_smoke(self, info, args_kwargs, device):
        dispatcher = script(info.dispatcher)

        (image_datapoint, *other_args), kwargs = args_kwargs.load(device)
        image_simple_tensor = torch.Tensor(image_datapoint)

        dispatcher(image_simple_tensor, *other_args, **kwargs)

    # TODO: We need this until the dispatchers below also have `DispatcherInfo`'s. If they do, `test_scripted_smoke`
    #  replaces this test for them.
    @ignore_jit_warning_no_profile
    @pytest.mark.parametrize(
        "dispatcher",
        [
            F.get_dimensions,
            F.get_image_num_channels,
            F.get_image_size,
            F.get_num_channels,
            F.get_num_frames,
            F.get_spatial_size,
            F.rgb_to_grayscale,
            F.uniform_temporal_subsample,
        ],
        ids=lambda dispatcher: dispatcher.__name__,
    )
    def test_scriptable(self, dispatcher):
        script(dispatcher)

    @image_sample_inputs
    def test_dispatch_simple_tensor(self, info, args_kwargs, spy_on):
        (image_datapoint, *other_args), kwargs = args_kwargs.load()
        image_simple_tensor = torch.Tensor(image_datapoint)

        kernel_info = info.kernel_infos[datapoints.Image]
        spy = spy_on(kernel_info.kernel, module=info.dispatcher.__module__, name=kernel_info.id)

        info.dispatcher(image_simple_tensor, *other_args, **kwargs)

        spy.assert_called_once()

    @image_sample_inputs
    def test_simple_tensor_output_type(self, info, args_kwargs):
        (image_datapoint, *other_args), kwargs = args_kwargs.load()
        image_simple_tensor = image_datapoint.as_subclass(torch.Tensor)

        output = info.dispatcher(image_simple_tensor, *other_args, **kwargs)

        # We cannot use `isinstance` here since all datapoints are instances of `torch.Tensor` as well
        assert type(output) is torch.Tensor

    @make_info_args_kwargs_parametrization(
        [info for info in DISPATCHER_INFOS if info.pil_kernel_info is not None],
        args_kwargs_fn=lambda info: info.sample_inputs(datapoints.Image),
    )
    def test_dispatch_pil(self, info, args_kwargs, spy_on):
        (image_datapoint, *other_args), kwargs = args_kwargs.load()

        if image_datapoint.ndim > 3:
            pytest.skip("Input is batched")

        image_pil = F.to_image_pil(image_datapoint)

        pil_kernel_info = info.pil_kernel_info
        spy = spy_on(pil_kernel_info.kernel, module=info.dispatcher.__module__, name=pil_kernel_info.id)

        info.dispatcher(image_pil, *other_args, **kwargs)

        spy.assert_called_once()

    @make_info_args_kwargs_parametrization(
        [info for info in DISPATCHER_INFOS if info.pil_kernel_info is not None],
        args_kwargs_fn=lambda info: info.sample_inputs(datapoints.Image),
    )
    def test_pil_output_type(self, info, args_kwargs):
        (image_datapoint, *other_args), kwargs = args_kwargs.load()

        if image_datapoint.ndim > 3:
            pytest.skip("Input is batched")

        image_pil = F.to_image_pil(image_datapoint)

        output = info.dispatcher(image_pil, *other_args, **kwargs)

        assert isinstance(output, PIL.Image.Image)

    @make_info_args_kwargs_parametrization(
        DISPATCHER_INFOS,
        args_kwargs_fn=lambda info: info.sample_inputs(),
    )
    def test_dispatch_datapoint(self, info, args_kwargs, spy_on):
        (datapoint, *other_args), kwargs = args_kwargs.load()

        method_name = info.id
        method = getattr(datapoint, method_name)
        datapoint_type = type(datapoint)
        spy = spy_on(method, module=datapoint_type.__module__, name=f"{datapoint_type.__name__}.{method_name}")

        info.dispatcher(datapoint, *other_args, **kwargs)

        spy.assert_called_once()

    @make_info_args_kwargs_parametrization(
        DISPATCHER_INFOS,
        args_kwargs_fn=lambda info: info.sample_inputs(),
    )
    def test_datapoint_output_type(self, info, args_kwargs):
        (datapoint, *other_args), kwargs = args_kwargs.load()

        output = info.dispatcher(datapoint, *other_args, **kwargs)

        assert isinstance(output, type(datapoint))

    @pytest.mark.parametrize(
        ("dispatcher_info", "datapoint_type", "kernel_info"),
        [
            pytest.param(
                dispatcher_info, datapoint_type, kernel_info, id=f"{dispatcher_info.id}-{datapoint_type.__name__}"
            )
            for dispatcher_info in DISPATCHER_INFOS
            for datapoint_type, kernel_info in dispatcher_info.kernel_infos.items()
        ],
    )
    def test_dispatcher_kernel_signatures_consistency(self, dispatcher_info, datapoint_type, kernel_info):
        dispatcher_signature = inspect.signature(dispatcher_info.dispatcher)
        dispatcher_params = list(dispatcher_signature.parameters.values())[1:]

        kernel_signature = inspect.signature(kernel_info.kernel)
        kernel_params = list(kernel_signature.parameters.values())[1:]

        # We filter out metadata that is implicitly passed to the dispatcher through the input datapoint, but has to be
        # explicit passed to the kernel.
        datapoint_type_metadata = datapoint_type.__annotations__.keys()
        kernel_params = [param for param in kernel_params if param.name not in datapoint_type_metadata]

        dispatcher_params = iter(dispatcher_params)
        for dispatcher_param, kernel_param in zip(dispatcher_params, kernel_params):
            try:
                # In general, the dispatcher parameters are a superset of the kernel parameters. Thus, we filter out
                # dispatcher parameters that have no kernel equivalent while keeping the order intact.
                while dispatcher_param.name != kernel_param.name:
                    dispatcher_param = next(dispatcher_params)
            except StopIteration:
                raise AssertionError(
                    f"Parameter `{kernel_param.name}` of kernel `{kernel_info.id}` "
                    f"has no corresponding parameter on the dispatcher `{dispatcher_info.id}`."
                ) from None

            assert dispatcher_param == kernel_param

    @pytest.mark.parametrize("info", DISPATCHER_INFOS, ids=lambda info: info.id)
    def test_dispatcher_datapoint_signatures_consistency(self, info):
        try:
            datapoint_method = getattr(datapoints._datapoint.Datapoint, info.id)
        except AttributeError:
            pytest.skip("Dispatcher doesn't support arbitrary datapoint dispatch.")

        dispatcher_signature = inspect.signature(info.dispatcher)
        dispatcher_params = list(dispatcher_signature.parameters.values())[1:]

        datapoint_signature = inspect.signature(datapoint_method)
        datapoint_params = list(datapoint_signature.parameters.values())[1:]

        # Because we use `from __future__ import annotations` inside the module where `datapoints._datapoint` is
        # defined, the annotations are stored as strings. This makes them concrete again, so they can be compared to the
        # natively concrete dispatcher annotations.
        datapoint_annotations = get_type_hints(datapoint_method)
        for param in datapoint_params:
            param._annotation = datapoint_annotations[param.name]

        assert dispatcher_params == datapoint_params

    @pytest.mark.parametrize("info", DISPATCHER_INFOS, ids=lambda info: info.id)
    def test_unkown_type(self, info):
        unkown_input = object()
        (_, *other_args), kwargs = next(iter(info.sample_inputs())).load("cpu")

        with pytest.raises(TypeError, match=re.escape(str(type(unkown_input)))):
            info.dispatcher(unkown_input, *other_args, **kwargs)

    @make_info_args_kwargs_parametrization(
        [
            info
            for info in DISPATCHER_INFOS
            if datapoints.BBoxes in info.kernels and info.dispatcher is not F.convert_format_bboxes
        ],
        args_kwargs_fn=lambda info: info.sample_inputs(datapoints.BBoxes),
    )
    def test_bboxes_format_consistency(self, info, args_kwargs):
        (bboxes, *other_args), kwargs = args_kwargs.load()
        format = bboxes.format

        output = info.dispatcher(bboxes, *other_args, **kwargs)

        assert output.format == format


@pytest.mark.parametrize(
    ("alias", "target"),
    [
        pytest.param(alias, target, id=alias.__name__)
        for alias, target in [
            (F.hflip, F.horizontal_flip),
            (F.vflip, F.vertical_flip),
            (F.get_image_num_channels, F.get_num_channels),
            (F.to_pil_image, F.to_image_pil),
            (F.elastic_transform, F.elastic),
            (F.to_grayscale, F.rgb_to_grayscale),
        ]
    ],
)
def test_alias(alias, target):
    assert alias is target


@pytest.mark.parametrize("device", cpu_and_cuda())
@pytest.mark.parametrize("num_channels", [1, 3])
def test_normalize_image_tensor_stats(device, num_channels):
    stats = pytest.importorskip("scipy.stats", reason="SciPy is not available")

    def assert_samples_from_standard_normal(t):
        p_value = stats.kstest(t.flatten(), cdf="norm", args=(0, 1)).pvalue
        return p_value > 1e-4

    image = torch.rand(num_channels, DEFAULT_SQUARE_SPATIAL_SIZE, DEFAULT_SQUARE_SPATIAL_SIZE)
    mean = image.mean(dim=(1, 2)).tolist()
    std = image.std(dim=(1, 2)).tolist()

    assert_samples_from_standard_normal(F.normalize_image_tensor(image, mean, std))


class TestClampBBoxes:
    @pytest.mark.parametrize(
        "metadata",
        [
            dict(),
            dict(format=datapoints.BBoxFormat.XYXY),
            dict(spatial_size=(1, 1)),
        ],
    )
    def test_simple_tensor_insufficient_metadata(self, metadata):
        simple_tensor = next(make_bboxes()).as_subclass(torch.Tensor)

        with pytest.raises(ValueError, match=re.escape("`format` and `spatial_size` has to be passed")):
            F.clamp_bboxes(simple_tensor, **metadata)

    @pytest.mark.parametrize(
        "metadata",
        [
            dict(format=datapoints.BBoxFormat.XYXY),
            dict(spatial_size=(1, 1)),
            dict(format=datapoints.BBoxFormat.XYXY, spatial_size=(1, 1)),
        ],
    )
    def test_datapoint_explicit_metadata(self, metadata):
        datapoint = next(make_bboxes())

        with pytest.raises(ValueError, match=re.escape("`format` and `spatial_size` must not be passed")):
            F.clamp_bboxes(datapoint, **metadata)


class TestConvertFormatBBoxes:
    @pytest.mark.parametrize(
        ("inpt", "old_format"),
        [
            (next(make_bboxes()), None),
            (next(make_bboxes()).as_subclass(torch.Tensor), datapoints.BBoxFormat.XYXY),
        ],
    )
    def test_missing_new_format(self, inpt, old_format):
        with pytest.raises(TypeError, match=re.escape("missing 1 required argument: 'new_format'")):
            F.convert_format_bboxes(inpt, old_format)

    def test_simple_tensor_insufficient_metadata(self):
        simple_tensor = next(make_bboxes()).as_subclass(torch.Tensor)

        with pytest.raises(ValueError, match=re.escape("`old_format` has to be passed")):
            F.convert_format_bboxes(simple_tensor, new_format=datapoints.BBoxFormat.CXCYWH)

    def test_datapoint_explicit_metadata(self):
        datapoint = next(make_bboxes())

        with pytest.raises(ValueError, match=re.escape("`old_format` must not be passed")):
            F.convert_format_bboxes(datapoint, old_format=datapoint.format, new_format=datapoints.BBoxFormat.CXCYWH)


# TODO: All correctness checks below this line should be ported to be references on a `KernelInfo` in
#  `transforms_v2_kernel_infos.py`


def _compute_affine_matrix(angle_, translate_, scale_, shear_, center_):
    rot = math.radians(angle_)
    cx, cy = center_
    tx, ty = translate_
    sx, sy = [math.radians(sh_) for sh_ in shear_]

    c_matrix = np.array([[1, 0, cx], [0, 1, cy], [0, 0, 1]])
    t_matrix = np.array([[1, 0, tx], [0, 1, ty], [0, 0, 1]])
    c_matrix_inv = np.linalg.inv(c_matrix)
    rs_matrix = np.array(
        [
            [scale_ * math.cos(rot), -scale_ * math.sin(rot), 0],
            [scale_ * math.sin(rot), scale_ * math.cos(rot), 0],
            [0, 0, 1],
        ]
    )
    shear_x_matrix = np.array([[1, -math.tan(sx), 0], [0, 1, 0], [0, 0, 1]])
    shear_y_matrix = np.array([[1, 0, 0], [-math.tan(sy), 1, 0], [0, 0, 1]])
    rss_matrix = np.matmul(rs_matrix, np.matmul(shear_y_matrix, shear_x_matrix))
    true_matrix = np.matmul(t_matrix, np.matmul(c_matrix, np.matmul(rss_matrix, c_matrix_inv)))
    return true_matrix


@pytest.mark.parametrize("device", cpu_and_cuda())
@pytest.mark.parametrize(
    "format",
    [datapoints.BBoxFormat.XYXY, datapoints.BBoxFormat.XYWH, datapoints.BBoxFormat.CXCYWH],
)
@pytest.mark.parametrize(
    "top, left, height, width, expected_bboxes",
    [
        [8, 12, 30, 40, [(-2.0, 7.0, 13.0, 27.0), (38.0, -3.0, 58.0, 14.0), (33.0, 38.0, 44.0, 54.0)]],
        [-8, 12, 70, 40, [(-2.0, 23.0, 13.0, 43.0), (38.0, 13.0, 58.0, 30.0), (33.0, 54.0, 44.0, 70.0)]],
    ],
)
def test_correctness_crop_bboxes(device, format, top, left, height, width, expected_bboxes):

    # Expected bboxes computed using Albumentations:
    # import numpy as np
    # from albumentations.augmentations.crops.functional import crop_bbox_by_coords, normalize_bbox, denormalize_bbox
    # expected_bboxes = []
    # for in_box in in_boxes:
    #     n_in_box = normalize_bbox(in_box, *size)
    #     n_out_box = crop_bbox_by_coords(
    #         n_in_box, (left, top, left + width, top + height), height, width, *size
    #     )
    #     out_box = denormalize_bbox(n_out_box, height, width)
    #     expected_bboxes.append(out_box)

    format = datapoints.BBoxFormat.XYXY
    spatial_size = (64, 76)
    in_boxes = [
        [10.0, 15.0, 25.0, 35.0],
        [50.0, 5.0, 70.0, 22.0],
        [45.0, 46.0, 56.0, 62.0],
    ]
    in_boxes = torch.tensor(in_boxes, device=device)
    if format != datapoints.BBoxFormat.XYXY:
        in_boxes = convert_format_bboxes(in_boxes, datapoints.BBoxFormat.XYXY, format)

    expected_bboxes = clamp_bboxes(
        datapoints.BBoxes(expected_bboxes, format="XYXY", spatial_size=spatial_size)
    ).tolist()

    output_boxes, output_spatial_size = F.crop_bboxes(
        in_boxes,
        format,
        top,
        left,
        spatial_size[0],
        spatial_size[1],
    )

    if format != datapoints.BBoxFormat.XYXY:
        output_boxes = convert_format_bboxes(output_boxes, format, datapoints.BBoxFormat.XYXY)

    torch.testing.assert_close(output_boxes.tolist(), expected_bboxes)
    torch.testing.assert_close(output_spatial_size, spatial_size)


@pytest.mark.parametrize("device", cpu_and_cuda())
def test_correctness_vertical_flip_segmentation_mask_on_fixed_input(device):
    mask = torch.zeros((3, 3, 3), dtype=torch.long, device=device)
    mask[:, 0, :] = 1

    out_mask = F.vertical_flip_mask(mask)

    expected_mask = torch.zeros((3, 3, 3), dtype=torch.long, device=device)
    expected_mask[:, -1, :] = 1
    torch.testing.assert_close(out_mask, expected_mask)


@pytest.mark.parametrize("device", cpu_and_cuda())
@pytest.mark.parametrize(
    "format",
    [datapoints.BBoxFormat.XYXY, datapoints.BBoxFormat.XYWH, datapoints.BBoxFormat.CXCYWH],
)
@pytest.mark.parametrize(
    "top, left, height, width, size",
    [
        [0, 0, 30, 30, (60, 60)],
        [-5, 5, 35, 45, (32, 34)],
    ],
)
def test_correctness_resized_crop_bboxes(device, format, top, left, height, width, size):
    def _compute_expected_bbox(bbox, top_, left_, height_, width_, size_):
        # bbox should be xyxy
        bbox[0] = (bbox[0] - left_) * size_[1] / width_
        bbox[1] = (bbox[1] - top_) * size_[0] / height_
        bbox[2] = (bbox[2] - left_) * size_[1] / width_
        bbox[3] = (bbox[3] - top_) * size_[0] / height_
        return bbox

    format = datapoints.BBoxFormat.XYXY
    spatial_size = (100, 100)
    in_boxes = [
        [10.0, 10.0, 20.0, 20.0],
        [5.0, 10.0, 15.0, 20.0],
    ]
    expected_bboxes = []
    for in_box in in_boxes:
        expected_bboxes.append(_compute_expected_bbox(list(in_box), top, left, height, width, size))
    expected_bboxes = torch.tensor(expected_bboxes, device=device)

    in_boxes = datapoints.BBoxes(in_boxes, format=datapoints.BBoxFormat.XYXY, spatial_size=spatial_size, device=device)
    if format != datapoints.BBoxFormat.XYXY:
        in_boxes = convert_format_bboxes(in_boxes, datapoints.BBoxFormat.XYXY, format)

    output_boxes, output_spatial_size = F.resized_crop_bboxes(in_boxes, format, top, left, height, width, size)

    if format != datapoints.BBoxFormat.XYXY:
        output_boxes = convert_format_bboxes(output_boxes, format, datapoints.BBoxFormat.XYXY)

    torch.testing.assert_close(output_boxes, expected_bboxes)
    torch.testing.assert_close(output_spatial_size, size)


def _parse_padding(padding):
    if isinstance(padding, int):
        return [padding] * 4
    if isinstance(padding, list):
        if len(padding) == 1:
            return padding * 4
        if len(padding) == 2:
            return padding * 2  # [left, up, right, down]

    return padding


@pytest.mark.parametrize("device", cpu_and_cuda())
@pytest.mark.parametrize("padding", [[1], [1, 1], [1, 1, 2, 2]])
def test_correctness_pad_bboxes(device, padding):
    def _compute_expected_bbox(bbox, padding_):
        pad_left, pad_up, _, _ = _parse_padding(padding_)

        dtype = bbox.dtype
        format = bbox.format
        bbox = (
            bbox.clone()
            if format == datapoints.BBoxFormat.XYXY
            else convert_format_bboxes(bbox, new_format=datapoints.BBoxFormat.XYXY)
        )

        bbox[0::2] += pad_left
        bbox[1::2] += pad_up

        bbox = convert_format_bboxes(bbox, new_format=format)
        if bbox.dtype != dtype:
            # Temporary cast to original dtype
            # e.g. float32 -> int
            bbox = bbox.to(dtype)
        return bbox

    def _compute_expected_spatial_size(bbox, padding_):
        pad_left, pad_up, pad_right, pad_down = _parse_padding(padding_)
        height, width = bbox.spatial_size
        return height + pad_up + pad_down, width + pad_left + pad_right

    for bboxes in make_bboxes():
        bboxes = bboxes.to(device)
        bboxes_format = bboxes.format
        bboxes_spatial_size = bboxes.spatial_size

        output_boxes, output_spatial_size = F.pad_bboxes(
            bboxes, format=bboxes_format, spatial_size=bboxes_spatial_size, padding=padding
        )

        torch.testing.assert_close(output_spatial_size, _compute_expected_spatial_size(bboxes, padding))

        if bboxes.ndim < 2 or bboxes.shape[0] == 0:
            bboxes = [bboxes]

        expected_bboxes = []
        for bbox in bboxes:
            bbox = datapoints.BBoxes(bbox, format=bboxes_format, spatial_size=bboxes_spatial_size)
            expected_bboxes.append(_compute_expected_bbox(bbox, padding))

        if len(expected_bboxes) > 1:
            expected_bboxes = torch.stack(expected_bboxes)
        else:
            expected_bboxes = expected_bboxes[0]
        torch.testing.assert_close(output_boxes, expected_bboxes, atol=1, rtol=0)


@pytest.mark.parametrize("device", cpu_and_cuda())
def test_correctness_pad_segmentation_mask_on_fixed_input(device):
    mask = torch.ones((1, 3, 3), dtype=torch.long, device=device)

    out_mask = F.pad_mask(mask, padding=[1, 1, 1, 1])

    expected_mask = torch.zeros((1, 5, 5), dtype=torch.long, device=device)
    expected_mask[:, 1:-1, 1:-1] = 1
    torch.testing.assert_close(out_mask, expected_mask)


@pytest.mark.parametrize("device", cpu_and_cuda())
@pytest.mark.parametrize(
    "startpoints, endpoints",
    [
        [[[0, 0], [33, 0], [33, 25], [0, 25]], [[3, 2], [32, 3], [30, 24], [2, 25]]],
        [[[3, 2], [32, 3], [30, 24], [2, 25]], [[0, 0], [33, 0], [33, 25], [0, 25]]],
        [[[3, 2], [32, 3], [30, 24], [2, 25]], [[5, 5], [30, 3], [33, 19], [4, 25]]],
    ],
)
def test_correctness_perspective_bboxes(device, startpoints, endpoints):
    def _compute_expected_bbox(bbox, pcoeffs_):
        m1 = np.array(
            [
                [pcoeffs_[0], pcoeffs_[1], pcoeffs_[2]],
                [pcoeffs_[3], pcoeffs_[4], pcoeffs_[5]],
            ]
        )
        m2 = np.array(
            [
                [pcoeffs_[6], pcoeffs_[7], 1.0],
                [pcoeffs_[6], pcoeffs_[7], 1.0],
            ]
        )

        bbox_xyxy = convert_format_bboxes(bbox, new_format=datapoints.BBoxFormat.XYXY)
        points = np.array(
            [
                [bbox_xyxy[0].item(), bbox_xyxy[1].item(), 1.0],
                [bbox_xyxy[2].item(), bbox_xyxy[1].item(), 1.0],
                [bbox_xyxy[0].item(), bbox_xyxy[3].item(), 1.0],
                [bbox_xyxy[2].item(), bbox_xyxy[3].item(), 1.0],
            ]
        )
        numer = np.matmul(points, m1.T)
        denom = np.matmul(points, m2.T)
        transformed_points = numer / denom
        out_bbox = np.array(
            [
                np.min(transformed_points[:, 0]),
                np.min(transformed_points[:, 1]),
                np.max(transformed_points[:, 0]),
                np.max(transformed_points[:, 1]),
            ]
        )
        out_bbox = datapoints.BBoxes(
            out_bbox,
            format=datapoints.BBoxFormat.XYXY,
            spatial_size=bbox.spatial_size,
            dtype=bbox.dtype,
            device=bbox.device,
        )
        return clamp_bboxes(convert_format_bboxes(out_bbox, new_format=bbox.format))

    spatial_size = (32, 38)

    pcoeffs = _get_perspective_coeffs(startpoints, endpoints)
    inv_pcoeffs = _get_perspective_coeffs(endpoints, startpoints)

    for bboxes in make_bboxes(spatial_size=spatial_size, extra_dims=((4,),)):
        bboxes = bboxes.to(device)

        output_bboxes = F.perspective_bboxes(
            bboxes.as_subclass(torch.Tensor),
            format=bboxes.format,
            spatial_size=bboxes.spatial_size,
            startpoints=None,
            endpoints=None,
            coefficients=pcoeffs,
        )

        if bboxes.ndim < 2:
            bboxes = [bboxes]

        expected_bboxes = []
        for bbox in bboxes:
            bbox = datapoints.BBoxes(bbox, format=bboxes.format, spatial_size=bboxes.spatial_size)
            expected_bboxes.append(_compute_expected_bbox(bbox, inv_pcoeffs))
        if len(expected_bboxes) > 1:
            expected_bboxes = torch.stack(expected_bboxes)
        else:
            expected_bboxes = expected_bboxes[0]
        torch.testing.assert_close(output_bboxes, expected_bboxes, rtol=0, atol=1)


@pytest.mark.parametrize("device", cpu_and_cuda())
@pytest.mark.parametrize(
    "output_size",
    [(18, 18), [18, 15], (16, 19), [12], [46, 48]],
)
def test_correctness_center_crop_bboxes(device, output_size):
    def _compute_expected_bbox(bbox, output_size_):
        format_ = bbox.format
        spatial_size_ = bbox.spatial_size
        dtype = bbox.dtype
        bbox = convert_format_bboxes(bbox.float(), format_, datapoints.BBoxFormat.XYWH)

        if len(output_size_) == 1:
            output_size_.append(output_size_[-1])

        cy = int(round((spatial_size_[0] - output_size_[0]) * 0.5))
        cx = int(round((spatial_size_[1] - output_size_[1]) * 0.5))
        out_bbox = [
            bbox[0].item() - cx,
            bbox[1].item() - cy,
            bbox[2].item(),
            bbox[3].item(),
        ]
        out_bbox = torch.tensor(out_bbox)
        out_bbox = convert_format_bboxes(out_bbox, datapoints.BBoxFormat.XYWH, format_)
        out_bbox = clamp_bboxes(out_bbox, format=format_, spatial_size=output_size)
        return out_bbox.to(dtype=dtype, device=bbox.device)

    for bboxes in make_bboxes(extra_dims=((4,),)):
        bboxes = bboxes.to(device)
        bboxes_format = bboxes.format
        bboxes_spatial_size = bboxes.spatial_size

        output_boxes, output_spatial_size = F.center_crop_bboxes(
            bboxes, bboxes_format, bboxes_spatial_size, output_size
        )

        if bboxes.ndim < 2:
            bboxes = [bboxes]

        expected_bboxes = []
        for bbox in bboxes:
            bbox = datapoints.BBoxes(bbox, format=bboxes_format, spatial_size=bboxes_spatial_size)
            expected_bboxes.append(_compute_expected_bbox(bbox, output_size))

        if len(expected_bboxes) > 1:
            expected_bboxes = torch.stack(expected_bboxes)
        else:
            expected_bboxes = expected_bboxes[0]

        torch.testing.assert_close(output_boxes, expected_bboxes, atol=1, rtol=0)
        torch.testing.assert_close(output_spatial_size, output_size)


@pytest.mark.parametrize("device", cpu_and_cuda())
@pytest.mark.parametrize("output_size", [[4, 2], [4], [7, 6]])
def test_correctness_center_crop_mask(device, output_size):
    def _compute_expected_mask(mask, output_size):
        crop_height, crop_width = output_size if len(output_size) > 1 else [output_size[0], output_size[0]]

        _, image_height, image_width = mask.shape
        if crop_width > image_height or crop_height > image_width:
            padding = _center_crop_compute_padding(crop_height, crop_width, image_height, image_width)
            mask = F.pad_image_tensor(mask, padding, fill=0)

        left = round((image_width - crop_width) * 0.5)
        top = round((image_height - crop_height) * 0.5)

        return mask[:, top : top + crop_height, left : left + crop_width]

    mask = torch.randint(0, 2, size=(1, 6, 6), dtype=torch.long, device=device)
    actual = F.center_crop_mask(mask, output_size)

    expected = _compute_expected_mask(mask, output_size)
    torch.testing.assert_close(expected, actual)


# Copied from test/test_functional_tensor.py
@pytest.mark.parametrize("device", cpu_and_cuda())
@pytest.mark.parametrize("spatial_size", ("small", "large"))
@pytest.mark.parametrize("dt", [None, torch.float32, torch.float64, torch.float16])
@pytest.mark.parametrize("ksize", [(3, 3), [3, 5], (23, 23)])
@pytest.mark.parametrize("sigma", [[0.5, 0.5], (0.5, 0.5), (0.8, 0.8), (1.7, 1.7)])
def test_correctness_gaussian_blur_image_tensor(device, spatial_size, dt, ksize, sigma):
    fn = F.gaussian_blur_image_tensor

    # true_cv2_results = {
    #     # np_img = np.arange(3 * 10 * 12, dtype="uint8").reshape((10, 12, 3))
    #     # cv2.GaussianBlur(np_img, ksize=(3, 3), sigmaX=0.8)
    #     "3_3_0.8": ...
    #     # cv2.GaussianBlur(np_img, ksize=(3, 3), sigmaX=0.5)
    #     "3_3_0.5": ...
    #     # cv2.GaussianBlur(np_img, ksize=(3, 5), sigmaX=0.8)
    #     "3_5_0.8": ...
    #     # cv2.GaussianBlur(np_img, ksize=(3, 5), sigmaX=0.5)
    #     "3_5_0.5": ...
    #     # np_img2 = np.arange(26 * 28, dtype="uint8").reshape((26, 28))
    #     # cv2.GaussianBlur(np_img2, ksize=(23, 23), sigmaX=1.7)
    #     "23_23_1.7": ...
    # }
    p = os.path.join(os.path.dirname(os.path.abspath(__file__)), "assets", "gaussian_blur_opencv_results.pt")
    true_cv2_results = torch.load(p)

    if spatial_size == "small":
        tensor = (
            torch.from_numpy(np.arange(3 * 10 * 12, dtype="uint8").reshape((10, 12, 3))).permute(2, 0, 1).to(device)
        )
    else:
        tensor = torch.from_numpy(np.arange(26 * 28, dtype="uint8").reshape((1, 26, 28))).to(device)

    if dt == torch.float16 and device == "cpu":
        # skip float16 on CPU case
        return

    if dt is not None:
        tensor = tensor.to(dtype=dt)

    _ksize = (ksize, ksize) if isinstance(ksize, int) else ksize
    _sigma = sigma[0] if sigma is not None else None
    shape = tensor.shape
    gt_key = f"{shape[-2]}_{shape[-1]}_{shape[-3]}__{_ksize[0]}_{_ksize[1]}_{_sigma}"
    if gt_key not in true_cv2_results:
        return

    true_out = (
        torch.tensor(true_cv2_results[gt_key]).reshape(shape[-2], shape[-1], shape[-3]).permute(2, 0, 1).to(tensor)
    )

    image = datapoints.Image(tensor)

    out = fn(image, kernel_size=ksize, sigma=sigma)
    torch.testing.assert_close(out, true_out, rtol=0.0, atol=1.0, msg=f"{ksize}, {sigma}")


@pytest.mark.parametrize(
    "inpt",
    [
        127 * np.ones((32, 32, 3), dtype="uint8"),
        PIL.Image.new("RGB", (32, 32), 122),
    ],
)
def test_to_image_tensor(inpt):
    output = F.to_image_tensor(inpt)
    assert isinstance(output, torch.Tensor)
    assert output.shape == (3, 32, 32)

    assert np.asarray(inpt).sum() == output.sum().item()


@pytest.mark.parametrize(
    "inpt",
    [
        torch.randint(0, 256, size=(3, 32, 32), dtype=torch.uint8),
        127 * np.ones((32, 32, 3), dtype="uint8"),
    ],
)
@pytest.mark.parametrize("mode", [None, "RGB"])
def test_to_image_pil(inpt, mode):
    output = F.to_image_pil(inpt, mode=mode)
    assert isinstance(output, PIL.Image.Image)

    assert np.asarray(inpt).sum() == np.asarray(output).sum()


def test_equalize_image_tensor_edge_cases():
    inpt = torch.zeros(3, 200, 200, dtype=torch.uint8)
    output = F.equalize_image_tensor(inpt)
    torch.testing.assert_close(inpt, output)

    inpt = torch.zeros(5, 3, 200, 200, dtype=torch.uint8)
    inpt[..., 100:, 100:] = 1
    output = F.equalize_image_tensor(inpt)
    assert output.unique().tolist() == [0, 255]


@pytest.mark.parametrize("device", cpu_and_cuda())
def test_correctness_uniform_temporal_subsample(device):
    video = torch.arange(10, device=device)[:, None, None, None].expand(-1, 3, 8, 8)
    out_video = F.uniform_temporal_subsample(video, 5)
    assert out_video.unique().tolist() == [0, 2, 4, 6, 9]

    out_video = F.uniform_temporal_subsample(video, 8)
    assert out_video.unique().tolist() == [0, 1, 2, 3, 5, 6, 7, 9]


# TODO: We can remove this test and related torchvision workaround
# once we fixed related pytorch issue: https://github.com/pytorch/pytorch/issues/68430
@make_info_args_kwargs_parametrization(
    [info for info in KERNEL_INFOS if info.kernel is F.resize_image_tensor],
    args_kwargs_fn=lambda info: info.reference_inputs_fn(),
)
def test_memory_format_consistency_resize_image_tensor(test_id, info, args_kwargs):
    (input, *other_args), kwargs = args_kwargs.load("cpu")

    output = info.kernel(input.as_subclass(torch.Tensor), *other_args, **kwargs)

    error_msg_fn = parametrized_error_message(input, *other_args, **kwargs)
    assert input.ndim == 3, error_msg_fn
    input_stride = input.stride()
    output_stride = output.stride()
    # Here we check output memory format according to the input:
    # if input_stride is (..., 1) then input is most likely channels first and thus
    #   output strides should match channels first strides (H * W, H, 1)
    # if input_stride is (1, ...) then input is most likely channels last and thus
    #   output strides should match channels last strides (1, W * C, C)
    if input_stride[-1] == 1:
        expected_stride = (output.shape[-2] * output.shape[-1], output.shape[-1], 1)
        assert expected_stride == output_stride, error_msg_fn("")
    elif input_stride[0] == 1:
        expected_stride = (1, output.shape[0] * output.shape[-1], output.shape[0])
        assert expected_stride == output_stride, error_msg_fn("")
    else:
        assert False, error_msg_fn("")


def test_resize_float16_no_rounding():
    # Make sure Resize() doesn't round float16 images
    # Non-regression test for https://github.com/pytorch/vision/issues/7667

    img = torch.randint(0, 256, size=(1, 3, 100, 100), dtype=torch.float16)
    out = F.resize(img, size=(10, 10))
    assert out.dtype == torch.float16
    assert (out.round() - out).sum() > 0
