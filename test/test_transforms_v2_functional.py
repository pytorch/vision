import inspect
import re

import numpy as np
import PIL.Image
import pytest
import torch

from common_utils import assert_close, cache, cpu_and_cuda, needs_cuda, set_rng_seed
from torch.utils._pytree import tree_map
from torchvision import tv_tensors
from torchvision.transforms.v2 import functional as F
from torchvision.transforms.v2._utils import is_pure_tensor
from transforms_v2_dispatcher_infos import DISPATCHER_INFOS
from transforms_v2_kernel_infos import KERNEL_INFOS
from transforms_v2_legacy_utils import (
    DEFAULT_SQUARE_SPATIAL_SIZE,
    make_multiple_bounding_boxes,
    parametrized_error_message,
)


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

        tv_tensor_type = tv_tensors.Image if is_pure_tensor(batched_input) else type(batched_input)
        # This dictionary contains the number of rightmost dimensions that contain the actual data.
        # Everything to the left is considered a batch dimension.
        data_dims = {
            tv_tensors.Image: 3,
            tv_tensors.BoundingBoxes: 1,
            # `Mask`'s are special in the sense that the data dimensions depend on the type of mask. For detection masks
            # it is 3 `(*, N, H, W)`, but for segmentation masks it is 2 `(*, H, W)`. Since both a grouped under one
            # type all kernels should also work without differentiating between the two. Thus, we go with 2 here as
            # common ground.
            tv_tensors.Mask: 2,
            tv_tensors.Video: 4,
        }.get(tv_tensor_type)
        if data_dims is None:
            raise pytest.UsageError(
                f"The number of data dimensions cannot be determined for input of type {tv_tensor_type.__name__}."
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
            F.to_dtype_image(input, dtype=torch.float32, scale=True),
            *adapted_other_args,
            **adapted_kwargs,
        )

        expected = F.to_dtype_image(info.kernel(input, *other_args, **kwargs), dtype=torch.float32, scale=True)

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
        [info for info in DISPATCHER_INFOS if tv_tensors.Image in info.kernels],
        args_kwargs_fn=lambda info: info.sample_inputs(tv_tensors.Image),
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

        (image_tv_tensor, *other_args), kwargs = args_kwargs.load(device)
        image_pure_tensor = torch.Tensor(image_tv_tensor)

        dispatcher(image_pure_tensor, *other_args, **kwargs)

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
            F.get_size,
            F.rgb_to_grayscale,
            F.uniform_temporal_subsample,
        ],
        ids=lambda dispatcher: dispatcher.__name__,
    )
    def test_scriptable(self, dispatcher):
        script(dispatcher)

    @image_sample_inputs
    def test_pure_tensor_output_type(self, info, args_kwargs):
        (image_tv_tensor, *other_args), kwargs = args_kwargs.load()
        image_pure_tensor = image_tv_tensor.as_subclass(torch.Tensor)

        output = info.dispatcher(image_pure_tensor, *other_args, **kwargs)

        # We cannot use `isinstance` here since all tv_tensors are instances of `torch.Tensor` as well
        assert type(output) is torch.Tensor

    @make_info_args_kwargs_parametrization(
        [info for info in DISPATCHER_INFOS if info.pil_kernel_info is not None],
        args_kwargs_fn=lambda info: info.sample_inputs(tv_tensors.Image),
    )
    def test_pil_output_type(self, info, args_kwargs):
        (image_tv_tensor, *other_args), kwargs = args_kwargs.load()

        if image_tv_tensor.ndim > 3:
            pytest.skip("Input is batched")

        image_pil = F.to_pil_image(image_tv_tensor)

        output = info.dispatcher(image_pil, *other_args, **kwargs)

        assert isinstance(output, PIL.Image.Image)

    @make_info_args_kwargs_parametrization(
        DISPATCHER_INFOS,
        args_kwargs_fn=lambda info: info.sample_inputs(),
    )
    def test_tv_tensor_output_type(self, info, args_kwargs):
        (tv_tensor, *other_args), kwargs = args_kwargs.load()

        output = info.dispatcher(tv_tensor, *other_args, **kwargs)

        assert isinstance(output, type(tv_tensor))

        if isinstance(tv_tensor, tv_tensors.BoundingBoxes) and info.dispatcher is not F.convert_bounding_box_format:
            assert output.format == tv_tensor.format

    @pytest.mark.parametrize(
        ("dispatcher_info", "tv_tensor_type", "kernel_info"),
        [
            pytest.param(
                dispatcher_info, tv_tensor_type, kernel_info, id=f"{dispatcher_info.id}-{tv_tensor_type.__name__}"
            )
            for dispatcher_info in DISPATCHER_INFOS
            for tv_tensor_type, kernel_info in dispatcher_info.kernel_infos.items()
        ],
    )
    def test_dispatcher_kernel_signatures_consistency(self, dispatcher_info, tv_tensor_type, kernel_info):
        dispatcher_signature = inspect.signature(dispatcher_info.dispatcher)
        dispatcher_params = list(dispatcher_signature.parameters.values())[1:]

        kernel_signature = inspect.signature(kernel_info.kernel)
        kernel_params = list(kernel_signature.parameters.values())[1:]

        # We filter out metadata that is implicitly passed to the dispatcher through the input tv_tensor, but has to be
        # explicitly passed to the kernel.
        input_type = {v: k for k, v in dispatcher_info.kernels.items()}.get(kernel_info.kernel)
        explicit_metadata = {
            tv_tensors.BoundingBoxes: {"format", "canvas_size"},
        }
        kernel_params = [param for param in kernel_params if param.name not in explicit_metadata.get(input_type, set())]

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
    def test_unkown_type(self, info):
        unkown_input = object()
        (_, *other_args), kwargs = next(iter(info.sample_inputs())).load("cpu")

        with pytest.raises(TypeError, match=re.escape(str(type(unkown_input)))):
            info.dispatcher(unkown_input, *other_args, **kwargs)

    @make_info_args_kwargs_parametrization(
        [
            info
            for info in DISPATCHER_INFOS
            if tv_tensors.BoundingBoxes in info.kernels and info.dispatcher is not F.convert_bounding_box_format
        ],
        args_kwargs_fn=lambda info: info.sample_inputs(tv_tensors.BoundingBoxes),
    )
    def test_bounding_boxes_format_consistency(self, info, args_kwargs):
        (bounding_boxes, *other_args), kwargs = args_kwargs.load()
        format = bounding_boxes.format

        output = info.dispatcher(bounding_boxes, *other_args, **kwargs)

        assert output.format == format


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

    assert_samples_from_standard_normal(F.normalize_image(image, mean, std))


class TestClampBoundingBoxes:
    @pytest.mark.parametrize(
        "metadata",
        [
            dict(),
            dict(format=tv_tensors.BoundingBoxFormat.XYXY),
            dict(canvas_size=(1, 1)),
        ],
    )
    def test_pure_tensor_insufficient_metadata(self, metadata):
        pure_tensor = next(make_multiple_bounding_boxes()).as_subclass(torch.Tensor)

        with pytest.raises(ValueError, match=re.escape("`format` and `canvas_size` has to be passed")):
            F.clamp_bounding_boxes(pure_tensor, **metadata)

    @pytest.mark.parametrize(
        "metadata",
        [
            dict(format=tv_tensors.BoundingBoxFormat.XYXY),
            dict(canvas_size=(1, 1)),
            dict(format=tv_tensors.BoundingBoxFormat.XYXY, canvas_size=(1, 1)),
        ],
    )
    def test_tv_tensor_explicit_metadata(self, metadata):
        tv_tensor = next(make_multiple_bounding_boxes())

        with pytest.raises(ValueError, match=re.escape("`format` and `canvas_size` must not be passed")):
            F.clamp_bounding_boxes(tv_tensor, **metadata)


# TODO: All correctness checks below this line should be ported to be references on a `KernelInfo` in
#  `transforms_v2_kernel_infos.py`


@pytest.mark.parametrize(
    "inpt",
    [
        127 * np.ones((32, 32, 3), dtype="uint8"),
        PIL.Image.new("RGB", (32, 32), 122),
    ],
)
def test_to_image(inpt):
    output = F.to_image(inpt)
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
def test_to_pil_image(inpt, mode):
    output = F.to_pil_image(inpt, mode=mode)
    assert isinstance(output, PIL.Image.Image)

    assert np.asarray(inpt).sum() == np.asarray(output).sum()


@pytest.mark.parametrize("device", cpu_and_cuda())
def test_correctness_uniform_temporal_subsample(device):
    video = torch.arange(10, device=device)[:, None, None, None].expand(-1, 3, 8, 8)
    out_video = F.uniform_temporal_subsample(video, 5)
    assert out_video.unique().tolist() == [0, 2, 4, 6, 9]

    out_video = F.uniform_temporal_subsample(video, 8)
    assert out_video.unique().tolist() == [0, 1, 2, 3, 5, 6, 7, 9]
