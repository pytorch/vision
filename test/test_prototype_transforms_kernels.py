import pytest

import torch.testing
from common_utils import cpu_and_gpu, needs_cuda, script
from prototype_common_utils import assert_close
from prototype_transforms_kernel_infos import KERNEL_INFOS
from torch.utils._pytree import tree_map
from torchvision._utils import sequence_to_str
from torchvision.prototype import features
from torchvision.prototype.transforms import functional as F


def test_coverage():
    tested = {info.kernel_name for info in KERNEL_INFOS}
    exposed = {
        name
        for name, kernel in F.__dict__.items()
        if callable(kernel)
        and any(
            name.endswith(f"_{feature_name}")
            for feature_name in {
                "bounding_box",
                "image_tensor",
                "label",
                "mask",
            }
        )
        and name not in {"to_image_tensor"}
        # TODO: The list below should be quickly reduced in the transition period. There is nothing that prevents us
        #  from adding `KernelInfo`'s for these kernels other than time.
        and name
        not in {
            "adjust_brightness_image_tensor",
            "adjust_contrast_image_tensor",
            "adjust_gamma_image_tensor",
            "adjust_hue_image_tensor",
            "adjust_saturation_image_tensor",
            "clamp_bounding_box",
            "erase_image_tensor",
            "five_crop_image_tensor",
            "normalize_image_tensor",
            "ten_crop_image_tensor",
        }
    }

    needlessly_ignored = tested - exposed
    if needlessly_ignored:
        raise pytest.UsageError(
            f"The kernel(s) {sequence_to_str(sorted(needlessly_ignored), separate_last='and ')} "
            f"have an associated `KernelInfo` but are ignored by this test."
        )

    untested = exposed - tested
    if untested:
        raise AssertionError(
            f"The kernel(s) {sequence_to_str(sorted(untested), separate_last='and ')} "
            f"are exposed through `torchvision.prototype.transforms.functional`, but are not tested. "
            f"Please add a `KernelInfo` to the `KERNEL_INFOS` list in `test/prototype_transforms_kernel_infos.py`."
        )


class TestCommon:
    sample_inputs = pytest.mark.parametrize(
        ("info", "args_kwargs"),
        [
            pytest.param(info, args_kwargs, id=f"{info.kernel_name}-{idx}")
            for info in KERNEL_INFOS
            for idx, args_kwargs in enumerate(info.sample_inputs_fn())
        ],
    )

    @sample_inputs
    @pytest.mark.parametrize("device", cpu_and_gpu())
    def test_scripted_vs_eager(self, info, args_kwargs, device):
        kernel_eager = info.kernel
        kernel_scripted = script(kernel_eager)

        args, kwargs = args_kwargs.load(device)

        actual = kernel_scripted(*args, **kwargs)
        expected = kernel_eager(*args, **kwargs)

        assert_close(actual, expected, **info.closeness_kwargs)

    @sample_inputs
    @pytest.mark.parametrize("device", cpu_and_gpu())
    def test_batched_vs_single(self, info, args_kwargs, device):
        def unbind_batch_dims(batched_tensor, *, data_dims):
            if batched_tensor.ndim == data_dims:
                return batched_tensor

            return [unbind_batch_dims(t, data_dims=data_dims) for t in batched_tensor.unbind(0)]

        def stack_batch_dims(unbound_tensor):
            if isinstance(unbound_tensor[0], torch.Tensor):
                return torch.stack(unbound_tensor)

            return torch.stack([stack_batch_dims(t) for t in unbound_tensor])

        (batched_input, *other_args), kwargs = args_kwargs.load(device)

        feature_type = features.Image if features.is_simple_tensor(batched_input) else type(batched_input)
        # This dictionary contains the number of rightmost dimensions that contain the actual data.
        # Everything to the left is considered a batch dimension.
        data_dims = {
            features.Image: 3,
            features.BoundingBox: 1,
            # `Mask`'s are special in the sense that the data dimensions depend on the type of mask. For detection masks
            # it is 3 `(*, N, H, W)`, but for segmentation masks it is 2 `(*, H, W)`. Since both a grouped under one
            # type all kernels should also work without differentiating between the two. Thus, we go with 2 here as
            # common ground.
            features.Mask: 2,
        }.get(feature_type)
        if data_dims is None:
            raise pytest.UsageError(
                f"The number of data dimensions cannot be determined for input of type {feature_type.__name__}."
            ) from None
        elif batched_input.ndim <= data_dims:
            pytest.skip("Input is not batched.")
        elif not all(batched_input.shape[:-data_dims]):
            pytest.skip("Input has a degenerate batch shape.")

        actual = info.kernel(batched_input, *other_args, **kwargs)

        single_inputs = unbind_batch_dims(batched_input, data_dims=data_dims)
        single_outputs = tree_map(lambda single_input: info.kernel(single_input, *other_args, **kwargs), single_inputs)
        expected = stack_batch_dims(single_outputs)

        assert_close(actual, expected, **info.closeness_kwargs)

    @sample_inputs
    @pytest.mark.parametrize("device", cpu_and_gpu())
    def test_no_inplace(self, info, args_kwargs, device):
        (input, *other_args), kwargs = args_kwargs.load(device)

        if input.numel() == 0:
            pytest.skip("The input has a degenerate shape.")

        input_version = input._version
        output = info.kernel(input, *other_args, **kwargs)

        assert output is not input or output._version == input_version

    @sample_inputs
    @needs_cuda
    def test_cuda_vs_cpu(self, info, args_kwargs):
        (input_cpu, *other_args), kwargs = args_kwargs.load("cpu")
        input_cuda = input_cpu.to("cuda")

        output_cpu = info.kernel(input_cpu, *other_args, **kwargs)
        output_cuda = info.kernel(input_cuda, *other_args, **kwargs)

        assert_close(output_cuda, output_cpu, check_device=False, **info.closeness_kwargs)

    @sample_inputs
    @pytest.mark.parametrize("device", cpu_and_gpu())
    def test_dtype_and_device_consistency(self, info, args_kwargs, device):
        (input, *other_args), kwargs = args_kwargs.load(device)

        output = info.kernel(input, *other_args, **kwargs)

        assert output.dtype == input.dtype
        assert output.device == torch.device(device)

    @pytest.mark.parametrize(
        ("info", "args_kwargs"),
        [
            pytest.param(info, args_kwargs, id=f"{info.kernel_name}-{idx}")
            for info in KERNEL_INFOS
            for idx, args_kwargs in enumerate(info.reference_inputs_fn())
            if info.reference_fn is not None
        ],
    )
    def test_against_reference(self, info, args_kwargs):
        args, kwargs = args_kwargs.load("cpu")

        actual = info.kernel(*args, **kwargs)
        expected = info.reference_fn(*args, **kwargs)

        assert_close(actual, expected, check_dtype=False, **info.closeness_kwargs)
