import functools
import itertools

import PIL.Image
import pytest
import torch.testing
import torchvision.transforms.v2.functional as F
from transforms_v2_legacy_utils import (
    ArgsKwargs,
    InfoBase,
    make_image_loaders,
    make_video_loaders,
    mark_framework_limitation,
    TestMark,
)

__all__ = ["KernelInfo", "KERNEL_INFOS"]


class KernelInfo(InfoBase):
    def __init__(
        self,
        kernel,
        *,
        # Defaults to `kernel.__name__`. Should be set if the function is exposed under a different name
        # TODO: This can probably be removed after roll-out since we shouldn't have any aliasing then
        kernel_name=None,
        # Most common tests use these inputs to check the kernel. As such it should cover all valid code paths, but
        # should not include extensive parameter combinations to keep to overall test count moderate.
        sample_inputs_fn,
        # This function should mirror the kernel. It should have the same signature as the `kernel` and as such also
        # take tensors as inputs. Any conversion into another object type, e.g. PIL images or numpy arrays, should
        # happen inside the function. It should return a tensor or to be more precise an object that can be compared to
        # a tensor by `assert_close`. If omitted, no reference test will be performed.
        reference_fn=None,
        # These inputs are only used for the reference tests and thus can be comprehensive with regard to the parameter
        # values to be tested. If not specified, `sample_inputs_fn` will be used.
        reference_inputs_fn=None,
        # If true-ish, triggers a test that checks the kernel for consistency between uint8 and float32 inputs with the
        # reference inputs. This is usually used whenever we use a PIL kernel as reference.
        # Can be a callable in which case it will be called with `other_args, kwargs`. It should return the same
        # structure, but with adapted parameters. This is useful in case a parameter value is closely tied to the input
        # dtype.
        float32_vs_uint8=False,
        # Some kernels don't have dispatchers that would handle logging the usage. Thus, the kernel has to do it
        # manually. If set, triggers a test that makes sure this happens.
        logs_usage=False,
        # See InfoBase
        test_marks=None,
        # See InfoBase
        closeness_kwargs=None,
    ):
        super().__init__(id=kernel_name or kernel.__name__, test_marks=test_marks, closeness_kwargs=closeness_kwargs)
        self.kernel = kernel
        self.sample_inputs_fn = sample_inputs_fn
        self.reference_fn = reference_fn
        self.reference_inputs_fn = reference_inputs_fn

        if float32_vs_uint8 and not callable(float32_vs_uint8):
            float32_vs_uint8 = lambda other_args, kwargs: (other_args, kwargs)  # noqa: E731
        self.float32_vs_uint8 = float32_vs_uint8
        self.logs_usage = logs_usage


def pil_reference_wrapper(pil_kernel):
    @functools.wraps(pil_kernel)
    def wrapper(input_tensor, *other_args, **kwargs):
        if input_tensor.dtype != torch.uint8:
            raise pytest.UsageError(f"Can only test uint8 tensor images against PIL, but input is {input_tensor.dtype}")
        if input_tensor.ndim > 3:
            raise pytest.UsageError(
                f"Can only test single tensor images against PIL, but input has shape {input_tensor.shape}"
            )

        input_pil = F.to_pil_image(input_tensor)
        output_pil = pil_kernel(input_pil, *other_args, **kwargs)
        if not isinstance(output_pil, PIL.Image.Image):
            return output_pil

        output_tensor = F.to_image(output_pil)

        # 2D mask shenanigans
        if output_tensor.ndim == 2 and input_tensor.ndim == 3:
            output_tensor = output_tensor.unsqueeze(0)
        elif output_tensor.ndim == 3 and input_tensor.ndim == 2:
            output_tensor = output_tensor.squeeze(0)

        return output_tensor

    return wrapper


def xfail_jit(reason, *, condition=None):
    return TestMark(("TestKernels", "test_scripted_vs_eager"), pytest.mark.xfail(reason=reason), condition=condition)


def xfail_jit_python_scalar_arg(name, *, reason=None):
    return xfail_jit(
        reason or f"Python scalar int or float for `{name}` is not supported when scripting",
        condition=lambda args_kwargs: isinstance(args_kwargs.kwargs.get(name), (int, float)),
    )


KERNEL_INFOS = []


_FIVE_TEN_CROP_SIZES = [7, (6,), [5], (6, 5), [7, 6]]


def _get_five_ten_crop_canvas_size(size):
    if isinstance(size, int):
        crop_height = crop_width = size
    elif len(size) == 1:
        crop_height = crop_width = size[0]
    else:
        crop_height, crop_width = size
    return 2 * crop_height, 2 * crop_width


def sample_inputs_five_crop_image_tensor():
    for size in _FIVE_TEN_CROP_SIZES:
        for image_loader in make_image_loaders(
            sizes=[_get_five_ten_crop_canvas_size(size)],
            color_spaces=["RGB"],
            dtypes=[torch.float32],
        ):
            yield ArgsKwargs(image_loader, size=size)


def reference_inputs_five_crop_image_tensor():
    for size in _FIVE_TEN_CROP_SIZES:
        for image_loader in make_image_loaders(
            sizes=[_get_five_ten_crop_canvas_size(size)], extra_dims=[()], dtypes=[torch.uint8]
        ):
            yield ArgsKwargs(image_loader, size=size)


def sample_inputs_five_crop_video():
    size = _FIVE_TEN_CROP_SIZES[0]
    for video_loader in make_video_loaders(sizes=[_get_five_ten_crop_canvas_size(size)]):
        yield ArgsKwargs(video_loader, size=size)


def sample_inputs_ten_crop_image_tensor():
    for size, vertical_flip in itertools.product(_FIVE_TEN_CROP_SIZES, [False, True]):
        for image_loader in make_image_loaders(
            sizes=[_get_five_ten_crop_canvas_size(size)],
            color_spaces=["RGB"],
            dtypes=[torch.float32],
        ):
            yield ArgsKwargs(image_loader, size=size, vertical_flip=vertical_flip)


def reference_inputs_ten_crop_image_tensor():
    for size, vertical_flip in itertools.product(_FIVE_TEN_CROP_SIZES, [False, True]):
        for image_loader in make_image_loaders(
            sizes=[_get_five_ten_crop_canvas_size(size)], extra_dims=[()], dtypes=[torch.uint8]
        ):
            yield ArgsKwargs(image_loader, size=size, vertical_flip=vertical_flip)


def sample_inputs_ten_crop_video():
    size = _FIVE_TEN_CROP_SIZES[0]
    for video_loader in make_video_loaders(sizes=[_get_five_ten_crop_canvas_size(size)]):
        yield ArgsKwargs(video_loader, size=size)


def multi_crop_pil_reference_wrapper(pil_kernel):
    def wrapper(input_tensor, *other_args, **kwargs):
        output = pil_reference_wrapper(pil_kernel)(input_tensor, *other_args, **kwargs)
        return type(output)(
            F.to_dtype_image(F.to_image(output_pil), dtype=input_tensor.dtype, scale=True) for output_pil in output
        )

    return wrapper


_common_five_ten_crop_marks = [
    xfail_jit_python_scalar_arg("size"),
    mark_framework_limitation(("TestKernels", "test_batched_vs_single"), "Custom batching needed."),
]

KERNEL_INFOS.extend(
    [
        KernelInfo(
            F.five_crop_image,
            sample_inputs_fn=sample_inputs_five_crop_image_tensor,
            reference_fn=multi_crop_pil_reference_wrapper(F._five_crop_image_pil),
            reference_inputs_fn=reference_inputs_five_crop_image_tensor,
            test_marks=_common_five_ten_crop_marks,
        ),
        KernelInfo(
            F.five_crop_video,
            sample_inputs_fn=sample_inputs_five_crop_video,
            test_marks=_common_five_ten_crop_marks,
        ),
        KernelInfo(
            F.ten_crop_image,
            sample_inputs_fn=sample_inputs_ten_crop_image_tensor,
            reference_fn=multi_crop_pil_reference_wrapper(F._ten_crop_image_pil),
            reference_inputs_fn=reference_inputs_ten_crop_image_tensor,
            test_marks=_common_five_ten_crop_marks,
        ),
        KernelInfo(
            F.ten_crop_video,
            sample_inputs_fn=sample_inputs_ten_crop_video,
            test_marks=_common_five_ten_crop_marks,
        ),
    ]
)
