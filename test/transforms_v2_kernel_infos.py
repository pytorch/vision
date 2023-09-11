import functools
import itertools

import PIL.Image
import pytest
import torch.testing
import torchvision.transforms.v2.functional as F
from torchvision.transforms._functional_tensor import _max_value as get_max_value
from transforms_v2_legacy_utils import (
    ArgsKwargs,
    DEFAULT_PORTRAIT_SPATIAL_SIZE,
    InfoBase,
    make_bounding_box_loaders,
    make_image_loaders,
    make_image_loaders_for_interpolation,
    make_mask_loaders,
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


def pixel_difference_closeness_kwargs(uint8_atol, *, dtype=torch.uint8, mae=False):
    return dict(atol=uint8_atol / 255 * get_max_value(dtype), rtol=0, mae=mae)


def cuda_vs_cpu_pixel_difference(atol=1):
    return {
        (("TestKernels", "test_cuda_vs_cpu"), dtype, "cuda"): pixel_difference_closeness_kwargs(atol, dtype=dtype)
        for dtype in [torch.uint8, torch.float32]
    }


def pil_reference_pixel_difference(atol=1, mae=False):
    return {
        (("TestKernels", "test_against_reference"), torch.uint8, "cpu"): pixel_difference_closeness_kwargs(
            atol, mae=mae
        )
    }


def float32_vs_uint8_pixel_difference(atol=1, mae=False):
    return {
        (
            ("TestKernels", "test_float32_vs_uint8"),
            torch.float32,
            "cpu",
        ): pixel_difference_closeness_kwargs(atol, dtype=torch.float32, mae=mae)
    }


def scripted_vs_eager_float64_tolerances(device, atol=1e-6, rtol=1e-6):
    return {
        (("TestKernels", "test_scripted_vs_eager"), torch.float64, device): {"atol": atol, "rtol": rtol, "mae": False},
    }


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


def get_fills(*, num_channels, dtype):
    yield None

    int_value = get_max_value(dtype)
    float_value = int_value / 2
    yield int_value
    yield float_value

    for vector_type in [list, tuple]:
        yield vector_type([int_value])
        yield vector_type([float_value])

        if num_channels > 1:
            yield vector_type(float_value * c / 10 for c in range(num_channels))
            yield vector_type(int_value if c % 2 == 0 else 0 for c in range(num_channels))


def float32_vs_uint8_fill_adapter(other_args, kwargs):
    fill = kwargs.get("fill")
    if fill is None:
        return other_args, kwargs

    if isinstance(fill, (int, float)):
        fill /= 255
    else:
        fill = type(fill)(fill_ / 255 for fill_ in fill)

    return other_args, dict(kwargs, fill=fill)


def _get_elastic_displacement(canvas_size):
    return torch.rand(1, *canvas_size, 2)


def sample_inputs_elastic_image_tensor():
    for image_loader in make_image_loaders(sizes=[DEFAULT_PORTRAIT_SPATIAL_SIZE]):
        displacement = _get_elastic_displacement(image_loader.canvas_size)
        for fill in get_fills(num_channels=image_loader.num_channels, dtype=image_loader.dtype):
            yield ArgsKwargs(image_loader, displacement=displacement, fill=fill)


def reference_inputs_elastic_image_tensor():
    for image_loader, interpolation in itertools.product(
        make_image_loaders_for_interpolation(),
        [
            F.InterpolationMode.NEAREST,
            F.InterpolationMode.BILINEAR,
            F.InterpolationMode.BICUBIC,
        ],
    ):
        displacement = _get_elastic_displacement(image_loader.canvas_size)
        for fill in get_fills(num_channels=image_loader.num_channels, dtype=image_loader.dtype):
            yield ArgsKwargs(image_loader, interpolation=interpolation, displacement=displacement, fill=fill)


def sample_inputs_elastic_bounding_boxes():
    for bounding_boxes_loader in make_bounding_box_loaders():
        displacement = _get_elastic_displacement(bounding_boxes_loader.canvas_size)
        yield ArgsKwargs(
            bounding_boxes_loader,
            format=bounding_boxes_loader.format,
            canvas_size=bounding_boxes_loader.canvas_size,
            displacement=displacement,
        )


def sample_inputs_elastic_mask():
    for mask_loader in make_mask_loaders(sizes=[DEFAULT_PORTRAIT_SPATIAL_SIZE]):
        displacement = _get_elastic_displacement(mask_loader.shape[-2:])
        yield ArgsKwargs(mask_loader, displacement=displacement)


def sample_inputs_elastic_video():
    for video_loader in make_video_loaders(sizes=[DEFAULT_PORTRAIT_SPATIAL_SIZE], num_frames=[3]):
        displacement = _get_elastic_displacement(video_loader.shape[-2:])
        yield ArgsKwargs(video_loader, displacement=displacement)


KERNEL_INFOS.extend(
    [
        KernelInfo(
            F.elastic_image,
            sample_inputs_fn=sample_inputs_elastic_image_tensor,
            reference_inputs_fn=reference_inputs_elastic_image_tensor,
            float32_vs_uint8=float32_vs_uint8_fill_adapter,
            closeness_kwargs={
                **float32_vs_uint8_pixel_difference(6, mae=True),
                **cuda_vs_cpu_pixel_difference(),
            },
            test_marks=[xfail_jit_python_scalar_arg("fill")],
        ),
        KernelInfo(
            F.elastic_bounding_boxes,
            sample_inputs_fn=sample_inputs_elastic_bounding_boxes,
        ),
        KernelInfo(
            F.elastic_mask,
            sample_inputs_fn=sample_inputs_elastic_mask,
        ),
        KernelInfo(
            F.elastic_video,
            sample_inputs_fn=sample_inputs_elastic_video,
            closeness_kwargs=cuda_vs_cpu_pixel_difference(),
        ),
    ]
)


def sample_inputs_invert_image_tensor():
    for image_loader in make_image_loaders(sizes=[DEFAULT_PORTRAIT_SPATIAL_SIZE], color_spaces=("GRAY", "RGB")):
        yield ArgsKwargs(image_loader)


def reference_inputs_invert_image_tensor():
    for image_loader in make_image_loaders(color_spaces=("GRAY", "RGB"), extra_dims=[()], dtypes=[torch.uint8]):
        yield ArgsKwargs(image_loader)


def sample_inputs_invert_video():
    for video_loader in make_video_loaders(sizes=[DEFAULT_PORTRAIT_SPATIAL_SIZE], num_frames=[3]):
        yield ArgsKwargs(video_loader)


KERNEL_INFOS.extend(
    [
        KernelInfo(
            F.invert_image,
            kernel_name="invert_image_tensor",
            sample_inputs_fn=sample_inputs_invert_image_tensor,
            reference_fn=pil_reference_wrapper(F._invert_image_pil),
            reference_inputs_fn=reference_inputs_invert_image_tensor,
            float32_vs_uint8=True,
        ),
        KernelInfo(
            F.invert_video,
            sample_inputs_fn=sample_inputs_invert_video,
        ),
    ]
)


_POSTERIZE_BITS = [1, 4, 8]


def sample_inputs_posterize_image_tensor():
    for image_loader in make_image_loaders(sizes=[DEFAULT_PORTRAIT_SPATIAL_SIZE], color_spaces=("GRAY", "RGB")):
        yield ArgsKwargs(image_loader, bits=_POSTERIZE_BITS[0])


def reference_inputs_posterize_image_tensor():
    for image_loader, bits in itertools.product(
        make_image_loaders(color_spaces=("GRAY", "RGB"), extra_dims=[()], dtypes=[torch.uint8]),
        _POSTERIZE_BITS,
    ):
        yield ArgsKwargs(image_loader, bits=bits)


def sample_inputs_posterize_video():
    for video_loader in make_video_loaders(sizes=[DEFAULT_PORTRAIT_SPATIAL_SIZE], num_frames=[3]):
        yield ArgsKwargs(video_loader, bits=_POSTERIZE_BITS[0])


KERNEL_INFOS.extend(
    [
        KernelInfo(
            F.posterize_image,
            kernel_name="posterize_image_tensor",
            sample_inputs_fn=sample_inputs_posterize_image_tensor,
            reference_fn=pil_reference_wrapper(F._posterize_image_pil),
            reference_inputs_fn=reference_inputs_posterize_image_tensor,
            float32_vs_uint8=True,
            closeness_kwargs=float32_vs_uint8_pixel_difference(),
        ),
        KernelInfo(
            F.posterize_video,
            sample_inputs_fn=sample_inputs_posterize_video,
        ),
    ]
)


def _get_solarize_thresholds(dtype):
    for factor in [0.1, 0.5]:
        max_value = get_max_value(dtype)
        yield (float if dtype.is_floating_point else int)(max_value * factor)


def sample_inputs_solarize_image_tensor():
    for image_loader in make_image_loaders(sizes=[DEFAULT_PORTRAIT_SPATIAL_SIZE], color_spaces=("GRAY", "RGB")):
        yield ArgsKwargs(image_loader, threshold=next(_get_solarize_thresholds(image_loader.dtype)))


def reference_inputs_solarize_image_tensor():
    for image_loader in make_image_loaders(color_spaces=("GRAY", "RGB"), extra_dims=[()], dtypes=[torch.uint8]):
        for threshold in _get_solarize_thresholds(image_loader.dtype):
            yield ArgsKwargs(image_loader, threshold=threshold)


def uint8_to_float32_threshold_adapter(other_args, kwargs):
    return other_args, dict(threshold=kwargs["threshold"] / 255)


def sample_inputs_solarize_video():
    for video_loader in make_video_loaders(sizes=[DEFAULT_PORTRAIT_SPATIAL_SIZE], num_frames=[3]):
        yield ArgsKwargs(video_loader, threshold=next(_get_solarize_thresholds(video_loader.dtype)))


KERNEL_INFOS.extend(
    [
        KernelInfo(
            F.solarize_image,
            kernel_name="solarize_image_tensor",
            sample_inputs_fn=sample_inputs_solarize_image_tensor,
            reference_fn=pil_reference_wrapper(F._solarize_image_pil),
            reference_inputs_fn=reference_inputs_solarize_image_tensor,
            float32_vs_uint8=uint8_to_float32_threshold_adapter,
            closeness_kwargs=float32_vs_uint8_pixel_difference(),
        ),
        KernelInfo(
            F.solarize_video,
            sample_inputs_fn=sample_inputs_solarize_video,
        ),
    ]
)


def sample_inputs_autocontrast_image_tensor():
    for image_loader in make_image_loaders(sizes=[DEFAULT_PORTRAIT_SPATIAL_SIZE], color_spaces=("GRAY", "RGB")):
        yield ArgsKwargs(image_loader)


def reference_inputs_autocontrast_image_tensor():
    for image_loader in make_image_loaders(color_spaces=("GRAY", "RGB"), extra_dims=[()], dtypes=[torch.uint8]):
        yield ArgsKwargs(image_loader)


def sample_inputs_autocontrast_video():
    for video_loader in make_video_loaders(sizes=[DEFAULT_PORTRAIT_SPATIAL_SIZE], num_frames=[3]):
        yield ArgsKwargs(video_loader)


KERNEL_INFOS.extend(
    [
        KernelInfo(
            F.autocontrast_image,
            kernel_name="autocontrast_image_tensor",
            sample_inputs_fn=sample_inputs_autocontrast_image_tensor,
            reference_fn=pil_reference_wrapper(F._autocontrast_image_pil),
            reference_inputs_fn=reference_inputs_autocontrast_image_tensor,
            float32_vs_uint8=True,
            closeness_kwargs={
                **pil_reference_pixel_difference(),
                **float32_vs_uint8_pixel_difference(),
            },
        ),
        KernelInfo(
            F.autocontrast_video,
            sample_inputs_fn=sample_inputs_autocontrast_video,
        ),
    ]
)

_ADJUST_SHARPNESS_FACTORS = [0.1, 0.5]


def sample_inputs_adjust_sharpness_image_tensor():
    for image_loader in make_image_loaders(
        sizes=[DEFAULT_PORTRAIT_SPATIAL_SIZE, (2, 2)],
        color_spaces=("GRAY", "RGB"),
    ):
        yield ArgsKwargs(image_loader, sharpness_factor=_ADJUST_SHARPNESS_FACTORS[0])


def reference_inputs_adjust_sharpness_image_tensor():
    for image_loader, sharpness_factor in itertools.product(
        make_image_loaders(color_spaces=("GRAY", "RGB"), extra_dims=[()], dtypes=[torch.uint8]),
        _ADJUST_SHARPNESS_FACTORS,
    ):
        yield ArgsKwargs(image_loader, sharpness_factor=sharpness_factor)


def sample_inputs_adjust_sharpness_video():
    for video_loader in make_video_loaders(sizes=[DEFAULT_PORTRAIT_SPATIAL_SIZE], num_frames=[3]):
        yield ArgsKwargs(video_loader, sharpness_factor=_ADJUST_SHARPNESS_FACTORS[0])


KERNEL_INFOS.extend(
    [
        KernelInfo(
            F.adjust_sharpness_image,
            kernel_name="adjust_sharpness_image_tensor",
            sample_inputs_fn=sample_inputs_adjust_sharpness_image_tensor,
            reference_fn=pil_reference_wrapper(F._adjust_sharpness_image_pil),
            reference_inputs_fn=reference_inputs_adjust_sharpness_image_tensor,
            float32_vs_uint8=True,
            closeness_kwargs=float32_vs_uint8_pixel_difference(2),
        ),
        KernelInfo(
            F.adjust_sharpness_video,
            sample_inputs_fn=sample_inputs_adjust_sharpness_video,
        ),
    ]
)


_ADJUST_CONTRAST_FACTORS = [0.1, 0.5]


def sample_inputs_adjust_contrast_image_tensor():
    for image_loader in make_image_loaders(sizes=[DEFAULT_PORTRAIT_SPATIAL_SIZE], color_spaces=("GRAY", "RGB")):
        yield ArgsKwargs(image_loader, contrast_factor=_ADJUST_CONTRAST_FACTORS[0])


def reference_inputs_adjust_contrast_image_tensor():
    for image_loader, contrast_factor in itertools.product(
        make_image_loaders(color_spaces=("GRAY", "RGB"), extra_dims=[()], dtypes=[torch.uint8]),
        _ADJUST_CONTRAST_FACTORS,
    ):
        yield ArgsKwargs(image_loader, contrast_factor=contrast_factor)


def sample_inputs_adjust_contrast_video():
    for video_loader in make_video_loaders(sizes=[DEFAULT_PORTRAIT_SPATIAL_SIZE], num_frames=[3]):
        yield ArgsKwargs(video_loader, contrast_factor=_ADJUST_CONTRAST_FACTORS[0])


KERNEL_INFOS.extend(
    [
        KernelInfo(
            F.adjust_contrast_image,
            kernel_name="adjust_contrast_image_tensor",
            sample_inputs_fn=sample_inputs_adjust_contrast_image_tensor,
            reference_fn=pil_reference_wrapper(F._adjust_contrast_image_pil),
            reference_inputs_fn=reference_inputs_adjust_contrast_image_tensor,
            float32_vs_uint8=True,
            closeness_kwargs={
                **pil_reference_pixel_difference(),
                **float32_vs_uint8_pixel_difference(2),
                **cuda_vs_cpu_pixel_difference(),
                (("TestKernels", "test_against_reference"), torch.uint8, "cpu"): pixel_difference_closeness_kwargs(1),
            },
        ),
        KernelInfo(
            F.adjust_contrast_video,
            sample_inputs_fn=sample_inputs_adjust_contrast_video,
            closeness_kwargs={
                **cuda_vs_cpu_pixel_difference(),
                (("TestKernels", "test_against_reference"), torch.uint8, "cpu"): pixel_difference_closeness_kwargs(1),
            },
        ),
    ]
)

_ADJUST_GAMMA_GAMMAS_GAINS = [
    (0.5, 2.0),
    (0.0, 1.0),
]


def sample_inputs_adjust_gamma_image_tensor():
    gamma, gain = _ADJUST_GAMMA_GAMMAS_GAINS[0]
    for image_loader in make_image_loaders(sizes=[DEFAULT_PORTRAIT_SPATIAL_SIZE], color_spaces=("GRAY", "RGB")):
        yield ArgsKwargs(image_loader, gamma=gamma, gain=gain)


def reference_inputs_adjust_gamma_image_tensor():
    for image_loader, (gamma, gain) in itertools.product(
        make_image_loaders(color_spaces=("GRAY", "RGB"), extra_dims=[()], dtypes=[torch.uint8]),
        _ADJUST_GAMMA_GAMMAS_GAINS,
    ):
        yield ArgsKwargs(image_loader, gamma=gamma, gain=gain)


def sample_inputs_adjust_gamma_video():
    gamma, gain = _ADJUST_GAMMA_GAMMAS_GAINS[0]
    for video_loader in make_video_loaders(sizes=[DEFAULT_PORTRAIT_SPATIAL_SIZE], num_frames=[3]):
        yield ArgsKwargs(video_loader, gamma=gamma, gain=gain)


KERNEL_INFOS.extend(
    [
        KernelInfo(
            F.adjust_gamma_image,
            kernel_name="adjust_gamma_image_tensor",
            sample_inputs_fn=sample_inputs_adjust_gamma_image_tensor,
            reference_fn=pil_reference_wrapper(F._adjust_gamma_image_pil),
            reference_inputs_fn=reference_inputs_adjust_gamma_image_tensor,
            float32_vs_uint8=True,
            closeness_kwargs={
                **pil_reference_pixel_difference(),
                **float32_vs_uint8_pixel_difference(),
            },
        ),
        KernelInfo(
            F.adjust_gamma_video,
            sample_inputs_fn=sample_inputs_adjust_gamma_video,
        ),
    ]
)


_ADJUST_HUE_FACTORS = [-0.1, 0.5]


def sample_inputs_adjust_hue_image_tensor():
    for image_loader in make_image_loaders(sizes=[DEFAULT_PORTRAIT_SPATIAL_SIZE], color_spaces=("GRAY", "RGB")):
        yield ArgsKwargs(image_loader, hue_factor=_ADJUST_HUE_FACTORS[0])


def reference_inputs_adjust_hue_image_tensor():
    for image_loader, hue_factor in itertools.product(
        make_image_loaders(color_spaces=("GRAY", "RGB"), extra_dims=[()], dtypes=[torch.uint8]),
        _ADJUST_HUE_FACTORS,
    ):
        yield ArgsKwargs(image_loader, hue_factor=hue_factor)


def sample_inputs_adjust_hue_video():
    for video_loader in make_video_loaders(sizes=[DEFAULT_PORTRAIT_SPATIAL_SIZE], num_frames=[3]):
        yield ArgsKwargs(video_loader, hue_factor=_ADJUST_HUE_FACTORS[0])


KERNEL_INFOS.extend(
    [
        KernelInfo(
            F.adjust_hue_image,
            kernel_name="adjust_hue_image_tensor",
            sample_inputs_fn=sample_inputs_adjust_hue_image_tensor,
            reference_fn=pil_reference_wrapper(F._adjust_hue_image_pil),
            reference_inputs_fn=reference_inputs_adjust_hue_image_tensor,
            float32_vs_uint8=True,
            closeness_kwargs={
                **pil_reference_pixel_difference(2, mae=True),
                **float32_vs_uint8_pixel_difference(),
            },
        ),
        KernelInfo(
            F.adjust_hue_video,
            sample_inputs_fn=sample_inputs_adjust_hue_video,
        ),
    ]
)

_ADJUST_SATURATION_FACTORS = [0.1, 0.5]


def sample_inputs_adjust_saturation_image_tensor():
    for image_loader in make_image_loaders(sizes=[DEFAULT_PORTRAIT_SPATIAL_SIZE], color_spaces=("GRAY", "RGB")):
        yield ArgsKwargs(image_loader, saturation_factor=_ADJUST_SATURATION_FACTORS[0])


def reference_inputs_adjust_saturation_image_tensor():
    for image_loader, saturation_factor in itertools.product(
        make_image_loaders(color_spaces=("GRAY", "RGB"), extra_dims=[()], dtypes=[torch.uint8]),
        _ADJUST_SATURATION_FACTORS,
    ):
        yield ArgsKwargs(image_loader, saturation_factor=saturation_factor)


def sample_inputs_adjust_saturation_video():
    for video_loader in make_video_loaders(sizes=[DEFAULT_PORTRAIT_SPATIAL_SIZE], num_frames=[3]):
        yield ArgsKwargs(video_loader, saturation_factor=_ADJUST_SATURATION_FACTORS[0])


KERNEL_INFOS.extend(
    [
        KernelInfo(
            F.adjust_saturation_image,
            kernel_name="adjust_saturation_image_tensor",
            sample_inputs_fn=sample_inputs_adjust_saturation_image_tensor,
            reference_fn=pil_reference_wrapper(F._adjust_saturation_image_pil),
            reference_inputs_fn=reference_inputs_adjust_saturation_image_tensor,
            float32_vs_uint8=True,
            closeness_kwargs={
                **pil_reference_pixel_difference(),
                **float32_vs_uint8_pixel_difference(2),
                **cuda_vs_cpu_pixel_difference(),
            },
        ),
        KernelInfo(
            F.adjust_saturation_video,
            sample_inputs_fn=sample_inputs_adjust_saturation_video,
            closeness_kwargs=cuda_vs_cpu_pixel_difference(),
        ),
    ]
)


def sample_inputs_clamp_bounding_boxes():
    for bounding_boxes_loader in make_bounding_box_loaders():
        yield ArgsKwargs(
            bounding_boxes_loader,
            format=bounding_boxes_loader.format,
            canvas_size=bounding_boxes_loader.canvas_size,
        )


KERNEL_INFOS.append(
    KernelInfo(
        F.clamp_bounding_boxes,
        sample_inputs_fn=sample_inputs_clamp_bounding_boxes,
        logs_usage=True,
    )
)

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
