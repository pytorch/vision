import functools
import itertools

import numpy as np
import PIL.Image
import pytest
import torch.testing
import torchvision.ops
import torchvision.transforms.v2.functional as F
from common_utils import (
    ArgsKwargs,
    combinations_grid,
    DEFAULT_PORTRAIT_SPATIAL_SIZE,
    get_num_channels,
    ImageLoader,
    InfoBase,
    make_bounding_box_loader,
    make_bounding_box_loaders,
    make_detection_mask_loader,
    make_image_loader,
    make_image_loaders,
    make_image_loaders_for_interpolation,
    make_mask_loaders,
    make_video_loader,
    make_video_loaders,
    mark_framework_limitation,
    TestMark,
)
from torchvision import datapoints
from torchvision.transforms._functional_tensor import _max_value as get_max_value, _parse_pad_padding

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

        input_pil = F.to_image_pil(input_tensor)
        output_pil = pil_kernel(input_pil, *other_args, **kwargs)
        if not isinstance(output_pil, PIL.Image.Image):
            return output_pil

        output_tensor = F.to_image_tensor(output_pil)

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


def reference_affine_bounding_boxes_helper(bounding_boxes, *, format, canvas_size, affine_matrix):
    def transform(bbox, affine_matrix_, format_, canvas_size_):
        # Go to float before converting to prevent precision loss in case of CXCYWH -> XYXY and W or H is 1
        in_dtype = bbox.dtype
        if not torch.is_floating_point(bbox):
            bbox = bbox.float()
        bbox_xyxy = F.convert_format_bounding_boxes(
            bbox.as_subclass(torch.Tensor),
            old_format=format_,
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
        transformed_points = np.matmul(points, affine_matrix_.T)
        out_bbox = torch.tensor(
            [
                np.min(transformed_points[:, 0]).item(),
                np.min(transformed_points[:, 1]).item(),
                np.max(transformed_points[:, 0]).item(),
                np.max(transformed_points[:, 1]).item(),
            ],
            dtype=bbox_xyxy.dtype,
        )
        out_bbox = F.convert_format_bounding_boxes(
            out_bbox, old_format=datapoints.BoundingBoxFormat.XYXY, new_format=format_, inplace=True
        )
        # It is important to clamp before casting, especially for CXCYWH format, dtype=int64
        out_bbox = F.clamp_bounding_boxes(out_bbox, format=format_, canvas_size=canvas_size_)
        out_bbox = out_bbox.to(dtype=in_dtype)
        return out_bbox

    return torch.stack(
        [transform(b, affine_matrix, format, canvas_size) for b in bounding_boxes.reshape(-1, 4).unbind()]
    ).reshape(bounding_boxes.shape)


def sample_inputs_convert_format_bounding_boxes():
    formats = list(datapoints.BoundingBoxFormat)
    for bounding_boxes_loader, new_format in itertools.product(make_bounding_box_loaders(formats=formats), formats):
        yield ArgsKwargs(bounding_boxes_loader, old_format=bounding_boxes_loader.format, new_format=new_format)


def reference_convert_format_bounding_boxes(bounding_boxes, old_format, new_format):
    return torchvision.ops.box_convert(
        bounding_boxes, in_fmt=old_format.name.lower(), out_fmt=new_format.name.lower()
    ).to(bounding_boxes.dtype)


def reference_inputs_convert_format_bounding_boxes():
    for args_kwargs in sample_inputs_convert_format_bounding_boxes():
        if len(args_kwargs.args[0].shape) == 2:
            yield args_kwargs


KERNEL_INFOS.append(
    KernelInfo(
        F.convert_format_bounding_boxes,
        sample_inputs_fn=sample_inputs_convert_format_bounding_boxes,
        reference_fn=reference_convert_format_bounding_boxes,
        reference_inputs_fn=reference_inputs_convert_format_bounding_boxes,
        logs_usage=True,
        closeness_kwargs={
            (("TestKernels", "test_against_reference"), torch.int64, "cpu"): dict(atol=1, rtol=0),
        },
    ),
)


_CROP_PARAMS = combinations_grid(top=[-8, 0, 9], left=[-8, 0, 9], height=[12, 20], width=[12, 20])


def sample_inputs_crop_image_tensor():
    for image_loader, params in itertools.product(
        make_image_loaders(sizes=[(16, 17)], color_spaces=["RGB"], dtypes=[torch.float32]),
        [
            dict(top=4, left=3, height=7, width=8),
            dict(top=-1, left=3, height=7, width=8),
            dict(top=4, left=-1, height=7, width=8),
            dict(top=4, left=3, height=17, width=8),
            dict(top=4, left=3, height=7, width=18),
        ],
    ):
        yield ArgsKwargs(image_loader, **params)


def reference_inputs_crop_image_tensor():
    for image_loader, params in itertools.product(
        make_image_loaders(extra_dims=[()], dtypes=[torch.uint8]), _CROP_PARAMS
    ):
        yield ArgsKwargs(image_loader, **params)


def sample_inputs_crop_bounding_boxes():
    for bounding_boxes_loader, params in itertools.product(
        make_bounding_box_loaders(), [_CROP_PARAMS[0], _CROP_PARAMS[-1]]
    ):
        yield ArgsKwargs(bounding_boxes_loader, format=bounding_boxes_loader.format, **params)


def sample_inputs_crop_mask():
    for mask_loader in make_mask_loaders(sizes=[(16, 17)], num_categories=[10], num_objects=[5]):
        yield ArgsKwargs(mask_loader, top=4, left=3, height=7, width=8)


def reference_inputs_crop_mask():
    for mask_loader, params in itertools.product(make_mask_loaders(extra_dims=[()], num_objects=[1]), _CROP_PARAMS):
        yield ArgsKwargs(mask_loader, **params)


def sample_inputs_crop_video():
    for video_loader in make_video_loaders(sizes=[(16, 17)], num_frames=[3]):
        yield ArgsKwargs(video_loader, top=4, left=3, height=7, width=8)


def reference_crop_bounding_boxes(bounding_boxes, *, format, top, left, height, width):
    affine_matrix = np.array(
        [
            [1, 0, -left],
            [0, 1, -top],
        ],
        dtype="float64" if bounding_boxes.dtype == torch.float64 else "float32",
    )

    canvas_size = (height, width)
    expected_bboxes = reference_affine_bounding_boxes_helper(
        bounding_boxes, format=format, canvas_size=canvas_size, affine_matrix=affine_matrix
    )
    return expected_bboxes, canvas_size


def reference_inputs_crop_bounding_boxes():
    for bounding_boxes_loader, params in itertools.product(
        make_bounding_box_loaders(extra_dims=((), (4,))), [_CROP_PARAMS[0], _CROP_PARAMS[-1]]
    ):
        yield ArgsKwargs(bounding_boxes_loader, format=bounding_boxes_loader.format, **params)


KERNEL_INFOS.extend(
    [
        KernelInfo(
            F.crop_image,
            kernel_name="crop_image_tensor",
            sample_inputs_fn=sample_inputs_crop_image_tensor,
            reference_fn=pil_reference_wrapper(F._crop_image_pil),
            reference_inputs_fn=reference_inputs_crop_image_tensor,
            float32_vs_uint8=True,
        ),
        KernelInfo(
            F.crop_bounding_boxes,
            sample_inputs_fn=sample_inputs_crop_bounding_boxes,
            reference_fn=reference_crop_bounding_boxes,
            reference_inputs_fn=reference_inputs_crop_bounding_boxes,
        ),
        KernelInfo(
            F.crop_mask,
            sample_inputs_fn=sample_inputs_crop_mask,
            reference_fn=pil_reference_wrapper(F._crop_image_pil),
            reference_inputs_fn=reference_inputs_crop_mask,
            float32_vs_uint8=True,
        ),
        KernelInfo(
            F.crop_video,
            sample_inputs_fn=sample_inputs_crop_video,
        ),
    ]
)

_RESIZED_CROP_PARAMS = combinations_grid(top=[-8, 9], left=[-8, 9], height=[12], width=[12], size=[(16, 18)])


def sample_inputs_resized_crop_image_tensor():
    for image_loader in make_image_loaders():
        yield ArgsKwargs(image_loader, **_RESIZED_CROP_PARAMS[0])


@pil_reference_wrapper
def reference_resized_crop_image_tensor(*args, **kwargs):
    if not kwargs.pop("antialias", False) and kwargs.get("interpolation", F.InterpolationMode.BILINEAR) in {
        F.InterpolationMode.BILINEAR,
        F.InterpolationMode.BICUBIC,
    }:
        raise pytest.UsageError("Anti-aliasing is always active in PIL")
    return F._resized_crop_image_pil(*args, **kwargs)


def reference_inputs_resized_crop_image_tensor():
    for image_loader, interpolation, params in itertools.product(
        make_image_loaders_for_interpolation(),
        [
            F.InterpolationMode.NEAREST,
            F.InterpolationMode.NEAREST_EXACT,
            F.InterpolationMode.BILINEAR,
            F.InterpolationMode.BICUBIC,
        ],
        _RESIZED_CROP_PARAMS,
    ):
        yield ArgsKwargs(
            image_loader,
            interpolation=interpolation,
            antialias=interpolation
            in {
                F.InterpolationMode.BILINEAR,
                F.InterpolationMode.BICUBIC,
            },
            **params,
        )


def sample_inputs_resized_crop_bounding_boxes():
    for bounding_boxes_loader in make_bounding_box_loaders():
        yield ArgsKwargs(bounding_boxes_loader, format=bounding_boxes_loader.format, **_RESIZED_CROP_PARAMS[0])


def sample_inputs_resized_crop_mask():
    for mask_loader in make_mask_loaders():
        yield ArgsKwargs(mask_loader, **_RESIZED_CROP_PARAMS[0])


def sample_inputs_resized_crop_video():
    for video_loader in make_video_loaders(sizes=[DEFAULT_PORTRAIT_SPATIAL_SIZE], num_frames=[3]):
        yield ArgsKwargs(video_loader, **_RESIZED_CROP_PARAMS[0])


KERNEL_INFOS.extend(
    [
        KernelInfo(
            F.resized_crop_image,
            sample_inputs_fn=sample_inputs_resized_crop_image_tensor,
            reference_fn=reference_resized_crop_image_tensor,
            reference_inputs_fn=reference_inputs_resized_crop_image_tensor,
            float32_vs_uint8=True,
            closeness_kwargs={
                **cuda_vs_cpu_pixel_difference(),
                **pil_reference_pixel_difference(3, mae=True),
                **float32_vs_uint8_pixel_difference(3, mae=True),
            },
        ),
        KernelInfo(
            F.resized_crop_bounding_boxes,
            sample_inputs_fn=sample_inputs_resized_crop_bounding_boxes,
        ),
        KernelInfo(
            F.resized_crop_mask,
            sample_inputs_fn=sample_inputs_resized_crop_mask,
        ),
        KernelInfo(
            F.resized_crop_video,
            sample_inputs_fn=sample_inputs_resized_crop_video,
            closeness_kwargs=cuda_vs_cpu_pixel_difference(),
        ),
    ]
)

_PAD_PARAMS = combinations_grid(
    padding=[[1], [1, 1], [1, 1, 2, 2]],
    padding_mode=["constant", "symmetric", "edge", "reflect"],
)


def sample_inputs_pad_image_tensor():
    make_pad_image_loaders = functools.partial(
        make_image_loaders, sizes=[DEFAULT_PORTRAIT_SPATIAL_SIZE], color_spaces=["RGB"], dtypes=[torch.float32]
    )

    for image_loader, padding in itertools.product(
        make_pad_image_loaders(),
        [1, (1,), (1, 2), (1, 2, 3, 4), [1], [1, 2], [1, 2, 3, 4]],
    ):
        yield ArgsKwargs(image_loader, padding=padding)

    for image_loader in make_pad_image_loaders():
        for fill in get_fills(num_channels=image_loader.num_channels, dtype=image_loader.dtype):
            yield ArgsKwargs(image_loader, padding=[1], fill=fill)

    for image_loader, padding_mode in itertools.product(
        # We branch for non-constant padding and integer inputs
        make_pad_image_loaders(dtypes=[torch.uint8]),
        ["constant", "symmetric", "edge", "reflect"],
    ):
        yield ArgsKwargs(image_loader, padding=[1], padding_mode=padding_mode)

    # `torch.nn.functional.pad` does not support symmetric padding, and thus we have a custom implementation. Besides
    # negative padding, this is already handled by the inputs above.
    for image_loader in make_pad_image_loaders():
        yield ArgsKwargs(image_loader, padding=[-1], padding_mode="symmetric")


def reference_inputs_pad_image_tensor():
    for image_loader, params in itertools.product(
        make_image_loaders(extra_dims=[()], dtypes=[torch.uint8]), _PAD_PARAMS
    ):
        for fill in get_fills(
            num_channels=image_loader.num_channels,
            dtype=image_loader.dtype,
        ):
            # FIXME: PIL kernel doesn't support sequences of length 1 if the number of channels is larger. Shouldn't it?
            if isinstance(fill, (list, tuple)):
                continue

            yield ArgsKwargs(image_loader, fill=fill, **params)


def sample_inputs_pad_bounding_boxes():
    for bounding_boxes_loader, padding in itertools.product(
        make_bounding_box_loaders(), [1, (1,), (1, 2), (1, 2, 3, 4), [1], [1, 2], [1, 2, 3, 4]]
    ):
        yield ArgsKwargs(
            bounding_boxes_loader,
            format=bounding_boxes_loader.format,
            canvas_size=bounding_boxes_loader.canvas_size,
            padding=padding,
            padding_mode="constant",
        )


def sample_inputs_pad_mask():
    for mask_loader in make_mask_loaders(sizes=[DEFAULT_PORTRAIT_SPATIAL_SIZE], num_categories=[10], num_objects=[5]):
        yield ArgsKwargs(mask_loader, padding=[1])


def reference_inputs_pad_mask():
    for mask_loader, fill, params in itertools.product(
        make_mask_loaders(num_objects=[1], extra_dims=[()]), [None, 127], _PAD_PARAMS
    ):
        yield ArgsKwargs(mask_loader, fill=fill, **params)


def sample_inputs_pad_video():
    for video_loader in make_video_loaders(sizes=[DEFAULT_PORTRAIT_SPATIAL_SIZE], num_frames=[3]):
        yield ArgsKwargs(video_loader, padding=[1])


def reference_pad_bounding_boxes(bounding_boxes, *, format, canvas_size, padding, padding_mode):

    left, right, top, bottom = _parse_pad_padding(padding)

    affine_matrix = np.array(
        [
            [1, 0, left],
            [0, 1, top],
        ],
        dtype="float64" if bounding_boxes.dtype == torch.float64 else "float32",
    )

    height = canvas_size[0] + top + bottom
    width = canvas_size[1] + left + right

    expected_bboxes = reference_affine_bounding_boxes_helper(
        bounding_boxes, format=format, canvas_size=(height, width), affine_matrix=affine_matrix
    )
    return expected_bboxes, (height, width)


def reference_inputs_pad_bounding_boxes():
    for bounding_boxes_loader, padding in itertools.product(
        make_bounding_box_loaders(extra_dims=((), (4,))), [1, (1,), (1, 2), (1, 2, 3, 4), [1], [1, 2], [1, 2, 3, 4]]
    ):
        yield ArgsKwargs(
            bounding_boxes_loader,
            format=bounding_boxes_loader.format,
            canvas_size=bounding_boxes_loader.canvas_size,
            padding=padding,
            padding_mode="constant",
        )


def pad_xfail_jit_fill_condition(args_kwargs):
    fill = args_kwargs.kwargs.get("fill")
    if not isinstance(fill, (list, tuple)):
        return False
    elif isinstance(fill, tuple):
        return True
    else:  # isinstance(fill, list):
        return all(isinstance(f, int) for f in fill)


KERNEL_INFOS.extend(
    [
        KernelInfo(
            F.pad_image,
            sample_inputs_fn=sample_inputs_pad_image_tensor,
            reference_fn=pil_reference_wrapper(F._pad_image_pil),
            reference_inputs_fn=reference_inputs_pad_image_tensor,
            float32_vs_uint8=float32_vs_uint8_fill_adapter,
            closeness_kwargs=float32_vs_uint8_pixel_difference(),
            test_marks=[
                xfail_jit_python_scalar_arg("padding"),
                xfail_jit(
                    "F.pad only supports vector fills for list of floats", condition=pad_xfail_jit_fill_condition
                ),
            ],
        ),
        KernelInfo(
            F.pad_bounding_boxes,
            sample_inputs_fn=sample_inputs_pad_bounding_boxes,
            reference_fn=reference_pad_bounding_boxes,
            reference_inputs_fn=reference_inputs_pad_bounding_boxes,
            test_marks=[
                xfail_jit_python_scalar_arg("padding"),
            ],
        ),
        KernelInfo(
            F.pad_mask,
            sample_inputs_fn=sample_inputs_pad_mask,
            reference_fn=pil_reference_wrapper(F._pad_image_pil),
            reference_inputs_fn=reference_inputs_pad_mask,
            float32_vs_uint8=float32_vs_uint8_fill_adapter,
        ),
        KernelInfo(
            F.pad_video,
            sample_inputs_fn=sample_inputs_pad_video,
        ),
    ]
)

_PERSPECTIVE_COEFFS = [
    [1.2405, 0.1772, -6.9113, 0.0463, 1.251, -5.235, 0.00013, 0.0018],
    [0.7366, -0.11724, 1.45775, -0.15012, 0.73406, 2.6019, -0.0072, -0.0063],
]
_STARTPOINTS = [[0, 1], [2, 3], [4, 5], [6, 7]]
_ENDPOINTS = [[9, 8], [7, 6], [5, 4], [3, 2]]


def sample_inputs_perspective_image_tensor():
    for image_loader in make_image_loaders(sizes=[DEFAULT_PORTRAIT_SPATIAL_SIZE]):
        for fill in get_fills(num_channels=image_loader.num_channels, dtype=image_loader.dtype):
            yield ArgsKwargs(
                image_loader, startpoints=None, endpoints=None, fill=fill, coefficients=_PERSPECTIVE_COEFFS[0]
            )

    yield ArgsKwargs(make_image_loader(), startpoints=_STARTPOINTS, endpoints=_ENDPOINTS)


def reference_inputs_perspective_image_tensor():
    for image_loader, coefficients, interpolation in itertools.product(
        make_image_loaders_for_interpolation(),
        _PERSPECTIVE_COEFFS,
        [
            F.InterpolationMode.NEAREST,
            F.InterpolationMode.BILINEAR,
        ],
    ):
        for fill in get_fills(num_channels=image_loader.num_channels, dtype=image_loader.dtype):
            # FIXME: PIL kernel doesn't support sequences of length 1 if the number of channels is larger. Shouldn't it?
            if isinstance(fill, (list, tuple)):
                continue

            yield ArgsKwargs(
                image_loader,
                startpoints=None,
                endpoints=None,
                interpolation=interpolation,
                fill=fill,
                coefficients=coefficients,
            )


def sample_inputs_perspective_bounding_boxes():
    for bounding_boxes_loader in make_bounding_box_loaders():
        yield ArgsKwargs(
            bounding_boxes_loader,
            format=bounding_boxes_loader.format,
            canvas_size=bounding_boxes_loader.canvas_size,
            startpoints=None,
            endpoints=None,
            coefficients=_PERSPECTIVE_COEFFS[0],
        )

    format = datapoints.BoundingBoxFormat.XYXY
    loader = make_bounding_box_loader(format=format)
    yield ArgsKwargs(
        loader, format=format, canvas_size=loader.canvas_size, startpoints=_STARTPOINTS, endpoints=_ENDPOINTS
    )


def sample_inputs_perspective_mask():
    for mask_loader in make_mask_loaders(sizes=[DEFAULT_PORTRAIT_SPATIAL_SIZE]):
        yield ArgsKwargs(mask_loader, startpoints=None, endpoints=None, coefficients=_PERSPECTIVE_COEFFS[0])

    yield ArgsKwargs(make_detection_mask_loader(), startpoints=_STARTPOINTS, endpoints=_ENDPOINTS)


def reference_inputs_perspective_mask():
    for mask_loader, perspective_coeffs in itertools.product(
        make_mask_loaders(extra_dims=[()], num_objects=[1]), _PERSPECTIVE_COEFFS
    ):
        yield ArgsKwargs(mask_loader, startpoints=None, endpoints=None, coefficients=perspective_coeffs)


def sample_inputs_perspective_video():
    for video_loader in make_video_loaders(sizes=[DEFAULT_PORTRAIT_SPATIAL_SIZE], num_frames=[3]):
        yield ArgsKwargs(video_loader, startpoints=None, endpoints=None, coefficients=_PERSPECTIVE_COEFFS[0])

    yield ArgsKwargs(make_video_loader(), startpoints=_STARTPOINTS, endpoints=_ENDPOINTS)


KERNEL_INFOS.extend(
    [
        KernelInfo(
            F.perspective_image,
            sample_inputs_fn=sample_inputs_perspective_image_tensor,
            reference_fn=pil_reference_wrapper(F._perspective_image_pil),
            reference_inputs_fn=reference_inputs_perspective_image_tensor,
            float32_vs_uint8=float32_vs_uint8_fill_adapter,
            closeness_kwargs={
                **pil_reference_pixel_difference(2, mae=True),
                **cuda_vs_cpu_pixel_difference(),
                **float32_vs_uint8_pixel_difference(),
                **scripted_vs_eager_float64_tolerances("cpu", atol=1e-5, rtol=1e-5),
                **scripted_vs_eager_float64_tolerances("cuda", atol=1e-5, rtol=1e-5),
            },
            test_marks=[xfail_jit_python_scalar_arg("fill")],
        ),
        KernelInfo(
            F.perspective_bounding_boxes,
            sample_inputs_fn=sample_inputs_perspective_bounding_boxes,
            closeness_kwargs={
                **scripted_vs_eager_float64_tolerances("cpu", atol=1e-6, rtol=1e-6),
                **scripted_vs_eager_float64_tolerances("cuda", atol=1e-6, rtol=1e-6),
            },
        ),
        KernelInfo(
            F.perspective_mask,
            sample_inputs_fn=sample_inputs_perspective_mask,
            reference_fn=pil_reference_wrapper(F._perspective_image_pil),
            reference_inputs_fn=reference_inputs_perspective_mask,
            float32_vs_uint8=True,
            closeness_kwargs={
                (("TestKernels", "test_against_reference"), torch.uint8, "cpu"): dict(atol=10, rtol=0),
            },
        ),
        KernelInfo(
            F.perspective_video,
            sample_inputs_fn=sample_inputs_perspective_video,
            closeness_kwargs={
                **cuda_vs_cpu_pixel_difference(),
                **scripted_vs_eager_float64_tolerances("cpu", atol=1e-5, rtol=1e-5),
                **scripted_vs_eager_float64_tolerances("cuda", atol=1e-5, rtol=1e-5),
            },
        ),
    ]
)


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


_CENTER_CROP_SPATIAL_SIZES = [(16, 16), (7, 33), (31, 9)]
_CENTER_CROP_OUTPUT_SIZES = [[4, 3], [42, 70], [4], 3, (5, 2), (6,)]


def sample_inputs_center_crop_image_tensor():
    for image_loader, output_size in itertools.product(
        make_image_loaders(sizes=[(16, 17)], color_spaces=["RGB"], dtypes=[torch.float32]),
        [
            # valid `output_size` types for which cropping is applied to both dimensions
            *[5, (4,), (2, 3), [6], [3, 2]],
            # `output_size`'s for which at least one dimension needs to be padded
            *[[4, 18], [17, 5], [17, 18]],
        ],
    ):
        yield ArgsKwargs(image_loader, output_size=output_size)


def reference_inputs_center_crop_image_tensor():
    for image_loader, output_size in itertools.product(
        make_image_loaders(sizes=_CENTER_CROP_SPATIAL_SIZES, extra_dims=[()], dtypes=[torch.uint8]),
        _CENTER_CROP_OUTPUT_SIZES,
    ):
        yield ArgsKwargs(image_loader, output_size=output_size)


def sample_inputs_center_crop_bounding_boxes():
    for bounding_boxes_loader, output_size in itertools.product(make_bounding_box_loaders(), _CENTER_CROP_OUTPUT_SIZES):
        yield ArgsKwargs(
            bounding_boxes_loader,
            format=bounding_boxes_loader.format,
            canvas_size=bounding_boxes_loader.canvas_size,
            output_size=output_size,
        )


def sample_inputs_center_crop_mask():
    for mask_loader in make_mask_loaders(sizes=[DEFAULT_PORTRAIT_SPATIAL_SIZE], num_categories=[10], num_objects=[5]):
        height, width = mask_loader.shape[-2:]
        yield ArgsKwargs(mask_loader, output_size=(height // 2, width // 2))


def reference_inputs_center_crop_mask():
    for mask_loader, output_size in itertools.product(
        make_mask_loaders(sizes=_CENTER_CROP_SPATIAL_SIZES, extra_dims=[()], num_objects=[1]), _CENTER_CROP_OUTPUT_SIZES
    ):
        yield ArgsKwargs(mask_loader, output_size=output_size)


def sample_inputs_center_crop_video():
    for video_loader in make_video_loaders(sizes=[DEFAULT_PORTRAIT_SPATIAL_SIZE], num_frames=[3]):
        height, width = video_loader.shape[-2:]
        yield ArgsKwargs(video_loader, output_size=(height // 2, width // 2))


KERNEL_INFOS.extend(
    [
        KernelInfo(
            F.center_crop_image,
            sample_inputs_fn=sample_inputs_center_crop_image_tensor,
            reference_fn=pil_reference_wrapper(F._center_crop_image_pil),
            reference_inputs_fn=reference_inputs_center_crop_image_tensor,
            float32_vs_uint8=True,
            test_marks=[
                xfail_jit_python_scalar_arg("output_size"),
            ],
        ),
        KernelInfo(
            F.center_crop_bounding_boxes,
            sample_inputs_fn=sample_inputs_center_crop_bounding_boxes,
            test_marks=[
                xfail_jit_python_scalar_arg("output_size"),
            ],
        ),
        KernelInfo(
            F.center_crop_mask,
            sample_inputs_fn=sample_inputs_center_crop_mask,
            reference_fn=pil_reference_wrapper(F._center_crop_image_pil),
            reference_inputs_fn=reference_inputs_center_crop_mask,
            float32_vs_uint8=True,
            test_marks=[
                xfail_jit_python_scalar_arg("output_size"),
            ],
        ),
        KernelInfo(
            F.center_crop_video,
            sample_inputs_fn=sample_inputs_center_crop_video,
        ),
    ]
)


def sample_inputs_gaussian_blur_image_tensor():
    make_gaussian_blur_image_loaders = functools.partial(make_image_loaders, sizes=[(7, 33)], color_spaces=["RGB"])

    for image_loader, kernel_size in itertools.product(make_gaussian_blur_image_loaders(), [5, (3, 3), [3, 3]]):
        yield ArgsKwargs(image_loader, kernel_size=kernel_size)

    for image_loader, sigma in itertools.product(
        make_gaussian_blur_image_loaders(), [None, (3.0, 3.0), [2.0, 2.0], 4.0, [1.5], (3.14,)]
    ):
        yield ArgsKwargs(image_loader, kernel_size=5, sigma=sigma)


def sample_inputs_gaussian_blur_video():
    for video_loader in make_video_loaders(sizes=[(7, 33)], num_frames=[5]):
        yield ArgsKwargs(video_loader, kernel_size=[3, 3])


KERNEL_INFOS.extend(
    [
        KernelInfo(
            F.gaussian_blur_image,
            sample_inputs_fn=sample_inputs_gaussian_blur_image_tensor,
            closeness_kwargs=cuda_vs_cpu_pixel_difference(),
            test_marks=[
                xfail_jit_python_scalar_arg("kernel_size"),
                xfail_jit_python_scalar_arg("sigma"),
            ],
        ),
        KernelInfo(
            F.gaussian_blur_video,
            sample_inputs_fn=sample_inputs_gaussian_blur_video,
            closeness_kwargs=cuda_vs_cpu_pixel_difference(),
        ),
    ]
)


def sample_inputs_equalize_image_tensor():
    for image_loader in make_image_loaders(sizes=[DEFAULT_PORTRAIT_SPATIAL_SIZE], color_spaces=("GRAY", "RGB")):
        yield ArgsKwargs(image_loader)


def reference_inputs_equalize_image_tensor():
    # We are not using `make_image_loaders` here since that uniformly samples the values over the whole value range.
    # Since the whole point of this kernel is to transform an arbitrary distribution of values into a uniform one,
    # the information gain is low if we already provide something really close to the expected value.
    def make_uniform_band_image(shape, dtype, device, *, low_factor, high_factor, memory_format):
        if dtype.is_floating_point:
            low = low_factor
            high = high_factor
        else:
            max_value = torch.iinfo(dtype).max
            low = int(low_factor * max_value)
            high = int(high_factor * max_value)
        return torch.testing.make_tensor(shape, dtype=dtype, device=device, low=low, high=high).to(
            memory_format=memory_format, copy=True
        )

    def make_beta_distributed_image(shape, dtype, device, *, alpha, beta, memory_format):
        image = torch.distributions.Beta(alpha, beta).sample(shape)
        if not dtype.is_floating_point:
            image.mul_(torch.iinfo(dtype).max).round_()
        return image.to(dtype=dtype, device=device, memory_format=memory_format, copy=True)

    canvas_size = (256, 256)
    for dtype, color_space, fn in itertools.product(
        [torch.uint8],
        ["GRAY", "RGB"],
        [
            lambda shape, dtype, device, memory_format: torch.zeros(shape, dtype=dtype, device=device).to(
                memory_format=memory_format, copy=True
            ),
            lambda shape, dtype, device, memory_format: torch.full(
                shape, 1.0 if dtype.is_floating_point else torch.iinfo(dtype).max, dtype=dtype, device=device
            ).to(memory_format=memory_format, copy=True),
            *[
                functools.partial(make_uniform_band_image, low_factor=low_factor, high_factor=high_factor)
                for low_factor, high_factor in [
                    (0.0, 0.25),
                    (0.25, 0.75),
                    (0.75, 1.0),
                ]
            ],
            *[
                functools.partial(make_beta_distributed_image, alpha=alpha, beta=beta)
                for alpha, beta in [
                    (0.5, 0.5),
                    (2, 2),
                    (2, 5),
                    (5, 2),
                ]
            ],
        ],
    ):
        image_loader = ImageLoader(fn, shape=(get_num_channels(color_space), *canvas_size), dtype=dtype)
        yield ArgsKwargs(image_loader)


def sample_inputs_equalize_video():
    for video_loader in make_video_loaders(sizes=[DEFAULT_PORTRAIT_SPATIAL_SIZE], num_frames=[3]):
        yield ArgsKwargs(video_loader)


KERNEL_INFOS.extend(
    [
        KernelInfo(
            F.equalize_image,
            kernel_name="equalize_image_tensor",
            sample_inputs_fn=sample_inputs_equalize_image_tensor,
            reference_fn=pil_reference_wrapper(F._equalize_image_pil),
            float32_vs_uint8=True,
            reference_inputs_fn=reference_inputs_equalize_image_tensor,
        ),
        KernelInfo(
            F.equalize_video,
            sample_inputs_fn=sample_inputs_equalize_video,
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


def sample_inputs_erase_image_tensor():
    for image_loader in make_image_loaders(sizes=[DEFAULT_PORTRAIT_SPATIAL_SIZE]):
        # FIXME: make the parameters more diverse
        h, w = 6, 7
        v = torch.rand(image_loader.num_channels, h, w)
        yield ArgsKwargs(image_loader, i=1, j=2, h=h, w=w, v=v)


def sample_inputs_erase_video():
    for video_loader in make_video_loaders(sizes=[DEFAULT_PORTRAIT_SPATIAL_SIZE], num_frames=[3]):
        # FIXME: make the parameters more diverse
        h, w = 6, 7
        v = torch.rand(video_loader.num_channels, h, w)
        yield ArgsKwargs(video_loader, i=1, j=2, h=h, w=w, v=v)


KERNEL_INFOS.extend(
    [
        KernelInfo(
            F.erase_image,
            kernel_name="erase_image_tensor",
            sample_inputs_fn=sample_inputs_erase_image_tensor,
        ),
        KernelInfo(
            F.erase_video,
            sample_inputs_fn=sample_inputs_erase_video,
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
            F.to_dtype_image(F.to_image_tensor(output_pil), dtype=input_tensor.dtype, scale=True)
            for output_pil in output
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

_NORMALIZE_MEANS_STDS = [
    ((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    ([0.0, 0.0, 0.0], [1.0, 1.0, 1.0]),
    (0.5, 2.0),
]


def sample_inputs_normalize_image_tensor():
    for image_loader, (mean, std) in itertools.product(
        make_image_loaders(sizes=[DEFAULT_PORTRAIT_SPATIAL_SIZE], color_spaces=["RGB"], dtypes=[torch.float32]),
        _NORMALIZE_MEANS_STDS,
    ):
        yield ArgsKwargs(image_loader, mean=mean, std=std)


def reference_normalize_image_tensor(image, mean, std, inplace=False):
    mean = torch.tensor(mean).view(-1, 1, 1)
    std = torch.tensor(std).view(-1, 1, 1)

    sub = torch.Tensor.sub_ if inplace else torch.Tensor.sub
    return sub(image, mean).div_(std)


def reference_inputs_normalize_image_tensor():
    yield ArgsKwargs(
        make_image_loader(size=(32, 32), color_space="RGB", extra_dims=[1]),
        mean=[0.5, 0.5, 0.5],
        std=[1.0, 1.0, 1.0],
    )


def sample_inputs_normalize_video():
    mean, std = _NORMALIZE_MEANS_STDS[0]
    for video_loader in make_video_loaders(
        sizes=[DEFAULT_PORTRAIT_SPATIAL_SIZE], color_spaces=["RGB"], num_frames=[3], dtypes=[torch.float32]
    ):
        yield ArgsKwargs(video_loader, mean=mean, std=std)


KERNEL_INFOS.extend(
    [
        KernelInfo(
            F.normalize_image,
            kernel_name="normalize_image_tensor",
            sample_inputs_fn=sample_inputs_normalize_image_tensor,
            reference_fn=reference_normalize_image_tensor,
            reference_inputs_fn=reference_inputs_normalize_image_tensor,
            test_marks=[
                xfail_jit_python_scalar_arg("mean"),
                xfail_jit_python_scalar_arg("std"),
            ],
        ),
        KernelInfo(
            F.normalize_video,
            sample_inputs_fn=sample_inputs_normalize_video,
        ),
    ]
)


def sample_inputs_uniform_temporal_subsample_video():
    for video_loader in make_video_loaders(sizes=[DEFAULT_PORTRAIT_SPATIAL_SIZE], num_frames=[4]):
        yield ArgsKwargs(video_loader, num_samples=2)


def reference_uniform_temporal_subsample_video(x, num_samples):
    # Copy-pasted from
    # https://github.com/facebookresearch/pytorchvideo/blob/c8d23d8b7e597586a9e2d18f6ed31ad8aa379a7a/pytorchvideo/transforms/functional.py#L19
    t = x.shape[-4]
    assert num_samples > 0 and t > 0
    # Sample by nearest neighbor interpolation if num_samples > t.
    indices = torch.linspace(0, t - 1, num_samples)
    indices = torch.clamp(indices, 0, t - 1).long()
    return torch.index_select(x, -4, indices)


def reference_inputs_uniform_temporal_subsample_video():
    for video_loader in make_video_loaders(
        sizes=[DEFAULT_PORTRAIT_SPATIAL_SIZE], color_spaces=["RGB"], num_frames=[10]
    ):
        for num_samples in range(1, video_loader.shape[-4] + 1):
            yield ArgsKwargs(video_loader, num_samples)


KERNEL_INFOS.append(
    KernelInfo(
        F.uniform_temporal_subsample_video,
        sample_inputs_fn=sample_inputs_uniform_temporal_subsample_video,
        reference_fn=reference_uniform_temporal_subsample_video,
        reference_inputs_fn=reference_inputs_uniform_temporal_subsample_video,
    )
)
