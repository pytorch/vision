import decimal
import functools
import itertools
import math
import re

import numpy as np
import pytest
import torch.testing
import torchvision.ops
import torchvision.prototype.transforms.functional as F
from common_utils import cycle_over
from datasets_utils import combinations_grid
from prototype_common_utils import (
    ArgsKwargs,
    InfoBase,
    make_bounding_box_loaders,
    make_image_loader,
    make_image_loaders,
    make_mask_loaders,
    make_video_loaders,
    mark_framework_limitation,
    TestMark,
)
from torch.utils._pytree import tree_map
from torchvision.prototype import features
from torchvision.transforms.functional_tensor import _max_value as get_max_value

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


DEFAULT_IMAGE_CLOSENESS_KWARGS = dict(
    atol=1e-5,
    rtol=0,
    agg_method="mean",
)


def pil_reference_wrapper(pil_kernel):
    @functools.wraps(pil_kernel)
    def wrapper(image_tensor, *other_args, **kwargs):
        if image_tensor.ndim > 3:
            raise pytest.UsageError(
                f"Can only test single tensor images against PIL, but input has shape {image_tensor.shape}"
            )

        # We don't need to convert back to tensor here, since `assert_close` does that automatically.
        return pil_kernel(F.to_image_pil(image_tensor), *other_args, **kwargs)

    return wrapper


def xfail_jit_python_scalar_arg(name, *, reason=None):
    reason = reason or f"Python scalar int or float for `{name}` is not supported when scripting"
    return TestMark(
        ("TestKernels", "test_scripted_vs_eager"),
        pytest.mark.xfail(reason=reason),
        condition=lambda args_kwargs: isinstance(args_kwargs.kwargs.get(name), (int, float)),
    )


def xfail_jit_integer_size(name="size"):
    return xfail_jit_python_scalar_arg(name, reason=f"Integer `{name}` is not supported when scripting.")


def xfail_jit_tuple_instead_of_list(name, *, reason=None):
    reason = reason or f"Passing a tuple instead of a list for `{name}` is not supported when scripting"
    return TestMark(
        ("TestKernels", "test_scripted_vs_eager"),
        pytest.mark.xfail(reason=reason),
        condition=lambda args_kwargs: isinstance(args_kwargs.kwargs.get(name), tuple),
    )


def is_list_of_ints(args_kwargs):
    fill = args_kwargs.kwargs.get("fill")
    return isinstance(fill, list) and any(isinstance(scalar_fill, int) for scalar_fill in fill)


def xfail_jit_list_of_ints(name, *, reason=None):
    reason = reason or f"Passing a list of integers for `{name}` is not supported when scripting"
    return TestMark(
        ("TestKernels", "test_scripted_vs_eager"),
        pytest.mark.xfail(reason=reason),
        condition=is_list_of_ints,
    )


def xfail_all_tests(*, reason, condition):
    return [
        TestMark(("TestKernels", test_name), pytest.mark.xfail(reason=reason), condition=condition)
        for test_name in [
            "test_scripted_vs_eager",
            "test_batched_vs_single",
            "test_no_inplace",
            "test_cuda_vs_cpu",
            "test_dtype_and_device_consistency",
        ]
    ]


KERNEL_INFOS = []


def sample_inputs_horizontal_flip_image_tensor():
    for image_loader in make_image_loaders(sizes=["random"], dtypes=[torch.float32]):
        yield ArgsKwargs(image_loader)


def reference_inputs_horizontal_flip_image_tensor():
    for image_loader in make_image_loaders(extra_dims=[()]):
        yield ArgsKwargs(image_loader)


def sample_inputs_horizontal_flip_bounding_box():
    for bounding_box_loader in make_bounding_box_loaders(
        formats=[features.BoundingBoxFormat.XYXY], dtypes=[torch.float32]
    ):
        yield ArgsKwargs(
            bounding_box_loader, format=bounding_box_loader.format, spatial_size=bounding_box_loader.spatial_size
        )


def sample_inputs_horizontal_flip_mask():
    for image_loader in make_mask_loaders(sizes=["random"], dtypes=[torch.uint8]):
        yield ArgsKwargs(image_loader)


def sample_inputs_horizontal_flip_video():
    for video_loader in make_video_loaders(sizes=["random"], num_frames=["random"]):
        yield ArgsKwargs(video_loader)


KERNEL_INFOS.extend(
    [
        KernelInfo(
            F.horizontal_flip_image_tensor,
            kernel_name="horizontal_flip_image_tensor",
            sample_inputs_fn=sample_inputs_horizontal_flip_image_tensor,
            reference_fn=pil_reference_wrapper(F.horizontal_flip_image_pil),
            reference_inputs_fn=reference_inputs_horizontal_flip_image_tensor,
            closeness_kwargs=DEFAULT_IMAGE_CLOSENESS_KWARGS,
        ),
        KernelInfo(
            F.horizontal_flip_bounding_box,
            sample_inputs_fn=sample_inputs_horizontal_flip_bounding_box,
            test_marks=[
                TestMark(
                    ("TestKernels", "test_scripted_vs_eager"),
                    pytest.mark.filterwarnings(f"ignore:{re.escape('operator() profile_node %72')}:UserWarning"),
                )
            ],
        ),
        KernelInfo(
            F.horizontal_flip_mask,
            sample_inputs_fn=sample_inputs_horizontal_flip_mask,
        ),
        KernelInfo(
            F.horizontal_flip_video,
            sample_inputs_fn=sample_inputs_horizontal_flip_video,
        ),
    ]
)


def _get_resize_sizes(spatial_size):
    height, width = spatial_size
    length = max(spatial_size)
    yield length
    yield [length]
    yield (length,)
    new_height = int(height * 0.75)
    new_width = int(width * 1.25)
    yield [new_height, new_width]
    yield height, width


def sample_inputs_resize_image_tensor():
    for image_loader in make_image_loaders(
        sizes=["random"], color_spaces=[features.ColorSpace.RGB], dtypes=[torch.float32]
    ):
        for size in _get_resize_sizes(image_loader.spatial_size):
            yield ArgsKwargs(image_loader, size=size)

    for image_loader, interpolation in itertools.product(
        make_image_loaders(sizes=["random"], color_spaces=[features.ColorSpace.RGB]),
        [
            F.InterpolationMode.NEAREST,
            F.InterpolationMode.BILINEAR,
            F.InterpolationMode.BICUBIC,
        ],
    ):
        yield ArgsKwargs(image_loader, size=[min(image_loader.spatial_size) + 1], interpolation=interpolation)

    yield ArgsKwargs(make_image_loader(size=(11, 17)), size=20, max_size=25)


@pil_reference_wrapper
def reference_resize_image_tensor(*args, **kwargs):
    if not kwargs.pop("antialias", False) and kwargs.get("interpolation", F.InterpolationMode.BILINEAR) in {
        F.InterpolationMode.BILINEAR,
        F.InterpolationMode.BICUBIC,
    }:
        raise pytest.UsageError("Anti-aliasing is always active in PIL")
    return F.resize_image_pil(*args, **kwargs)


def reference_inputs_resize_image_tensor():
    for image_loader, interpolation in itertools.product(
        make_image_loaders(extra_dims=[()]),
        [
            F.InterpolationMode.NEAREST,
            F.InterpolationMode.NEAREST_EXACT,
            F.InterpolationMode.BILINEAR,
            F.InterpolationMode.BICUBIC,
        ],
    ):
        for size in _get_resize_sizes(image_loader.spatial_size):
            yield ArgsKwargs(
                image_loader,
                size=size,
                interpolation=interpolation,
                antialias=interpolation
                in {
                    F.InterpolationMode.BILINEAR,
                    F.InterpolationMode.BICUBIC,
                },
            )


def sample_inputs_resize_bounding_box():
    for bounding_box_loader in make_bounding_box_loaders():
        for size in _get_resize_sizes(bounding_box_loader.spatial_size):
            yield ArgsKwargs(bounding_box_loader, size=size, spatial_size=bounding_box_loader.spatial_size)


def sample_inputs_resize_mask():
    for mask_loader in make_mask_loaders(sizes=["random"], num_categories=["random"], num_objects=["random"]):
        yield ArgsKwargs(mask_loader, size=[min(mask_loader.shape[-2:]) + 1])


@pil_reference_wrapper
def reference_resize_mask(*args, **kwargs):
    return F.resize_image_pil(*args, interpolation=F.InterpolationMode.NEAREST, **kwargs)


def reference_inputs_resize_mask():
    for mask_loader in make_mask_loaders(extra_dims=[()], num_objects=[1]):
        for size in _get_resize_sizes(mask_loader.shape[-2:]):
            yield ArgsKwargs(mask_loader, size=size)


def sample_inputs_resize_video():
    for video_loader in make_video_loaders(sizes=["random"], num_frames=["random"]):
        yield ArgsKwargs(video_loader, size=[min(video_loader.shape[-2:]) + 1])


KERNEL_INFOS.extend(
    [
        KernelInfo(
            F.resize_image_tensor,
            sample_inputs_fn=sample_inputs_resize_image_tensor,
            reference_fn=reference_resize_image_tensor,
            reference_inputs_fn=reference_inputs_resize_image_tensor,
            closeness_kwargs=DEFAULT_IMAGE_CLOSENESS_KWARGS,
            test_marks=[
                xfail_jit_integer_size(),
            ],
        ),
        KernelInfo(
            F.resize_bounding_box,
            sample_inputs_fn=sample_inputs_resize_bounding_box,
            test_marks=[
                xfail_jit_integer_size(),
            ],
        ),
        KernelInfo(
            F.resize_mask,
            sample_inputs_fn=sample_inputs_resize_mask,
            reference_fn=reference_resize_mask,
            reference_inputs_fn=reference_inputs_resize_mask,
            closeness_kwargs=DEFAULT_IMAGE_CLOSENESS_KWARGS,
            test_marks=[
                xfail_jit_integer_size(),
            ],
        ),
        KernelInfo(
            F.resize_video,
            sample_inputs_fn=sample_inputs_resize_video,
        ),
    ]
)


_AFFINE_KWARGS = combinations_grid(
    angle=[-87, 15, 90],
    translate=[(5, 5), (-5, -5)],
    scale=[0.77, 1.27],
    shear=[(12, 12), (0, 0)],
)


def _diversify_affine_kwargs_types(affine_kwargs):
    angle = affine_kwargs["angle"]
    for diverse_angle in [int(angle), float(angle)]:
        yield dict(affine_kwargs, angle=diverse_angle)

    shear = affine_kwargs["shear"]
    for diverse_shear in [tuple(shear), list(shear), int(shear[0]), float(shear[0])]:
        yield dict(affine_kwargs, shear=diverse_shear)


def _full_affine_params(**partial_params):
    partial_params.setdefault("angle", 0.0)
    partial_params.setdefault("translate", [0.0, 0.0])
    partial_params.setdefault("scale", 1.0)
    partial_params.setdefault("shear", [0.0, 0.0])
    partial_params.setdefault("center", None)
    return partial_params


_DIVERSE_AFFINE_PARAMS = [
    _full_affine_params(**{name: arg})
    for name, args in [
        ("angle", [1.0, 2]),
        ("translate", [[1.0, 0.5], [1, 2], (1.0, 0.5), (1, 2)]),
        ("scale", [0.5]),
        ("shear", [1.0, 2, [1.0], [2], (1.0,), (2,), [1.0, 0.5], [1, 2], (1.0, 0.5), (1, 2)]),
        ("center", [None, [1.0, 0.5], [1, 2], (1.0, 0.5), (1, 2)]),
    ]
    for arg in args
]


def sample_inputs_affine_image_tensor():
    make_affine_image_loaders = functools.partial(
        make_image_loaders, sizes=["random"], color_spaces=[features.ColorSpace.RGB], dtypes=[torch.float32]
    )

    for image_loader, affine_params in itertools.product(make_affine_image_loaders(), _DIVERSE_AFFINE_PARAMS):
        yield ArgsKwargs(image_loader, **affine_params)

    for image_loader in make_affine_image_loaders():
        fills = [None, 0.5]
        if image_loader.num_channels > 1:
            fills.extend(vector_fill * image_loader.num_channels for vector_fill in [(0.5,), (1,), [0.5], [1]])
        for fill in fills:
            yield ArgsKwargs(image_loader, **_full_affine_params(), fill=fill)

    for image_loader, interpolation in itertools.product(
        make_affine_image_loaders(),
        [
            F.InterpolationMode.NEAREST,
            F.InterpolationMode.BILINEAR,
        ],
    ):
        yield ArgsKwargs(image_loader, **_full_affine_params(), fill=0)


def reference_inputs_affine_image_tensor():
    for image_loader, affine_kwargs in itertools.product(make_image_loaders(extra_dims=[()]), _AFFINE_KWARGS):
        yield ArgsKwargs(
            image_loader,
            interpolation=F.InterpolationMode.NEAREST,
            **affine_kwargs,
        )


def sample_inputs_affine_bounding_box():
    for bounding_box_loader, affine_params in itertools.product(
        make_bounding_box_loaders(formats=[features.BoundingBoxFormat.XYXY]), _DIVERSE_AFFINE_PARAMS
    ):
        yield ArgsKwargs(
            bounding_box_loader,
            format=bounding_box_loader.format,
            spatial_size=bounding_box_loader.spatial_size,
            **affine_params,
        )


def _compute_affine_matrix(angle, translate, scale, shear, center):
    rot = math.radians(angle)
    cx, cy = center
    tx, ty = translate
    sx, sy = [math.radians(sh_) for sh_ in shear]

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


def reference_affine_bounding_box(bounding_box, *, format, spatial_size, angle, translate, scale, shear, center=None):
    if center is None:
        center = [s * 0.5 for s in spatial_size[::-1]]

    def transform(bbox):
        affine_matrix = _compute_affine_matrix(angle, translate, scale, shear, center)
        affine_matrix = affine_matrix[:2, :]

        bbox_xyxy = F.convert_format_bounding_box(bbox, old_format=format, new_format=features.BoundingBoxFormat.XYXY)
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
            dtype=bbox.dtype,
        )
        return F.convert_format_bounding_box(
            out_bbox, old_format=features.BoundingBoxFormat.XYXY, new_format=format, copy=False
        )

    if bounding_box.ndim < 2:
        bounding_box = [bounding_box]

    expected_bboxes = [transform(bbox) for bbox in bounding_box]
    if len(expected_bboxes) > 1:
        expected_bboxes = torch.stack(expected_bboxes)
    else:
        expected_bboxes = expected_bboxes[0]

    return expected_bboxes


def reference_inputs_affine_bounding_box():
    for bounding_box_loader, affine_kwargs in itertools.product(
        make_bounding_box_loaders(extra_dims=[()]),
        _AFFINE_KWARGS,
    ):
        yield ArgsKwargs(
            bounding_box_loader,
            format=bounding_box_loader.format,
            spatial_size=bounding_box_loader.spatial_size,
            **affine_kwargs,
        )


def sample_inputs_affine_mask():
    for mask_loader in make_mask_loaders(sizes=["random"], num_categories=["random"], num_objects=["random"]):
        yield ArgsKwargs(mask_loader, **_full_affine_params())


@pil_reference_wrapper
def reference_affine_mask(*args, **kwargs):
    return F.affine_image_pil(*args, interpolation=F.InterpolationMode.NEAREST, **kwargs)


def reference_inputs_resize_mask():
    for mask_loader, affine_kwargs in itertools.product(
        make_mask_loaders(extra_dims=[()], num_objects=[1]), _AFFINE_KWARGS
    ):
        yield ArgsKwargs(mask_loader, **affine_kwargs)


def sample_inputs_affine_video():
    for video_loader in make_video_loaders(sizes=["random"], num_frames=["random"]):
        yield ArgsKwargs(video_loader, **_full_affine_params())


KERNEL_INFOS.extend(
    [
        KernelInfo(
            F.affine_image_tensor,
            sample_inputs_fn=sample_inputs_affine_image_tensor,
            reference_fn=pil_reference_wrapper(F.affine_image_pil),
            reference_inputs_fn=reference_inputs_affine_image_tensor,
            closeness_kwargs=DEFAULT_IMAGE_CLOSENESS_KWARGS,
            test_marks=[
                xfail_jit_python_scalar_arg("shear"),
                xfail_jit_tuple_instead_of_list("fill"),
                # TODO: check if this is a regression since it seems that should be supported if `int` is ok
                xfail_jit_list_of_ints("fill"),
            ],
        ),
        KernelInfo(
            F.affine_bounding_box,
            sample_inputs_fn=sample_inputs_affine_bounding_box,
            reference_fn=reference_affine_bounding_box,
            reference_inputs_fn=reference_inputs_affine_bounding_box,
            closeness_kwargs=dict(atol=1, rtol=0),
            test_marks=[
                xfail_jit_python_scalar_arg("shear"),
            ],
        ),
        KernelInfo(
            F.affine_mask,
            sample_inputs_fn=sample_inputs_affine_mask,
            reference_fn=reference_affine_mask,
            reference_inputs_fn=reference_inputs_resize_mask,
            closeness_kwargs=DEFAULT_IMAGE_CLOSENESS_KWARGS,
            test_marks=[
                xfail_jit_python_scalar_arg("shear"),
            ],
        ),
        KernelInfo(
            F.affine_video,
            sample_inputs_fn=sample_inputs_affine_video,
        ),
    ]
)


def sample_inputs_convert_format_bounding_box():
    formats = set(features.BoundingBoxFormat)
    for bounding_box_loader in make_bounding_box_loaders(formats=formats):
        old_format = bounding_box_loader.format
        for params in combinations_grid(new_format=formats - {old_format}, copy=(True, False)):
            yield ArgsKwargs(bounding_box_loader, old_format=old_format, **params)


def reference_convert_format_bounding_box(bounding_box, old_format, new_format, copy):
    if not copy:
        raise pytest.UsageError("Reference for `convert_format_bounding_box` only supports `copy=True`")

    return torchvision.ops.box_convert(
        bounding_box, in_fmt=old_format.kernel_name.lower(), out_fmt=new_format.kernel_name.lower()
    )


def reference_inputs_convert_format_bounding_box():
    for args_kwargs in sample_inputs_convert_color_space_image_tensor():
        (image_loader, *other_args), kwargs = args_kwargs
        if len(image_loader.shape) == 2 and kwargs.setdefault("copy", True):
            yield args_kwargs


KERNEL_INFOS.append(
    KernelInfo(
        F.convert_format_bounding_box,
        sample_inputs_fn=sample_inputs_convert_format_bounding_box,
        reference_fn=reference_convert_format_bounding_box,
        reference_inputs_fn=reference_inputs_convert_format_bounding_box,
    ),
)


def sample_inputs_convert_color_space_image_tensor():
    color_spaces = list(set(features.ColorSpace) - {features.ColorSpace.OTHER})

    for old_color_space, new_color_space in cycle_over(color_spaces):
        for image_loader in make_image_loaders(sizes=["random"], color_spaces=[old_color_space], constant_alpha=True):
            yield ArgsKwargs(image_loader, old_color_space=old_color_space, new_color_space=new_color_space)

    for color_space in color_spaces:
        for image_loader in make_image_loaders(
            sizes=["random"], color_spaces=[color_space], dtypes=[torch.float32], constant_alpha=True
        ):
            yield ArgsKwargs(image_loader, old_color_space=color_space, new_color_space=color_space, copy=False)


@pil_reference_wrapper
def reference_convert_color_space_image_tensor(image_pil, old_color_space, new_color_space, copy=True):
    color_space_pil = features.ColorSpace.from_pil_mode(image_pil.mode)
    if color_space_pil != old_color_space:
        raise pytest.UsageError(
            f"Converting the tensor image into an PIL image changed the colorspace "
            f"from {old_color_space} to {color_space_pil}"
        )

    return F.convert_color_space_image_pil(image_pil, color_space=new_color_space, copy=copy)


def reference_inputs_convert_color_space_image_tensor():
    for args_kwargs in sample_inputs_convert_color_space_image_tensor():
        (image_loader, *other_args), kwargs = args_kwargs
        if len(image_loader.shape) == 3:
            yield args_kwargs


def sample_inputs_convert_color_space_video():
    color_spaces = [features.ColorSpace.GRAY, features.ColorSpace.RGB]

    for old_color_space, new_color_space in cycle_over(color_spaces):
        for video_loader in make_video_loaders(sizes=["random"], color_spaces=[old_color_space], num_frames=["random"]):
            yield ArgsKwargs(video_loader, old_color_space=old_color_space, new_color_space=new_color_space)


KERNEL_INFOS.extend(
    [
        KernelInfo(
            F.convert_color_space_image_tensor,
            sample_inputs_fn=sample_inputs_convert_color_space_image_tensor,
            reference_fn=reference_convert_color_space_image_tensor,
            reference_inputs_fn=reference_inputs_convert_color_space_image_tensor,
            closeness_kwargs=DEFAULT_IMAGE_CLOSENESS_KWARGS,
        ),
        KernelInfo(
            F.convert_color_space_video,
            sample_inputs_fn=sample_inputs_convert_color_space_video,
        ),
    ]
)


def sample_inputs_vertical_flip_image_tensor():
    for image_loader in make_image_loaders(sizes=["random"], dtypes=[torch.float32]):
        yield ArgsKwargs(image_loader)


def reference_inputs_vertical_flip_image_tensor():
    for image_loader in make_image_loaders(extra_dims=[()]):
        yield ArgsKwargs(image_loader)


def sample_inputs_vertical_flip_bounding_box():
    for bounding_box_loader in make_bounding_box_loaders(
        formats=[features.BoundingBoxFormat.XYXY], dtypes=[torch.float32]
    ):
        yield ArgsKwargs(
            bounding_box_loader, format=bounding_box_loader.format, spatial_size=bounding_box_loader.spatial_size
        )


def sample_inputs_vertical_flip_mask():
    for image_loader in make_mask_loaders(sizes=["random"], dtypes=[torch.uint8]):
        yield ArgsKwargs(image_loader)


def sample_inputs_vertical_flip_video():
    for video_loader in make_video_loaders(sizes=["random"], num_frames=["random"]):
        yield ArgsKwargs(video_loader)


KERNEL_INFOS.extend(
    [
        KernelInfo(
            F.vertical_flip_image_tensor,
            kernel_name="vertical_flip_image_tensor",
            sample_inputs_fn=sample_inputs_vertical_flip_image_tensor,
            reference_fn=pil_reference_wrapper(F.vertical_flip_image_pil),
            reference_inputs_fn=reference_inputs_vertical_flip_image_tensor,
            closeness_kwargs=DEFAULT_IMAGE_CLOSENESS_KWARGS,
        ),
        KernelInfo(
            F.vertical_flip_bounding_box,
            sample_inputs_fn=sample_inputs_vertical_flip_bounding_box,
        ),
        KernelInfo(
            F.vertical_flip_mask,
            sample_inputs_fn=sample_inputs_vertical_flip_mask,
        ),
        KernelInfo(
            F.vertical_flip_video,
            sample_inputs_fn=sample_inputs_vertical_flip_video,
        ),
    ]
)

_ROTATE_ANGLES = [-87, 15, 90]


def sample_inputs_rotate_image_tensor():
    make_rotate_image_loaders = functools.partial(
        make_image_loaders, sizes=["random"], color_spaces=[features.ColorSpace.RGB], dtypes=[torch.float32]
    )

    for image_loader in make_rotate_image_loaders():
        yield ArgsKwargs(image_loader, angle=15.0, expand=True)

    for image_loader, center in itertools.product(
        make_rotate_image_loaders(), [None, [1.0, 0.5], [1, 2], (1.0, 0.5), (1, 2)]
    ):
        yield ArgsKwargs(image_loader, angle=15.0, center=center)

    for image_loader in make_rotate_image_loaders():
        fills = [None, 0.5]
        if image_loader.num_channels > 1:
            fills.extend(vector_fill * image_loader.num_channels for vector_fill in [(0.5,), (1,), [0.5], [1]])
        for fill in fills:
            yield ArgsKwargs(image_loader, angle=15.0, fill=fill)

    for image_loader, interpolation in itertools.product(
        make_rotate_image_loaders(),
        [F.InterpolationMode.NEAREST, F.InterpolationMode.BILINEAR],
    ):
        yield ArgsKwargs(image_loader, angle=15.0, fill=0)


def reference_inputs_rotate_image_tensor():
    for image_loader, angle in itertools.product(make_image_loaders(extra_dims=[()]), _ROTATE_ANGLES):
        yield ArgsKwargs(image_loader, angle=angle)


def sample_inputs_rotate_bounding_box():
    for bounding_box_loader in make_bounding_box_loaders():
        yield ArgsKwargs(
            bounding_box_loader,
            format=bounding_box_loader.format,
            spatial_size=bounding_box_loader.spatial_size,
            angle=_ROTATE_ANGLES[0],
        )


def sample_inputs_rotate_mask():
    for mask_loader in make_mask_loaders(sizes=["random"], num_categories=["random"], num_objects=["random"]):
        yield ArgsKwargs(mask_loader, angle=15.0)


@pil_reference_wrapper
def reference_rotate_mask(*args, **kwargs):
    return F.rotate_image_pil(*args, interpolation=F.InterpolationMode.NEAREST, **kwargs)


def reference_inputs_rotate_mask():
    for mask_loader, angle in itertools.product(make_mask_loaders(extra_dims=[()], num_objects=[1]), _ROTATE_ANGLES):
        yield ArgsKwargs(mask_loader, angle=angle)


def sample_inputs_rotate_video():
    for video_loader in make_video_loaders(sizes=["random"], num_frames=["random"]):
        yield ArgsKwargs(video_loader, angle=15.0)


KERNEL_INFOS.extend(
    [
        KernelInfo(
            F.rotate_image_tensor,
            sample_inputs_fn=sample_inputs_rotate_image_tensor,
            reference_fn=pil_reference_wrapper(F.rotate_image_pil),
            reference_inputs_fn=reference_inputs_rotate_image_tensor,
            closeness_kwargs=DEFAULT_IMAGE_CLOSENESS_KWARGS,
            test_marks=[
                xfail_jit_tuple_instead_of_list("fill"),
                # TODO: check if this is a regression since it seems that should be supported if `int` is ok
                xfail_jit_list_of_ints("fill"),
            ],
        ),
        KernelInfo(
            F.rotate_bounding_box,
            sample_inputs_fn=sample_inputs_rotate_bounding_box,
        ),
        KernelInfo(
            F.rotate_mask,
            sample_inputs_fn=sample_inputs_rotate_mask,
            reference_fn=reference_rotate_mask,
            reference_inputs_fn=reference_inputs_rotate_mask,
            closeness_kwargs=DEFAULT_IMAGE_CLOSENESS_KWARGS,
        ),
        KernelInfo(
            F.rotate_video,
            sample_inputs_fn=sample_inputs_rotate_video,
        ),
    ]
)

_CROP_PARAMS = combinations_grid(top=[-8, 0, 9], left=[-8, 0, 9], height=[12, 20], width=[12, 20])


def sample_inputs_crop_image_tensor():
    for image_loader, params in itertools.product(
        make_image_loaders(sizes=[(16, 17)], color_spaces=[features.ColorSpace.RGB], dtypes=[torch.float32]),
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
    for image_loader, params in itertools.product(make_image_loaders(extra_dims=[()]), _CROP_PARAMS):
        yield ArgsKwargs(image_loader, **params)


def sample_inputs_crop_bounding_box():
    for bounding_box_loader, params in itertools.product(
        make_bounding_box_loaders(), [_CROP_PARAMS[0], _CROP_PARAMS[-1]]
    ):
        yield ArgsKwargs(bounding_box_loader, format=bounding_box_loader.format, **params)


def sample_inputs_crop_mask():
    for mask_loader in make_mask_loaders(sizes=[(16, 17)], num_categories=["random"], num_objects=["random"]):
        yield ArgsKwargs(mask_loader, top=4, left=3, height=7, width=8)


def reference_inputs_crop_mask():
    for mask_loader, params in itertools.product(make_mask_loaders(extra_dims=[()], num_objects=[1]), _CROP_PARAMS):
        yield ArgsKwargs(mask_loader, **params)


def sample_inputs_crop_video():
    for video_loader in make_video_loaders(sizes=[(16, 17)], num_frames=["random"]):
        yield ArgsKwargs(video_loader, top=4, left=3, height=7, width=8)


KERNEL_INFOS.extend(
    [
        KernelInfo(
            F.crop_image_tensor,
            kernel_name="crop_image_tensor",
            sample_inputs_fn=sample_inputs_crop_image_tensor,
            reference_fn=pil_reference_wrapper(F.crop_image_pil),
            reference_inputs_fn=reference_inputs_crop_image_tensor,
            closeness_kwargs=DEFAULT_IMAGE_CLOSENESS_KWARGS,
        ),
        KernelInfo(
            F.crop_bounding_box,
            sample_inputs_fn=sample_inputs_crop_bounding_box,
        ),
        KernelInfo(
            F.crop_mask,
            sample_inputs_fn=sample_inputs_crop_mask,
            reference_fn=pil_reference_wrapper(F.crop_image_pil),
            reference_inputs_fn=reference_inputs_crop_mask,
            closeness_kwargs=DEFAULT_IMAGE_CLOSENESS_KWARGS,
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
    return F.resized_crop_image_pil(*args, **kwargs)


def reference_inputs_resized_crop_image_tensor():
    for image_loader, interpolation, params in itertools.product(
        make_image_loaders(extra_dims=[()]),
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


def sample_inputs_resized_crop_bounding_box():
    for bounding_box_loader in make_bounding_box_loaders():
        yield ArgsKwargs(bounding_box_loader, format=bounding_box_loader.format, **_RESIZED_CROP_PARAMS[0])


def sample_inputs_resized_crop_mask():
    for mask_loader in make_mask_loaders():
        yield ArgsKwargs(mask_loader, **_RESIZED_CROP_PARAMS[0])


def reference_inputs_resized_crop_mask():
    for mask_loader, params in itertools.product(
        make_mask_loaders(extra_dims=[()], num_objects=[1]), _RESIZED_CROP_PARAMS
    ):
        yield ArgsKwargs(mask_loader, **params)


def sample_inputs_resized_crop_video():
    for video_loader in make_video_loaders(sizes=["random"], num_frames=["random"]):
        yield ArgsKwargs(video_loader, **_RESIZED_CROP_PARAMS[0])


KERNEL_INFOS.extend(
    [
        KernelInfo(
            F.resized_crop_image_tensor,
            sample_inputs_fn=sample_inputs_resized_crop_image_tensor,
            reference_fn=reference_resized_crop_image_tensor,
            reference_inputs_fn=reference_inputs_resized_crop_image_tensor,
            closeness_kwargs=DEFAULT_IMAGE_CLOSENESS_KWARGS,
        ),
        KernelInfo(
            F.resized_crop_bounding_box,
            sample_inputs_fn=sample_inputs_resized_crop_bounding_box,
        ),
        KernelInfo(
            F.resized_crop_mask,
            sample_inputs_fn=sample_inputs_resized_crop_mask,
            reference_fn=pil_reference_wrapper(F.resized_crop_image_pil),
            reference_inputs_fn=reference_inputs_resized_crop_mask,
            closeness_kwargs=DEFAULT_IMAGE_CLOSENESS_KWARGS,
        ),
        KernelInfo(
            F.resized_crop_video,
            sample_inputs_fn=sample_inputs_resized_crop_video,
        ),
    ]
)

_PAD_PARAMS = combinations_grid(
    padding=[[1], [1, 1], [1, 1, 2, 2]],
    padding_mode=["constant", "symmetric", "edge", "reflect"],
)


def sample_inputs_pad_image_tensor():
    make_pad_image_loaders = functools.partial(
        make_image_loaders, sizes=["random"], color_spaces=[features.ColorSpace.RGB], dtypes=[torch.float32]
    )

    for image_loader, padding in itertools.product(
        make_pad_image_loaders(),
        [1, (1,), (1, 2), (1, 2, 3, 4), [1], [1, 2], [1, 2, 3, 4]],
    ):
        yield ArgsKwargs(image_loader, padding=padding)

    for image_loader in make_pad_image_loaders():
        fills = [None, 0.5]
        if image_loader.num_channels > 1:
            fills.extend(vector_fill * image_loader.num_channels for vector_fill in [(0.5,), (1,), [0.5], [1]])
        for fill in fills:
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
    for image_loader, params in itertools.product(make_image_loaders(extra_dims=[()]), _PAD_PARAMS):
        # FIXME: PIL kernel doesn't support sequences of length 1 if the number of channels is larger. Shouldn't it?
        fills = [None, 128.0, 128]
        if params["padding_mode"] == "constant":
            fills.append([12.0 + c for c in range(image_loader.num_channels)])
        for fill in fills:
            yield ArgsKwargs(image_loader, fill=fill, **params)


def sample_inputs_pad_bounding_box():
    for bounding_box_loader, padding in itertools.product(
        make_bounding_box_loaders(), [1, (1,), (1, 2), (1, 2, 3, 4), [1], [1, 2], [1, 2, 3, 4]]
    ):
        yield ArgsKwargs(
            bounding_box_loader,
            format=bounding_box_loader.format,
            spatial_size=bounding_box_loader.spatial_size,
            padding=padding,
            padding_mode="constant",
        )


def sample_inputs_pad_mask():
    for mask_loader in make_mask_loaders(sizes=["random"], num_categories=["random"], num_objects=["random"]):
        yield ArgsKwargs(mask_loader, padding=[1])


def reference_inputs_pad_mask():
    for image_loader, fill, params in itertools.product(make_image_loaders(extra_dims=[()]), [None, 127], _PAD_PARAMS):
        yield ArgsKwargs(image_loader, fill=fill, **params)


def sample_inputs_pad_video():
    for video_loader in make_video_loaders(sizes=["random"], num_frames=["random"]):
        yield ArgsKwargs(video_loader, padding=[1])


KERNEL_INFOS.extend(
    [
        KernelInfo(
            F.pad_image_tensor,
            sample_inputs_fn=sample_inputs_pad_image_tensor,
            reference_fn=pil_reference_wrapper(F.pad_image_pil),
            reference_inputs_fn=reference_inputs_pad_image_tensor,
            closeness_kwargs=DEFAULT_IMAGE_CLOSENESS_KWARGS,
            test_marks=[
                xfail_jit_python_scalar_arg("padding"),
                xfail_jit_tuple_instead_of_list("padding"),
                xfail_jit_tuple_instead_of_list("fill"),
                # TODO: check if this is a regression since it seems that should be supported if `int` is ok
                xfail_jit_list_of_ints("fill"),
            ],
        ),
        KernelInfo(
            F.pad_bounding_box,
            sample_inputs_fn=sample_inputs_pad_bounding_box,
            test_marks=[
                xfail_jit_python_scalar_arg("padding"),
                xfail_jit_tuple_instead_of_list("padding"),
            ],
        ),
        KernelInfo(
            F.pad_mask,
            sample_inputs_fn=sample_inputs_pad_mask,
            reference_fn=pil_reference_wrapper(F.pad_image_pil),
            reference_inputs_fn=reference_inputs_pad_mask,
            closeness_kwargs=DEFAULT_IMAGE_CLOSENESS_KWARGS,
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


def sample_inputs_perspective_image_tensor():
    for image_loader in make_image_loaders(sizes=["random"]):
        for fill in [None, 128.0, 128, [12.0], [12.0 + c for c in range(image_loader.num_channels)]]:
            yield ArgsKwargs(image_loader, fill=fill, perspective_coeffs=_PERSPECTIVE_COEFFS[0])


def reference_inputs_perspective_image_tensor():
    for image_loader, perspective_coeffs in itertools.product(make_image_loaders(extra_dims=[()]), _PERSPECTIVE_COEFFS):
        # FIXME: PIL kernel doesn't support sequences of length 1 if the number of channels is larger. Shouldn't it?
        for fill in [None, 128.0, 128, [12.0 + c for c in range(image_loader.num_channels)]]:
            yield ArgsKwargs(image_loader, fill=fill, perspective_coeffs=perspective_coeffs)


def sample_inputs_perspective_bounding_box():
    for bounding_box_loader in make_bounding_box_loaders():
        yield ArgsKwargs(
            bounding_box_loader, format=bounding_box_loader.format, perspective_coeffs=_PERSPECTIVE_COEFFS[0]
        )


def sample_inputs_perspective_mask():
    for mask_loader in make_mask_loaders(sizes=["random"]):
        yield ArgsKwargs(mask_loader, perspective_coeffs=_PERSPECTIVE_COEFFS[0])


def reference_inputs_perspective_mask():
    for mask_loader, perspective_coeffs in itertools.product(
        make_mask_loaders(extra_dims=[()], num_objects=[1]), _PERSPECTIVE_COEFFS
    ):
        yield ArgsKwargs(mask_loader, perspective_coeffs=perspective_coeffs)


def sample_inputs_perspective_video():
    for video_loader in make_video_loaders(sizes=["random"], num_frames=["random"]):
        yield ArgsKwargs(video_loader, perspective_coeffs=_PERSPECTIVE_COEFFS[0])


KERNEL_INFOS.extend(
    [
        KernelInfo(
            F.perspective_image_tensor,
            sample_inputs_fn=sample_inputs_perspective_image_tensor,
            reference_fn=pil_reference_wrapper(F.perspective_image_pil),
            reference_inputs_fn=reference_inputs_perspective_image_tensor,
            closeness_kwargs=DEFAULT_IMAGE_CLOSENESS_KWARGS,
        ),
        KernelInfo(
            F.perspective_bounding_box,
            sample_inputs_fn=sample_inputs_perspective_bounding_box,
        ),
        KernelInfo(
            F.perspective_mask,
            sample_inputs_fn=sample_inputs_perspective_mask,
            reference_fn=pil_reference_wrapper(F.perspective_image_pil),
            reference_inputs_fn=reference_inputs_perspective_mask,
            closeness_kwargs=DEFAULT_IMAGE_CLOSENESS_KWARGS,
        ),
        KernelInfo(
            F.perspective_video,
            sample_inputs_fn=sample_inputs_perspective_video,
        ),
    ]
)


def _get_elastic_displacement(spatial_size):
    return torch.rand(1, *spatial_size, 2)


def sample_inputs_elastic_image_tensor():
    for image_loader in make_image_loaders(sizes=["random"]):
        displacement = _get_elastic_displacement(image_loader.spatial_size)
        for fill in [None, 128.0, 128, [12.0], [12.0 + c for c in range(image_loader.num_channels)]]:
            yield ArgsKwargs(image_loader, displacement=displacement, fill=fill)


def reference_inputs_elastic_image_tensor():
    for image_loader, interpolation in itertools.product(
        make_image_loaders(extra_dims=[()]),
        [
            F.InterpolationMode.NEAREST,
            F.InterpolationMode.BILINEAR,
            F.InterpolationMode.BICUBIC,
        ],
    ):
        displacement = _get_elastic_displacement(image_loader.spatial_size)
        for fill in [None, 128.0, 128, [12.0], [12.0 + c for c in range(image_loader.num_channels)]]:
            yield ArgsKwargs(image_loader, interpolation=interpolation, displacement=displacement, fill=fill)


def sample_inputs_elastic_bounding_box():
    for bounding_box_loader in make_bounding_box_loaders():
        displacement = _get_elastic_displacement(bounding_box_loader.spatial_size)
        yield ArgsKwargs(
            bounding_box_loader,
            format=bounding_box_loader.format,
            displacement=displacement,
        )


def sample_inputs_elastic_mask():
    for mask_loader in make_mask_loaders(sizes=["random"]):
        displacement = _get_elastic_displacement(mask_loader.shape[-2:])
        yield ArgsKwargs(mask_loader, displacement=displacement)


def reference_inputs_elastic_mask():
    for mask_loader in make_mask_loaders(extra_dims=[()], num_objects=[1]):
        displacement = _get_elastic_displacement(mask_loader.shape[-2:])
        yield ArgsKwargs(mask_loader, displacement=displacement)


def sample_inputs_elastic_video():
    for video_loader in make_video_loaders(sizes=["random"], num_frames=["random"]):
        displacement = _get_elastic_displacement(video_loader.shape[-2:])
        yield ArgsKwargs(video_loader, displacement=displacement)


KERNEL_INFOS.extend(
    [
        KernelInfo(
            F.elastic_image_tensor,
            sample_inputs_fn=sample_inputs_elastic_image_tensor,
            reference_fn=pil_reference_wrapper(F.elastic_image_pil),
            reference_inputs_fn=reference_inputs_elastic_image_tensor,
            closeness_kwargs=DEFAULT_IMAGE_CLOSENESS_KWARGS,
        ),
        KernelInfo(
            F.elastic_bounding_box,
            sample_inputs_fn=sample_inputs_elastic_bounding_box,
        ),
        KernelInfo(
            F.elastic_mask,
            sample_inputs_fn=sample_inputs_elastic_mask,
            reference_fn=pil_reference_wrapper(F.elastic_image_pil),
            reference_inputs_fn=reference_inputs_elastic_mask,
            closeness_kwargs=DEFAULT_IMAGE_CLOSENESS_KWARGS,
        ),
        KernelInfo(
            F.elastic_video,
            sample_inputs_fn=sample_inputs_elastic_video,
        ),
    ]
)


_CENTER_CROP_SPATIAL_SIZES = [(16, 16), (7, 33), (31, 9)]
_CENTER_CROP_OUTPUT_SIZES = [[4, 3], [42, 70], [4], 3, (5, 2), (6,)]


def sample_inputs_center_crop_image_tensor():
    for image_loader, output_size in itertools.product(
        make_image_loaders(sizes=[(16, 17)], color_spaces=[features.ColorSpace.RGB], dtypes=[torch.float32]),
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
        make_image_loaders(sizes=_CENTER_CROP_SPATIAL_SIZES, extra_dims=[()]), _CENTER_CROP_OUTPUT_SIZES
    ):
        yield ArgsKwargs(image_loader, output_size=output_size)


def sample_inputs_center_crop_bounding_box():
    for bounding_box_loader, output_size in itertools.product(make_bounding_box_loaders(), _CENTER_CROP_OUTPUT_SIZES):
        yield ArgsKwargs(
            bounding_box_loader,
            format=bounding_box_loader.format,
            spatial_size=bounding_box_loader.spatial_size,
            output_size=output_size,
        )


def sample_inputs_center_crop_mask():
    for mask_loader in make_mask_loaders(sizes=["random"], num_categories=["random"], num_objects=["random"]):
        height, width = mask_loader.shape[-2:]
        yield ArgsKwargs(mask_loader, output_size=(height // 2, width // 2))


def reference_inputs_center_crop_mask():
    for mask_loader, output_size in itertools.product(
        make_mask_loaders(sizes=_CENTER_CROP_SPATIAL_SIZES, extra_dims=[()], num_objects=[1]), _CENTER_CROP_OUTPUT_SIZES
    ):
        yield ArgsKwargs(mask_loader, output_size=output_size)


def sample_inputs_center_crop_video():
    for video_loader in make_video_loaders(sizes=["random"], num_frames=["random"]):
        height, width = video_loader.shape[-2:]
        yield ArgsKwargs(video_loader, output_size=(height // 2, width // 2))


KERNEL_INFOS.extend(
    [
        KernelInfo(
            F.center_crop_image_tensor,
            sample_inputs_fn=sample_inputs_center_crop_image_tensor,
            reference_fn=pil_reference_wrapper(F.center_crop_image_pil),
            reference_inputs_fn=reference_inputs_center_crop_image_tensor,
            closeness_kwargs=DEFAULT_IMAGE_CLOSENESS_KWARGS,
            test_marks=[
                xfail_jit_integer_size("output_size"),
            ],
        ),
        KernelInfo(
            F.center_crop_bounding_box,
            sample_inputs_fn=sample_inputs_center_crop_bounding_box,
            test_marks=[
                xfail_jit_integer_size("output_size"),
            ],
        ),
        KernelInfo(
            F.center_crop_mask,
            sample_inputs_fn=sample_inputs_center_crop_mask,
            reference_fn=pil_reference_wrapper(F.center_crop_image_pil),
            reference_inputs_fn=reference_inputs_center_crop_mask,
            closeness_kwargs=DEFAULT_IMAGE_CLOSENESS_KWARGS,
            test_marks=[
                xfail_jit_integer_size("output_size"),
            ],
        ),
        KernelInfo(
            F.center_crop_video,
            sample_inputs_fn=sample_inputs_center_crop_video,
        ),
    ]
)


def sample_inputs_gaussian_blur_image_tensor():
    make_gaussian_blur_image_loaders = functools.partial(
        make_image_loaders, sizes=[(7, 33)], color_spaces=[features.ColorSpace.RGB]
    )

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
            F.gaussian_blur_image_tensor,
            sample_inputs_fn=sample_inputs_gaussian_blur_image_tensor,
            closeness_kwargs=DEFAULT_IMAGE_CLOSENESS_KWARGS,
            test_marks=[
                xfail_jit_python_scalar_arg("kernel_size"),
                xfail_jit_python_scalar_arg("sigma"),
            ],
        ),
        KernelInfo(
            F.gaussian_blur_video,
            sample_inputs_fn=sample_inputs_gaussian_blur_video,
        ),
    ]
)


def sample_inputs_equalize_image_tensor():
    for image_loader in make_image_loaders(
        sizes=["random"], color_spaces=(features.ColorSpace.GRAY, features.ColorSpace.RGB), dtypes=[torch.uint8]
    ):
        yield ArgsKwargs(image_loader)


def reference_inputs_equalize_image_tensor():
    for image_loader in make_image_loaders(
        extra_dims=[()], color_spaces=(features.ColorSpace.GRAY, features.ColorSpace.RGB), dtypes=[torch.uint8]
    ):
        yield ArgsKwargs(image_loader)


def sample_inputs_equalize_video():
    for video_loader in make_video_loaders(sizes=["random"], num_frames=["random"]):
        yield ArgsKwargs(video_loader)


KERNEL_INFOS.extend(
    [
        KernelInfo(
            F.equalize_image_tensor,
            kernel_name="equalize_image_tensor",
            sample_inputs_fn=sample_inputs_equalize_image_tensor,
            reference_fn=pil_reference_wrapper(F.equalize_image_pil),
            reference_inputs_fn=reference_inputs_equalize_image_tensor,
            closeness_kwargs=DEFAULT_IMAGE_CLOSENESS_KWARGS,
        ),
        KernelInfo(
            F.equalize_video,
            sample_inputs_fn=sample_inputs_equalize_video,
        ),
    ]
)


def sample_inputs_invert_image_tensor():
    for image_loader in make_image_loaders(
        sizes=["random"], color_spaces=(features.ColorSpace.GRAY, features.ColorSpace.RGB)
    ):
        yield ArgsKwargs(image_loader)


def reference_inputs_invert_image_tensor():
    for image_loader in make_image_loaders(
        color_spaces=(features.ColorSpace.GRAY, features.ColorSpace.RGB), extra_dims=[()]
    ):
        yield ArgsKwargs(image_loader)


def sample_inputs_invert_video():
    for video_loader in make_video_loaders(sizes=["random"], num_frames=["random"]):
        yield ArgsKwargs(video_loader)


KERNEL_INFOS.extend(
    [
        KernelInfo(
            F.invert_image_tensor,
            kernel_name="invert_image_tensor",
            sample_inputs_fn=sample_inputs_invert_image_tensor,
            reference_fn=pil_reference_wrapper(F.invert_image_pil),
            reference_inputs_fn=reference_inputs_invert_image_tensor,
            closeness_kwargs=DEFAULT_IMAGE_CLOSENESS_KWARGS,
        ),
        KernelInfo(
            F.invert_video,
            sample_inputs_fn=sample_inputs_invert_video,
        ),
    ]
)


_POSTERIZE_BITS = [1, 4, 8]


def sample_inputs_posterize_image_tensor():
    for image_loader in make_image_loaders(
        sizes=["random"], color_spaces=(features.ColorSpace.GRAY, features.ColorSpace.RGB), dtypes=[torch.uint8]
    ):
        yield ArgsKwargs(image_loader, bits=_POSTERIZE_BITS[0])


def reference_inputs_posterize_image_tensor():
    for image_loader, bits in itertools.product(
        make_image_loaders(
            color_spaces=(features.ColorSpace.GRAY, features.ColorSpace.RGB), extra_dims=[()], dtypes=[torch.uint8]
        ),
        _POSTERIZE_BITS,
    ):
        yield ArgsKwargs(image_loader, bits=bits)


def sample_inputs_posterize_video():
    for video_loader in make_video_loaders(sizes=["random"], num_frames=["random"]):
        yield ArgsKwargs(video_loader, bits=_POSTERIZE_BITS[0])


KERNEL_INFOS.extend(
    [
        KernelInfo(
            F.posterize_image_tensor,
            kernel_name="posterize_image_tensor",
            sample_inputs_fn=sample_inputs_posterize_image_tensor,
            reference_fn=pil_reference_wrapper(F.posterize_image_pil),
            reference_inputs_fn=reference_inputs_posterize_image_tensor,
            closeness_kwargs=DEFAULT_IMAGE_CLOSENESS_KWARGS,
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
    for image_loader in make_image_loaders(
        sizes=["random"], color_spaces=(features.ColorSpace.GRAY, features.ColorSpace.RGB)
    ):
        yield ArgsKwargs(image_loader, threshold=next(_get_solarize_thresholds(image_loader.dtype)))


def reference_inputs_solarize_image_tensor():
    for image_loader in make_image_loaders(
        color_spaces=(features.ColorSpace.GRAY, features.ColorSpace.RGB), extra_dims=[()]
    ):
        for threshold in _get_solarize_thresholds(image_loader.dtype):
            yield ArgsKwargs(image_loader, threshold=threshold)


def sample_inputs_solarize_video():
    for video_loader in make_video_loaders(sizes=["random"], num_frames=["random"]):
        yield ArgsKwargs(video_loader, threshold=next(_get_solarize_thresholds(video_loader.dtype)))


KERNEL_INFOS.extend(
    [
        KernelInfo(
            F.solarize_image_tensor,
            kernel_name="solarize_image_tensor",
            sample_inputs_fn=sample_inputs_solarize_image_tensor,
            reference_fn=pil_reference_wrapper(F.solarize_image_pil),
            reference_inputs_fn=reference_inputs_solarize_image_tensor,
            closeness_kwargs=DEFAULT_IMAGE_CLOSENESS_KWARGS,
        ),
        KernelInfo(
            F.solarize_video,
            sample_inputs_fn=sample_inputs_solarize_video,
        ),
    ]
)


def sample_inputs_autocontrast_image_tensor():
    for image_loader in make_image_loaders(
        sizes=["random"], color_spaces=(features.ColorSpace.GRAY, features.ColorSpace.RGB)
    ):
        yield ArgsKwargs(image_loader)


def reference_inputs_autocontrast_image_tensor():
    for image_loader in make_image_loaders(
        color_spaces=(features.ColorSpace.GRAY, features.ColorSpace.RGB), extra_dims=[()]
    ):
        yield ArgsKwargs(image_loader)


def sample_inputs_autocontrast_video():
    for video_loader in make_video_loaders(sizes=["random"], num_frames=["random"]):
        yield ArgsKwargs(video_loader)


KERNEL_INFOS.extend(
    [
        KernelInfo(
            F.autocontrast_image_tensor,
            kernel_name="autocontrast_image_tensor",
            sample_inputs_fn=sample_inputs_autocontrast_image_tensor,
            reference_fn=pil_reference_wrapper(F.autocontrast_image_pil),
            reference_inputs_fn=reference_inputs_autocontrast_image_tensor,
            closeness_kwargs=DEFAULT_IMAGE_CLOSENESS_KWARGS,
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
        sizes=["random", (2, 2)],
        color_spaces=(features.ColorSpace.GRAY, features.ColorSpace.RGB),
    ):
        yield ArgsKwargs(image_loader, sharpness_factor=_ADJUST_SHARPNESS_FACTORS[0])


def reference_inputs_adjust_sharpness_image_tensor():
    for image_loader, sharpness_factor in itertools.product(
        make_image_loaders(color_spaces=(features.ColorSpace.GRAY, features.ColorSpace.RGB), extra_dims=[()]),
        _ADJUST_SHARPNESS_FACTORS,
    ):
        yield ArgsKwargs(image_loader, sharpness_factor=sharpness_factor)


def sample_inputs_adjust_sharpness_video():
    for video_loader in make_video_loaders(sizes=["random"], num_frames=["random"]):
        yield ArgsKwargs(video_loader, sharpness_factor=_ADJUST_SHARPNESS_FACTORS[0])


KERNEL_INFOS.extend(
    [
        KernelInfo(
            F.adjust_sharpness_image_tensor,
            kernel_name="adjust_sharpness_image_tensor",
            sample_inputs_fn=sample_inputs_adjust_sharpness_image_tensor,
            reference_fn=pil_reference_wrapper(F.adjust_sharpness_image_pil),
            reference_inputs_fn=reference_inputs_adjust_sharpness_image_tensor,
            closeness_kwargs=DEFAULT_IMAGE_CLOSENESS_KWARGS,
        ),
        KernelInfo(
            F.adjust_sharpness_video,
            sample_inputs_fn=sample_inputs_adjust_sharpness_video,
        ),
    ]
)


def sample_inputs_erase_image_tensor():
    for image_loader in make_image_loaders(sizes=["random"]):
        # FIXME: make the parameters more diverse
        h, w = 6, 7
        v = torch.rand(image_loader.num_channels, h, w)
        yield ArgsKwargs(image_loader, i=1, j=2, h=h, w=w, v=v)


def sample_inputs_erase_video():
    for video_loader in make_video_loaders(sizes=["random"], num_frames=["random"]):
        # FIXME: make the parameters more diverse
        h, w = 6, 7
        v = torch.rand(video_loader.num_channels, h, w)
        yield ArgsKwargs(video_loader, i=1, j=2, h=h, w=w, v=v)


KERNEL_INFOS.extend(
    [
        KernelInfo(
            F.erase_image_tensor,
            kernel_name="erase_image_tensor",
            sample_inputs_fn=sample_inputs_erase_image_tensor,
        ),
        KernelInfo(
            F.erase_video,
            sample_inputs_fn=sample_inputs_erase_video,
        ),
    ]
)

_ADJUST_BRIGHTNESS_FACTORS = [0.1, 0.5]


def sample_inputs_adjust_brightness_image_tensor():
    for image_loader in make_image_loaders(
        sizes=["random"], color_spaces=(features.ColorSpace.GRAY, features.ColorSpace.RGB)
    ):
        yield ArgsKwargs(image_loader, brightness_factor=_ADJUST_BRIGHTNESS_FACTORS[0])


def reference_inputs_adjust_brightness_image_tensor():
    for image_loader, brightness_factor in itertools.product(
        make_image_loaders(color_spaces=(features.ColorSpace.GRAY, features.ColorSpace.RGB), extra_dims=[()]),
        _ADJUST_BRIGHTNESS_FACTORS,
    ):
        yield ArgsKwargs(image_loader, brightness_factor=brightness_factor)


def sample_inputs_adjust_brightness_video():
    for video_loader in make_video_loaders(sizes=["random"], num_frames=["random"]):
        yield ArgsKwargs(video_loader, brightness_factor=_ADJUST_BRIGHTNESS_FACTORS[0])


KERNEL_INFOS.extend(
    [
        KernelInfo(
            F.adjust_brightness_image_tensor,
            kernel_name="adjust_brightness_image_tensor",
            sample_inputs_fn=sample_inputs_adjust_brightness_image_tensor,
            reference_fn=pil_reference_wrapper(F.adjust_brightness_image_pil),
            reference_inputs_fn=reference_inputs_adjust_brightness_image_tensor,
            closeness_kwargs=DEFAULT_IMAGE_CLOSENESS_KWARGS,
        ),
        KernelInfo(
            F.adjust_brightness_video,
            sample_inputs_fn=sample_inputs_adjust_brightness_video,
        ),
    ]
)


_ADJUST_CONTRAST_FACTORS = [0.1, 0.5]


def sample_inputs_adjust_contrast_image_tensor():
    for image_loader in make_image_loaders(
        sizes=["random"], color_spaces=(features.ColorSpace.GRAY, features.ColorSpace.RGB)
    ):
        yield ArgsKwargs(image_loader, contrast_factor=_ADJUST_CONTRAST_FACTORS[0])


def reference_inputs_adjust_contrast_image_tensor():
    for image_loader, contrast_factor in itertools.product(
        make_image_loaders(color_spaces=(features.ColorSpace.GRAY, features.ColorSpace.RGB), extra_dims=[()]),
        _ADJUST_CONTRAST_FACTORS,
    ):
        yield ArgsKwargs(image_loader, contrast_factor=contrast_factor)


def sample_inputs_adjust_contrast_video():
    for video_loader in make_video_loaders(sizes=["random"], num_frames=["random"]):
        yield ArgsKwargs(video_loader, contrast_factor=_ADJUST_CONTRAST_FACTORS[0])


KERNEL_INFOS.extend(
    [
        KernelInfo(
            F.adjust_contrast_image_tensor,
            kernel_name="adjust_contrast_image_tensor",
            sample_inputs_fn=sample_inputs_adjust_contrast_image_tensor,
            reference_fn=pil_reference_wrapper(F.adjust_contrast_image_pil),
            reference_inputs_fn=reference_inputs_adjust_contrast_image_tensor,
            closeness_kwargs=DEFAULT_IMAGE_CLOSENESS_KWARGS,
        ),
        KernelInfo(
            F.adjust_contrast_video,
            sample_inputs_fn=sample_inputs_adjust_contrast_video,
        ),
    ]
)

_ADJUST_GAMMA_GAMMAS_GAINS = [
    (0.5, 2.0),
    (0.0, 1.0),
]


def sample_inputs_adjust_gamma_image_tensor():
    gamma, gain = _ADJUST_GAMMA_GAMMAS_GAINS[0]
    for image_loader in make_image_loaders(
        sizes=["random"], color_spaces=(features.ColorSpace.GRAY, features.ColorSpace.RGB)
    ):
        yield ArgsKwargs(image_loader, gamma=gamma, gain=gain)


def reference_inputs_adjust_gamma_image_tensor():
    for image_loader, (gamma, gain) in itertools.product(
        make_image_loaders(color_spaces=(features.ColorSpace.GRAY, features.ColorSpace.RGB), extra_dims=[()]),
        _ADJUST_GAMMA_GAMMAS_GAINS,
    ):
        yield ArgsKwargs(image_loader, gamma=gamma, gain=gain)


def sample_inputs_adjust_gamma_video():
    gamma, gain = _ADJUST_GAMMA_GAMMAS_GAINS[0]
    for video_loader in make_video_loaders(sizes=["random"], num_frames=["random"]):
        yield ArgsKwargs(video_loader, gamma=gamma, gain=gain)


KERNEL_INFOS.extend(
    [
        KernelInfo(
            F.adjust_gamma_image_tensor,
            kernel_name="adjust_gamma_image_tensor",
            sample_inputs_fn=sample_inputs_adjust_gamma_image_tensor,
            reference_fn=pil_reference_wrapper(F.adjust_gamma_image_pil),
            reference_inputs_fn=reference_inputs_adjust_gamma_image_tensor,
            closeness_kwargs=DEFAULT_IMAGE_CLOSENESS_KWARGS,
        ),
        KernelInfo(
            F.adjust_gamma_video,
            sample_inputs_fn=sample_inputs_adjust_gamma_video,
        ),
    ]
)


_ADJUST_HUE_FACTORS = [-0.1, 0.5]


def sample_inputs_adjust_hue_image_tensor():
    for image_loader in make_image_loaders(
        sizes=["random"], color_spaces=(features.ColorSpace.GRAY, features.ColorSpace.RGB)
    ):
        yield ArgsKwargs(image_loader, hue_factor=_ADJUST_HUE_FACTORS[0])


def reference_inputs_adjust_hue_image_tensor():
    for image_loader, hue_factor in itertools.product(
        make_image_loaders(color_spaces=(features.ColorSpace.GRAY, features.ColorSpace.RGB), extra_dims=[()]),
        _ADJUST_HUE_FACTORS,
    ):
        yield ArgsKwargs(image_loader, hue_factor=hue_factor)


def sample_inputs_adjust_hue_video():
    for video_loader in make_video_loaders(sizes=["random"], num_frames=["random"]):
        yield ArgsKwargs(video_loader, hue_factor=_ADJUST_HUE_FACTORS[0])


KERNEL_INFOS.extend(
    [
        KernelInfo(
            F.adjust_hue_image_tensor,
            kernel_name="adjust_hue_image_tensor",
            sample_inputs_fn=sample_inputs_adjust_hue_image_tensor,
            reference_fn=pil_reference_wrapper(F.adjust_hue_image_pil),
            reference_inputs_fn=reference_inputs_adjust_hue_image_tensor,
            closeness_kwargs=DEFAULT_IMAGE_CLOSENESS_KWARGS,
        ),
        KernelInfo(
            F.adjust_hue_video,
            sample_inputs_fn=sample_inputs_adjust_hue_video,
        ),
    ]
)

_ADJUST_SATURATION_FACTORS = [0.1, 0.5]


def sample_inputs_adjust_saturation_image_tensor():
    for image_loader in make_image_loaders(
        sizes=["random"], color_spaces=(features.ColorSpace.GRAY, features.ColorSpace.RGB)
    ):
        yield ArgsKwargs(image_loader, saturation_factor=_ADJUST_SATURATION_FACTORS[0])


def reference_inputs_adjust_saturation_image_tensor():
    for image_loader, saturation_factor in itertools.product(
        make_image_loaders(color_spaces=(features.ColorSpace.GRAY, features.ColorSpace.RGB), extra_dims=[()]),
        _ADJUST_SATURATION_FACTORS,
    ):
        yield ArgsKwargs(image_loader, saturation_factor=saturation_factor)


def sample_inputs_adjust_saturation_video():
    for video_loader in make_video_loaders(sizes=["random"], num_frames=["random"]):
        yield ArgsKwargs(video_loader, saturation_factor=_ADJUST_SATURATION_FACTORS[0])


KERNEL_INFOS.extend(
    [
        KernelInfo(
            F.adjust_saturation_image_tensor,
            kernel_name="adjust_saturation_image_tensor",
            sample_inputs_fn=sample_inputs_adjust_saturation_image_tensor,
            reference_fn=pil_reference_wrapper(F.adjust_saturation_image_pil),
            reference_inputs_fn=reference_inputs_adjust_saturation_image_tensor,
            closeness_kwargs=DEFAULT_IMAGE_CLOSENESS_KWARGS,
        ),
        KernelInfo(
            F.adjust_saturation_video,
            sample_inputs_fn=sample_inputs_adjust_saturation_video,
        ),
    ]
)


def sample_inputs_clamp_bounding_box():
    for bounding_box_loader in make_bounding_box_loaders():
        yield ArgsKwargs(
            bounding_box_loader, format=bounding_box_loader.format, spatial_size=bounding_box_loader.spatial_size
        )


KERNEL_INFOS.append(
    KernelInfo(
        F.clamp_bounding_box,
        sample_inputs_fn=sample_inputs_clamp_bounding_box,
    )
)

_FIVE_TEN_CROP_SIZES = [7, (6,), [5], (6, 5), [7, 6]]


def _get_five_ten_crop_spatial_size(size):
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
            sizes=[_get_five_ten_crop_spatial_size(size)],
            color_spaces=[features.ColorSpace.RGB],
            dtypes=[torch.float32],
        ):
            yield ArgsKwargs(image_loader, size=size)


def reference_inputs_five_crop_image_tensor():
    for size in _FIVE_TEN_CROP_SIZES:
        for image_loader in make_image_loaders(sizes=[_get_five_ten_crop_spatial_size(size)], extra_dims=[()]):
            yield ArgsKwargs(image_loader, size=size)


def sample_inputs_ten_crop_image_tensor():
    for size, vertical_flip in itertools.product(_FIVE_TEN_CROP_SIZES, [False, True]):
        for image_loader in make_image_loaders(
            sizes=[_get_five_ten_crop_spatial_size(size)],
            color_spaces=[features.ColorSpace.RGB],
            dtypes=[torch.float32],
        ):
            yield ArgsKwargs(image_loader, size=size, vertical_flip=vertical_flip)


def reference_inputs_ten_crop_image_tensor():
    for size, vertical_flip in itertools.product(_FIVE_TEN_CROP_SIZES, [False, True]):
        for image_loader in make_image_loaders(sizes=[_get_five_ten_crop_spatial_size(size)], extra_dims=[()]):
            yield ArgsKwargs(image_loader, size=size, vertical_flip=vertical_flip)


KERNEL_INFOS.extend(
    [
        KernelInfo(
            F.five_crop_image_tensor,
            sample_inputs_fn=sample_inputs_five_crop_image_tensor,
            reference_fn=pil_reference_wrapper(F.five_crop_image_pil),
            reference_inputs_fn=reference_inputs_five_crop_image_tensor,
            test_marks=[
                xfail_jit_integer_size(),
                mark_framework_limitation(("TestKernels", "test_batched_vs_single"), "Custom batching needed."),
            ],
            closeness_kwargs=DEFAULT_IMAGE_CLOSENESS_KWARGS,
        ),
        KernelInfo(
            F.ten_crop_image_tensor,
            sample_inputs_fn=sample_inputs_ten_crop_image_tensor,
            reference_fn=pil_reference_wrapper(F.ten_crop_image_pil),
            reference_inputs_fn=reference_inputs_ten_crop_image_tensor,
            test_marks=[
                xfail_jit_integer_size(),
                mark_framework_limitation(("TestKernels", "test_batched_vs_single"), "Custom batching needed."),
            ],
            closeness_kwargs=DEFAULT_IMAGE_CLOSENESS_KWARGS,
        ),
    ]
)

_NORMALIZE_MEANS_STDS = [
    ((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    ([0.0, 0.0, 0.0], [1.0, 1.0, 1.0]),
]


def sample_inputs_normalize_image_tensor():
    for image_loader, (mean, std) in itertools.product(
        make_image_loaders(sizes=["random"], color_spaces=[features.ColorSpace.RGB], dtypes=[torch.float32]),
        _NORMALIZE_MEANS_STDS,
    ):
        yield ArgsKwargs(image_loader, mean=mean, std=std)


def sample_inputs_normalize_video():
    mean, std = _NORMALIZE_MEANS_STDS[0]
    for video_loader in make_video_loaders(
        sizes=["random"], color_spaces=[features.ColorSpace.RGB], num_frames=["random"], dtypes=[torch.float32]
    ):
        yield ArgsKwargs(video_loader, mean=mean, std=std)


KERNEL_INFOS.extend(
    [
        KernelInfo(
            F.normalize_image_tensor,
            kernel_name="normalize_image_tensor",
            sample_inputs_fn=sample_inputs_normalize_image_tensor,
        ),
        KernelInfo(
            F.normalize_video,
            sample_inputs_fn=sample_inputs_normalize_video,
        ),
    ]
)


def sample_inputs_convert_image_dtype():
    for input_dtype, output_dtype in itertools.product(
        [torch.uint8, torch.int64, torch.float32, torch.float64], repeat=2
    ):
        if input_dtype.is_floating_point and output_dtype == torch.int64:
            # conversion cannot be performed safely
            continue

        for image_loader in make_image_loaders(
            sizes=["random"], color_spaces=[features.ColorSpace.RGB], dtypes=[input_dtype]
        ):
            yield ArgsKwargs(image_loader, dtype=output_dtype)

    yield ArgsKwargs(make_image_loader(color_space=features.ColorSpace.RGB), dtype=torch.uint8)


def reference_convert_image_dtype(image, dtype=torch.float):
    input_dtype = image.dtype
    output_dtype = dtype

    if output_dtype == input_dtype:
        return image

    def fn(value):
        if input_dtype.is_floating_point:
            if output_dtype.is_floating_point:
                return value
            else:
                return int(decimal.Decimal(value) * torch.iinfo(output_dtype).max)
        else:
            input_max_value = torch.iinfo(input_dtype).max

            if output_dtype.is_floating_point:
                return float(decimal.Decimal(value) / input_max_value)
            else:
                output_max_value = torch.iinfo(output_dtype).max

                if input_max_value > output_max_value:
                    factor = (input_max_value + 1) // (output_max_value + 1)
                    return value // factor
                else:
                    factor = (output_max_value + 1) // (input_max_value + 1)
                    return value * factor

    return torch.tensor(tree_map(fn, image.tolist()), dtype=dtype)


def reference_inputs_convert_image_dtype():
    for input_dtype, output_dtype in itertools.product(
        [
            torch.uint8,
            torch.int16,
            torch.int32,
            torch.int64,
            torch.float16,
            torch.float32,
            torch.float64,
            torch.bfloat16,
        ],
        repeat=2,
    ):
        if (input_dtype == torch.float32 and output_dtype in {torch.int32, torch.int64}) or (
            input_dtype == torch.float64 and output_dtype == torch.int64
        ):
            continue

        if input_dtype.is_floating_point:
            data = [0.0, 0.5, 1.0]
        else:
            max_value = torch.iinfo(input_dtype).max
            data = [0, max_value // 2, max_value]
        image = torch.tensor(data, dtype=input_dtype)

        yield ArgsKwargs(image, dtype=output_dtype)


KERNEL_INFOS.extend(
    [
        KernelInfo(
            F.convert_image_dtype,
            sample_inputs_fn=sample_inputs_convert_image_dtype,
            reference_fn=reference_convert_image_dtype,
            reference_inputs_fn=reference_inputs_convert_image_dtype,
            test_marks=[
                TestMark(
                    ("TestKernels", "test_scripted_vs_eager"),
                    pytest.mark.filterwarnings(f"ignore:{re.escape('operator() profile_node %41')}:UserWarning"),
                ),
                TestMark(
                    ("TestKernels", "test_dtype_and_device_consistency"),
                    pytest.mark.skip(reason="`convert_dtype_*` kernels convert the dtype by design"),
                    condition=lambda args_kwargs: args_kwargs.args[0].dtype
                    != args_kwargs.kwargs.get("dtype", torch.float32),
                ),
                TestMark(
                    ("TestKernels", "test_against_reference"),
                    pytest.mark.xfail(reason="Conversion overflows"),
                    condition=lambda args_kwargs: (
                        args_kwargs.args[0].dtype in {torch.float16, torch.bfloat16}
                        and not args_kwargs.kwargs["dtype"].is_floating_point
                    )
                    or (
                        args_kwargs.args[0].dtype in {torch.float16, torch.bfloat16}
                        and args_kwargs.kwargs["dtype"] == torch.int64
                    )
                    or (
                        args_kwargs.args[0].dtype in {torch.int32, torch.int64}
                        and args_kwargs.kwargs["dtype"] == torch.float16
                    ),
                ),
            ],
        ),
    ]
)
