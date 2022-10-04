import dataclasses
import functools
import itertools
import math
from collections import defaultdict
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence

import numpy as np
import pytest
import torch.testing
import torchvision.ops
import torchvision.prototype.transforms.functional as F
from datasets_utils import combinations_grid
from prototype_common_utils import ArgsKwargs, make_bounding_box_loaders, make_image_loaders, make_mask_loaders
from torchvision.prototype import features
from torchvision.transforms.functional_tensor import _max_value as get_max_value

__all__ = ["KernelInfo", "KERNEL_INFOS"]


@dataclasses.dataclass
class Skip:
    test_name: str
    reason: str
    condition: Callable[[ArgsKwargs, str], bool] = lambda args_kwargs, device: True


@dataclasses.dataclass
class KernelInfo:
    kernel: Callable
    # Most common tests use these inputs to check the kernel. As such it should cover all valid code paths, but should
    # not include extensive parameter combinations to keep to overall test count moderate.
    sample_inputs_fn: Callable[[], Iterable[ArgsKwargs]]
    # Defaults to `kernel.__name__`. Should be set if the function is exposed under a different name
    # TODO: This can probably be removed after roll-out since we shouldn't have any aliasing then
    kernel_name: Optional[str] = None
    # This function should mirror the kernel. It should have the same signature as the `kernel` and as such also take
    # tensors as inputs. Any conversion into another object type, e.g. PIL images or numpy arrays, should happen
    # inside the function. It should return a tensor or to be more precise an object that can be compared to a
    # tensor by `assert_close`. If omitted, no reference test will be performed.
    reference_fn: Optional[Callable] = None
    # These inputs are only used for the reference tests and thus can be comprehensive with regard to the parameter
    # values to be tested. If not specified, `sample_inputs_fn` will be used.
    reference_inputs_fn: Optional[Callable[[], Iterable[ArgsKwargs]]] = None
    # Additional parameters, e.g. `rtol=1e-3`, passed to `assert_close`.
    closeness_kwargs: Dict[str, Any] = dataclasses.field(default_factory=dict)
    skips: Sequence[Skip] = dataclasses.field(default_factory=list)
    _skips_map: Dict[str, List[Skip]] = dataclasses.field(default=None, init=False)

    def __post_init__(self):
        self.kernel_name = self.kernel_name or self.kernel.__name__
        self.reference_inputs_fn = self.reference_inputs_fn or self.sample_inputs_fn

        skips_map = defaultdict(list)
        for skip in self.skips:
            skips_map[skip.test_name].append(skip)
        self._skips_map = dict(skips_map)

    def maybe_skip(self, *, test_name, args_kwargs, device):
        skips = self._skips_map.get(test_name)
        if not skips:
            return

        for skip in skips:
            if skip.condition(args_kwargs, device):
                pytest.skip(skip.reason)


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


def skip_python_scalar_arg_jit(name, *, reason="Python scalar int or float is not supported when scripting"):
    return Skip(
        "test_scripted_vs_eager",
        condition=lambda args_kwargs, device: isinstance(args_kwargs.kwargs[name], (int, float)),
        reason=reason,
    )


def skip_integer_size_jit(name="size"):
    return skip_python_scalar_arg_jit(name, reason="Integer size is not supported when scripting.")


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
            bounding_box_loader, format=bounding_box_loader.format, image_size=bounding_box_loader.image_size
        )


def sample_inputs_horizontal_flip_mask():
    for image_loader in make_mask_loaders(sizes=["random"], dtypes=[torch.uint8]):
        yield ArgsKwargs(image_loader)


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
        ),
        KernelInfo(
            F.horizontal_flip_mask,
            sample_inputs_fn=sample_inputs_horizontal_flip_mask,
        ),
    ]
)


def _get_resize_sizes(image_size):
    height, width = image_size
    length = max(image_size)
    # FIXME: enable me when the kernels are fixed
    # yield length
    yield [length]
    yield (length,)
    new_height = int(height * 0.75)
    new_width = int(width * 1.25)
    yield [new_height, new_width]
    yield height, width


def sample_inputs_resize_image_tensor():
    for image_loader, interpolation in itertools.product(
        make_image_loaders(dtypes=[torch.float32]),
        [
            F.InterpolationMode.NEAREST,
            F.InterpolationMode.BICUBIC,
        ],
    ):
        for size in _get_resize_sizes(image_loader.image_size):
            yield ArgsKwargs(image_loader, size=size, interpolation=interpolation)


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
            F.InterpolationMode.BILINEAR,
            F.InterpolationMode.BICUBIC,
        ],
    ):
        for size in _get_resize_sizes(image_loader.image_size):
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
    for bounding_box_loader in make_bounding_box_loaders(formats=[features.BoundingBoxFormat.XYXY]):
        for size in _get_resize_sizes(bounding_box_loader.image_size):
            yield ArgsKwargs(bounding_box_loader, size=size, image_size=bounding_box_loader.image_size)


def sample_inputs_resize_mask():
    for mask_loader in make_mask_loaders(dtypes=[torch.uint8]):
        for size in _get_resize_sizes(mask_loader.shape[-2:]):
            yield ArgsKwargs(mask_loader, size=size)


@pil_reference_wrapper
def reference_resize_mask(*args, **kwargs):
    return F.resize_image_pil(*args, interpolation=F.InterpolationMode.NEAREST, **kwargs)


def reference_inputs_resize_mask():
    for mask_loader in make_mask_loaders(extra_dims=[()], num_objects=[1]):
        for size in _get_resize_sizes(mask_loader.shape[-2:]):
            yield ArgsKwargs(mask_loader, size=size)


KERNEL_INFOS.extend(
    [
        KernelInfo(
            F.resize_image_tensor,
            sample_inputs_fn=sample_inputs_resize_image_tensor,
            reference_fn=reference_resize_image_tensor,
            reference_inputs_fn=reference_inputs_resize_image_tensor,
            closeness_kwargs=DEFAULT_IMAGE_CLOSENESS_KWARGS,
            skips=[
                skip_integer_size_jit(),
            ],
        ),
        KernelInfo(
            F.resize_bounding_box,
            sample_inputs_fn=sample_inputs_resize_bounding_box,
            skips=[
                skip_integer_size_jit(),
            ],
        ),
        KernelInfo(
            F.resize_mask,
            sample_inputs_fn=sample_inputs_resize_mask,
            reference_fn=reference_resize_mask,
            reference_inputs_fn=reference_inputs_resize_mask,
            closeness_kwargs=DEFAULT_IMAGE_CLOSENESS_KWARGS,
            skips=[
                skip_integer_size_jit(),
            ],
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


def sample_inputs_affine_image_tensor():
    for image_loader, interpolation_mode, center in itertools.product(
        make_image_loaders(sizes=["random"], dtypes=[torch.float32]),
        [
            F.InterpolationMode.NEAREST,
            F.InterpolationMode.BILINEAR,
        ],
        [None, (0, 0)],
    ):
        for fill in [None, 128.0, 128, [12.0], [0.5] * image_loader.num_channels]:
            yield ArgsKwargs(
                image_loader,
                interpolation=interpolation_mode,
                center=center,
                fill=fill,
                **_AFFINE_KWARGS[0],
            )

    for image_loader, affine_kwargs in itertools.product(
        make_image_loaders(sizes=["random"], dtypes=[torch.float32]), _diversify_affine_kwargs_types(_AFFINE_KWARGS[0])
    ):
        yield ArgsKwargs(image_loader, **affine_kwargs)


def reference_inputs_affine_image_tensor():
    for image_loader, affine_kwargs in itertools.product(make_image_loaders(extra_dims=[()]), _AFFINE_KWARGS):
        yield ArgsKwargs(
            image_loader,
            interpolation=F.InterpolationMode.NEAREST,
            **affine_kwargs,
        )


def sample_inputs_affine_bounding_box():
    for bounding_box_loader in make_bounding_box_loaders():
        yield ArgsKwargs(
            bounding_box_loader,
            format=bounding_box_loader.format,
            image_size=bounding_box_loader.image_size,
            **_AFFINE_KWARGS[0],
        )

    for bounding_box_loader, affine_kwargs in itertools.product(
        make_bounding_box_loaders(), _diversify_affine_kwargs_types(_AFFINE_KWARGS[0])
    ):
        yield ArgsKwargs(
            bounding_box_loader,
            format=bounding_box_loader.format,
            image_size=bounding_box_loader.image_size,
            **affine_kwargs,
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


def reference_affine_bounding_box(bounding_box, *, format, image_size, angle, translate, scale, shear, center=None):
    if center is None:
        center = [s * 0.5 for s in image_size[::-1]]

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
                np.min(transformed_points[:, 0]),
                np.min(transformed_points[:, 1]),
                np.max(transformed_points[:, 0]),
                np.max(transformed_points[:, 1]),
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
            image_size=bounding_box_loader.image_size,
            **affine_kwargs,
        )


def sample_inputs_affine_image_mask():
    for mask_loader, center in itertools.product(
        make_mask_loaders(sizes=["random"], dtypes=[torch.uint8]),
        [None, (0, 0)],
    ):
        yield ArgsKwargs(mask_loader, center=center, **_AFFINE_KWARGS[0])

    for mask_loader, affine_kwargs in itertools.product(
        make_mask_loaders(sizes=["random"], dtypes=[torch.uint8]), _diversify_affine_kwargs_types(_AFFINE_KWARGS[0])
    ):
        yield ArgsKwargs(mask_loader, **affine_kwargs)


@pil_reference_wrapper
def reference_affine_mask(*args, **kwargs):
    return F.affine_image_pil(*args, interpolation=F.InterpolationMode.NEAREST, **kwargs)


def reference_inputs_resize_mask():
    for mask_loader, affine_kwargs in itertools.product(
        make_mask_loaders(extra_dims=[()], num_objects=[1]), _AFFINE_KWARGS
    ):
        yield ArgsKwargs(mask_loader, **affine_kwargs)


# FIXME: @datumbox, remove this as soon as you have fixed the behavior in https://github.com/pytorch/vision/pull/6636
def skip_scalar_shears(*test_names):
    for test_name in test_names:
        yield Skip(
            test_name,
            condition=lambda args_kwargs, device: isinstance(args_kwargs.kwargs["shear"], (int, float)),
            reason="The kernel is broken for a scalar `shear`",
        )


KERNEL_INFOS.extend(
    [
        KernelInfo(
            F.affine_image_tensor,
            sample_inputs_fn=sample_inputs_affine_image_tensor,
            reference_fn=pil_reference_wrapper(F.affine_image_pil),
            reference_inputs_fn=reference_inputs_affine_image_tensor,
            closeness_kwargs=DEFAULT_IMAGE_CLOSENESS_KWARGS,
            skips=[skip_python_scalar_arg_jit("shear", reason="Scalar shear is not supported by JIT")],
        ),
        KernelInfo(
            F.affine_bounding_box,
            sample_inputs_fn=sample_inputs_affine_bounding_box,
            reference_fn=reference_affine_bounding_box,
            reference_inputs_fn=reference_inputs_affine_bounding_box,
            closeness_kwargs=dict(atol=1, rtol=0),
            skips=[
                skip_python_scalar_arg_jit("shear", reason="Scalar shear is not supported by JIT"),
                *skip_scalar_shears(
                    "test_batched_vs_single",
                    "test_no_inplace",
                    "test_dtype_and_device_consistency",
                ),
            ],
        ),
        KernelInfo(
            F.affine_mask,
            sample_inputs_fn=sample_inputs_affine_image_mask,
            reference_fn=reference_affine_mask,
            reference_inputs_fn=reference_inputs_resize_mask,
            closeness_kwargs=DEFAULT_IMAGE_CLOSENESS_KWARGS,
            skips=[skip_python_scalar_arg_jit("shear", reason="Scalar shear is not supported by JIT")],
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
    color_spaces = set(features.ColorSpace) - {features.ColorSpace.OTHER}
    for image_loader in make_image_loaders(sizes=["random"], color_spaces=color_spaces, constant_alpha=True):
        old_color_space = image_loader.color_space
        for params in combinations_grid(new_color_space=color_spaces - {old_color_space}, copy=(True, False)):
            yield ArgsKwargs(image_loader, old_color_space=old_color_space, **params)


@pil_reference_wrapper
def reference_convert_color_space_image_tensor(image_pil, old_color_space, new_color_space, copy):
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


KERNEL_INFOS.append(
    KernelInfo(
        F.convert_color_space_image_tensor,
        sample_inputs_fn=sample_inputs_convert_color_space_image_tensor,
        reference_fn=reference_convert_color_space_image_tensor,
        reference_inputs_fn=reference_inputs_convert_color_space_image_tensor,
        closeness_kwargs=DEFAULT_IMAGE_CLOSENESS_KWARGS,
    ),
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
            bounding_box_loader, format=bounding_box_loader.format, image_size=bounding_box_loader.image_size
        )


def sample_inputs_vertical_flip_mask():
    for image_loader in make_mask_loaders(sizes=["random"], dtypes=[torch.uint8]):
        yield ArgsKwargs(image_loader)


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
    ]
)

_ROTATE_ANGLES = [-87, 15, 90]


def sample_inputs_rotate_image_tensor():
    for image_loader, params in itertools.product(
        make_image_loaders(sizes=["random"], dtypes=[torch.float32]),
        combinations_grid(
            interpolation=[F.InterpolationMode.NEAREST, F.InterpolationMode.BILINEAR],
            expand=[True, False],
            center=[None, (0, 0)],
        ),
    ):
        if params["center"] is not None and params["expand"]:
            # Otherwise this will emit a warning and ignore center anyway
            continue

        for fill in [None, 0.5, [0.5] * image_loader.num_channels]:
            yield ArgsKwargs(
                image_loader,
                angle=_ROTATE_ANGLES[0],
                fill=fill,
                **params,
            )


def reference_inputs_rotate_image_tensor():
    for image_loader, angle in itertools.product(make_image_loaders(extra_dims=[()]), _ROTATE_ANGLES):
        yield ArgsKwargs(image_loader, angle=angle)


def sample_inputs_rotate_bounding_box():
    for bounding_box_loader in make_bounding_box_loaders():
        yield ArgsKwargs(
            bounding_box_loader,
            format=bounding_box_loader.format,
            image_size=bounding_box_loader.image_size,
            angle=_ROTATE_ANGLES[0],
        )


def sample_inputs_rotate_mask():
    for image_loader, params in itertools.product(
        make_image_loaders(sizes=["random"], dtypes=[torch.uint8]),
        combinations_grid(
            expand=[True, False],
            center=[None, (0, 0)],
        ),
    ):
        if params["center"] is not None and params["expand"]:
            # Otherwise this will emit a warning and ignore center anyway
            continue

        yield ArgsKwargs(
            image_loader,
            angle=_ROTATE_ANGLES[0],
            **params,
        )


@pil_reference_wrapper
def reference_rotate_mask(*args, **kwargs):
    return F.rotate_image_pil(*args, interpolation=F.InterpolationMode.NEAREST, **kwargs)


def reference_inputs_rotate_mask():
    for mask_loader, angle in itertools.product(make_mask_loaders(extra_dims=[()], num_objects=[1]), _ROTATE_ANGLES):
        yield ArgsKwargs(mask_loader, angle=angle)


KERNEL_INFOS.extend(
    [
        KernelInfo(
            F.rotate_image_tensor,
            sample_inputs_fn=sample_inputs_rotate_image_tensor,
            reference_fn=pil_reference_wrapper(F.rotate_image_pil),
            reference_inputs_fn=reference_inputs_rotate_image_tensor,
            closeness_kwargs=DEFAULT_IMAGE_CLOSENESS_KWARGS,
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
    ]
)

_CROP_PARAMS = combinations_grid(top=[-8, 0, 9], left=[-8, 0, 9], height=[12, 20], width=[12, 20])


def sample_inputs_crop_image_tensor():
    for image_loader, params in itertools.product(make_image_loaders(), [_CROP_PARAMS[0], _CROP_PARAMS[-1]]):
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
    for mask_loader, params in itertools.product(make_mask_loaders(), [_CROP_PARAMS[0], _CROP_PARAMS[-1]]):
        yield ArgsKwargs(mask_loader, **params)


def reference_inputs_crop_mask():
    for mask_loader, params in itertools.product(make_mask_loaders(extra_dims=[()], num_objects=[1]), _CROP_PARAMS):
        yield ArgsKwargs(mask_loader, **params)


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
    ]
)

_PAD_PARAMS = combinations_grid(
    padding=[[1], [1, 1], [1, 1, 2, 2]],
    padding_mode=["constant", "symmetric", "edge", "reflect"],
)


def sample_inputs_pad_image_tensor():
    for image_loader, params in itertools.product(make_image_loaders(sizes=["random"]), _PAD_PARAMS):
        fills = [None, 128.0, 128, [12.0]]
        if params["padding_mode"] == "constant":
            fills.append([12.0 + c for c in range(image_loader.num_channels)])
        for fill in fills:
            yield ArgsKwargs(image_loader, fill=fill, **params)


def reference_inputs_pad_image_tensor():
    for image_loader, params in itertools.product(make_image_loaders(extra_dims=[()]), _PAD_PARAMS):
        # FIXME: PIL kernel doesn't support sequences of length 1 if the number of channels is larger. Shouldn't it?
        fills = [None, 128.0, 128]
        if params["padding_mode"] == "constant":
            fills.append([12.0 + c for c in range(image_loader.num_channels)])
        for fill in fills:
            yield ArgsKwargs(image_loader, fill=fill, **params)


def sample_inputs_pad_bounding_box():
    for bounding_box_loader, params in itertools.product(make_bounding_box_loaders(), _PAD_PARAMS):
        if params["padding_mode"] != "constant":
            continue

        yield ArgsKwargs(
            bounding_box_loader, format=bounding_box_loader.format, image_size=bounding_box_loader.image_size, **params
        )


def sample_inputs_pad_mask():
    for image_loader, fill, params in itertools.product(make_mask_loaders(sizes=["random"]), [None, 127], _PAD_PARAMS):
        yield ArgsKwargs(image_loader, fill=fill, **params)


def reference_inputs_pad_mask():
    for image_loader, fill, params in itertools.product(make_image_loaders(extra_dims=[()]), [None, 127], _PAD_PARAMS):
        yield ArgsKwargs(image_loader, fill=fill, **params)


KERNEL_INFOS.extend(
    [
        KernelInfo(
            F.pad_image_tensor,
            sample_inputs_fn=sample_inputs_pad_image_tensor,
            reference_fn=pil_reference_wrapper(F.pad_image_pil),
            reference_inputs_fn=reference_inputs_pad_image_tensor,
            closeness_kwargs=DEFAULT_IMAGE_CLOSENESS_KWARGS,
        ),
        KernelInfo(
            F.pad_bounding_box,
            sample_inputs_fn=sample_inputs_pad_bounding_box,
        ),
        KernelInfo(
            F.pad_mask,
            sample_inputs_fn=sample_inputs_pad_mask,
            reference_fn=pil_reference_wrapper(F.pad_image_pil),
            reference_inputs_fn=reference_inputs_pad_mask,
            closeness_kwargs=DEFAULT_IMAGE_CLOSENESS_KWARGS,
        ),
    ]
)

_PERSPECTIVE_COEFFS = [
    [1.2405, 0.1772, -6.9113, 0.0463, 1.251, -5.235, 0.00013, 0.0018],
    [0.7366, -0.11724, 1.45775, -0.15012, 0.73406, 2.6019, -0.0072, -0.0063],
]


def sample_inputs_perspective_image_tensor():
    for image_loader in make_image_loaders(
        sizes=["random"],
        # FIXME: kernel should support arbitrary batch sizes
        extra_dims=[(), (4,)],
    ):
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
    for mask_loader in make_mask_loaders(
        sizes=["random"],
        # FIXME: kernel should support arbitrary batch sizes
        extra_dims=[(), (4,)],
    ):
        yield ArgsKwargs(mask_loader, perspective_coeffs=_PERSPECTIVE_COEFFS[0])


def reference_inputs_perspective_mask():
    for mask_loader, perspective_coeffs in itertools.product(
        make_mask_loaders(extra_dims=[()], num_objects=[1]), _PERSPECTIVE_COEFFS
    ):
        yield ArgsKwargs(mask_loader, perspective_coeffs=perspective_coeffs)


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
    ]
)


def _get_elastic_displacement(image_size):
    return torch.rand(1, *image_size, 2)


def sample_inputs_elastic_image_tensor():
    for image_loader in make_image_loaders(
        sizes=["random"],
        # FIXME: kernel should support arbitrary batch sizes
        extra_dims=[(), (4,)],
    ):
        displacement = _get_elastic_displacement(image_loader.image_size)
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
        displacement = _get_elastic_displacement(image_loader.image_size)
        for fill in [None, 128.0, 128, [12.0], [12.0 + c for c in range(image_loader.num_channels)]]:
            yield ArgsKwargs(image_loader, interpolation=interpolation, displacement=displacement, fill=fill)


def sample_inputs_elastic_bounding_box():
    for bounding_box_loader in make_bounding_box_loaders():
        displacement = _get_elastic_displacement(bounding_box_loader.image_size)
        yield ArgsKwargs(
            bounding_box_loader,
            format=bounding_box_loader.format,
            displacement=displacement,
        )


def sample_inputs_elastic_mask():
    for mask_loader in make_mask_loaders(
        sizes=["random"],
        # FIXME: kernel should support arbitrary batch sizes
        extra_dims=[(), (4,)],
    ):
        displacement = _get_elastic_displacement(mask_loader.shape[-2:])
        yield ArgsKwargs(mask_loader, displacement=displacement)


def reference_inputs_elastic_mask():
    for mask_loader in make_mask_loaders(extra_dims=[()], num_objects=[1]):
        displacement = _get_elastic_displacement(mask_loader.shape[-2:])
        yield ArgsKwargs(mask_loader, displacement=displacement)


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
    ]
)


_CENTER_CROP_IMAGE_SIZES = [(16, 16), (7, 33), (31, 9)]
_CENTER_CROP_OUTPUT_SIZES = [[4, 3], [42, 70], [4], 3, (5, 2), (6,)]


def sample_inputs_center_crop_image_tensor():
    for image_loader, output_size in itertools.product(
        make_image_loaders(sizes=_CENTER_CROP_IMAGE_SIZES), _CENTER_CROP_OUTPUT_SIZES
    ):
        yield ArgsKwargs(image_loader, output_size=output_size)


def reference_inputs_center_crop_image_tensor():
    for image_loader, output_size in itertools.product(
        make_image_loaders(sizes=_CENTER_CROP_IMAGE_SIZES, extra_dims=[()]), _CENTER_CROP_OUTPUT_SIZES
    ):
        yield ArgsKwargs(image_loader, output_size=output_size)


def sample_inputs_center_crop_bounding_box():
    for bounding_box_loader, output_size in itertools.product(make_bounding_box_loaders(), _CENTER_CROP_OUTPUT_SIZES):
        yield ArgsKwargs(
            bounding_box_loader,
            format=bounding_box_loader.format,
            image_size=bounding_box_loader.image_size,
            output_size=output_size,
        )


def sample_inputs_center_crop_mask():
    for mask_loader, output_size in itertools.product(
        make_mask_loaders(sizes=_CENTER_CROP_IMAGE_SIZES), _CENTER_CROP_OUTPUT_SIZES
    ):
        yield ArgsKwargs(mask_loader, output_size=output_size)


def reference_inputs_center_crop_mask():
    for mask_loader, output_size in itertools.product(
        make_mask_loaders(sizes=_CENTER_CROP_IMAGE_SIZES, extra_dims=[()], num_objects=[1]), _CENTER_CROP_OUTPUT_SIZES
    ):
        yield ArgsKwargs(mask_loader, output_size=output_size)


KERNEL_INFOS.extend(
    [
        KernelInfo(
            F.center_crop_image_tensor,
            sample_inputs_fn=sample_inputs_center_crop_image_tensor,
            reference_fn=pil_reference_wrapper(F.center_crop_image_pil),
            reference_inputs_fn=reference_inputs_center_crop_image_tensor,
            closeness_kwargs=DEFAULT_IMAGE_CLOSENESS_KWARGS,
            skips=[
                skip_integer_size_jit("output_size"),
            ],
        ),
        KernelInfo(
            F.center_crop_bounding_box,
            sample_inputs_fn=sample_inputs_center_crop_bounding_box,
            skips=[
                skip_integer_size_jit("output_size"),
            ],
        ),
        KernelInfo(
            F.center_crop_mask,
            sample_inputs_fn=sample_inputs_center_crop_mask,
            reference_fn=pil_reference_wrapper(F.center_crop_image_pil),
            reference_inputs_fn=reference_inputs_center_crop_mask,
            closeness_kwargs=DEFAULT_IMAGE_CLOSENESS_KWARGS,
            skips=[
                skip_integer_size_jit("output_size"),
            ],
        ),
    ]
)


def sample_inputs_gaussian_blur_image_tensor():
    for image_loader, params in itertools.product(
        make_image_loaders(
            sizes=["random"],
            # FIXME: kernel should support arbitrary batch sizes
            extra_dims=[(), (4,)],
        ),
        combinations_grid(
            kernel_size=[(3, 3), [3, 3], 5],
            sigma=[None, (3.0, 3.0), [2.0, 2.0], 4.0, [1.5], (3.14,)],
        ),
    ):
        yield ArgsKwargs(image_loader, **params)


KERNEL_INFOS.append(
    KernelInfo(
        F.gaussian_blur_image_tensor,
        sample_inputs_fn=sample_inputs_gaussian_blur_image_tensor,
        closeness_kwargs=DEFAULT_IMAGE_CLOSENESS_KWARGS,
        skips=[
            skip_python_scalar_arg_jit("kernel_size"),
            skip_python_scalar_arg_jit("sigma"),
        ],
    )
)


def sample_inputs_equalize_image_tensor():
    for image_loader in make_image_loaders(
        sizes=["random"],
        # FIXME: kernel should support arbitrary batch sizes
        extra_dims=[(), (4,)],
        color_spaces=(features.ColorSpace.GRAY, features.ColorSpace.RGB),
        dtypes=[torch.uint8],
    ):
        yield ArgsKwargs(image_loader)


def reference_inputs_equalize_image_tensor():
    for image_loader in make_image_loaders(
        extra_dims=[()], color_spaces=(features.ColorSpace.GRAY, features.ColorSpace.RGB), dtypes=[torch.uint8]
    ):
        yield ArgsKwargs(image_loader)


KERNEL_INFOS.append(
    KernelInfo(
        F.equalize_image_tensor,
        kernel_name="equalize_image_tensor",
        sample_inputs_fn=sample_inputs_equalize_image_tensor,
        reference_fn=pil_reference_wrapper(F.equalize_image_pil),
        reference_inputs_fn=reference_inputs_equalize_image_tensor,
        closeness_kwargs=DEFAULT_IMAGE_CLOSENESS_KWARGS,
    )
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


KERNEL_INFOS.append(
    KernelInfo(
        F.invert_image_tensor,
        kernel_name="invert_image_tensor",
        sample_inputs_fn=sample_inputs_invert_image_tensor,
        reference_fn=pil_reference_wrapper(F.invert_image_pil),
        reference_inputs_fn=reference_inputs_invert_image_tensor,
        closeness_kwargs=DEFAULT_IMAGE_CLOSENESS_KWARGS,
    )
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


KERNEL_INFOS.append(
    KernelInfo(
        F.posterize_image_tensor,
        kernel_name="posterize_image_tensor",
        sample_inputs_fn=sample_inputs_posterize_image_tensor,
        reference_fn=pil_reference_wrapper(F.posterize_image_pil),
        reference_inputs_fn=reference_inputs_posterize_image_tensor,
        closeness_kwargs=DEFAULT_IMAGE_CLOSENESS_KWARGS,
    )
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


KERNEL_INFOS.append(
    KernelInfo(
        F.solarize_image_tensor,
        kernel_name="solarize_image_tensor",
        sample_inputs_fn=sample_inputs_solarize_image_tensor,
        reference_fn=pil_reference_wrapper(F.solarize_image_pil),
        reference_inputs_fn=reference_inputs_solarize_image_tensor,
        closeness_kwargs=DEFAULT_IMAGE_CLOSENESS_KWARGS,
    )
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


KERNEL_INFOS.append(
    KernelInfo(
        F.autocontrast_image_tensor,
        kernel_name="autocontrast_image_tensor",
        sample_inputs_fn=sample_inputs_autocontrast_image_tensor,
        reference_fn=pil_reference_wrapper(F.autocontrast_image_pil),
        reference_inputs_fn=reference_inputs_autocontrast_image_tensor,
        closeness_kwargs=DEFAULT_IMAGE_CLOSENESS_KWARGS,
    )
)

_ADJUST_SHARPNESS_FACTORS = [0.1, 0.5]


def sample_inputs_adjust_sharpness_image_tensor():
    for image_loader in make_image_loaders(
        sizes=["random", (2, 2)],
        color_spaces=(features.ColorSpace.GRAY, features.ColorSpace.RGB),
        # FIXME: kernel should support arbitrary batch sizes
        extra_dims=[(), (4,)],
    ):
        yield ArgsKwargs(image_loader, sharpness_factor=_ADJUST_SHARPNESS_FACTORS[0])


def reference_inputs_adjust_sharpness_image_tensor():
    for image_loader, sharpness_factor in itertools.product(
        make_image_loaders(color_spaces=(features.ColorSpace.GRAY, features.ColorSpace.RGB), extra_dims=[()]),
        _ADJUST_SHARPNESS_FACTORS,
    ):
        yield ArgsKwargs(image_loader, sharpness_factor=sharpness_factor)


KERNEL_INFOS.append(
    KernelInfo(
        F.adjust_sharpness_image_tensor,
        kernel_name="adjust_sharpness_image_tensor",
        sample_inputs_fn=sample_inputs_adjust_sharpness_image_tensor,
        reference_fn=pil_reference_wrapper(F.adjust_sharpness_image_pil),
        reference_inputs_fn=reference_inputs_adjust_sharpness_image_tensor,
        closeness_kwargs=DEFAULT_IMAGE_CLOSENESS_KWARGS,
    )
)


def sample_inputs_erase_image_tensor():
    for image_loader in make_image_loaders(sizes=["random"]):
        # FIXME: make the parameters more diverse
        h, w = 6, 7
        v = torch.rand(image_loader.num_channels, h, w)
        yield ArgsKwargs(image_loader, i=1, j=2, h=h, w=w, v=v)


KERNEL_INFOS.append(
    KernelInfo(
        F.erase_image_tensor,
        kernel_name="erase_image_tensor",
        sample_inputs_fn=sample_inputs_erase_image_tensor,
    )
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


KERNEL_INFOS.append(
    KernelInfo(
        F.adjust_brightness_image_tensor,
        kernel_name="adjust_brightness_image_tensor",
        sample_inputs_fn=sample_inputs_adjust_brightness_image_tensor,
        reference_fn=pil_reference_wrapper(F.adjust_brightness_image_pil),
        reference_inputs_fn=reference_inputs_adjust_brightness_image_tensor,
        closeness_kwargs=DEFAULT_IMAGE_CLOSENESS_KWARGS,
    )
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


KERNEL_INFOS.append(
    KernelInfo(
        F.adjust_contrast_image_tensor,
        kernel_name="adjust_contrast_image_tensor",
        sample_inputs_fn=sample_inputs_adjust_contrast_image_tensor,
        reference_fn=pil_reference_wrapper(F.adjust_contrast_image_pil),
        reference_inputs_fn=reference_inputs_adjust_contrast_image_tensor,
        closeness_kwargs=DEFAULT_IMAGE_CLOSENESS_KWARGS,
    )
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


KERNEL_INFOS.append(
    KernelInfo(
        F.adjust_gamma_image_tensor,
        kernel_name="adjust_gamma_image_tensor",
        sample_inputs_fn=sample_inputs_adjust_gamma_image_tensor,
        reference_fn=pil_reference_wrapper(F.adjust_gamma_image_pil),
        reference_inputs_fn=reference_inputs_adjust_gamma_image_tensor,
        closeness_kwargs=DEFAULT_IMAGE_CLOSENESS_KWARGS,
    )
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


KERNEL_INFOS.append(
    KernelInfo(
        F.adjust_hue_image_tensor,
        kernel_name="adjust_hue_image_tensor",
        sample_inputs_fn=sample_inputs_adjust_hue_image_tensor,
        reference_fn=pil_reference_wrapper(F.adjust_hue_image_pil),
        reference_inputs_fn=reference_inputs_adjust_hue_image_tensor,
        closeness_kwargs=DEFAULT_IMAGE_CLOSENESS_KWARGS,
    )
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


KERNEL_INFOS.append(
    KernelInfo(
        F.adjust_saturation_image_tensor,
        kernel_name="adjust_saturation_image_tensor",
        sample_inputs_fn=sample_inputs_adjust_saturation_image_tensor,
        reference_fn=pil_reference_wrapper(F.adjust_saturation_image_pil),
        reference_inputs_fn=reference_inputs_adjust_saturation_image_tensor,
        closeness_kwargs=DEFAULT_IMAGE_CLOSENESS_KWARGS,
    )
)


def sample_inputs_clamp_bounding_box():
    for bounding_box_loader in make_bounding_box_loaders():
        yield ArgsKwargs(
            bounding_box_loader, format=bounding_box_loader.format, image_size=bounding_box_loader.image_size
        )


KERNEL_INFOS.append(
    KernelInfo(
        F.clamp_bounding_box,
        sample_inputs_fn=sample_inputs_clamp_bounding_box,
    )
)

_FIVE_TEN_CROP_SIZES = [7, (6,), [5], (6, 5), [7, 6]]


def _get_five_ten_crop_image_size(size):
    if isinstance(size, int):
        crop_height = crop_width = size
    elif len(size) == 1:
        crop_height = crop_width = size[0]
    else:
        crop_height, crop_width = size
    return 2 * crop_height, 2 * crop_width


def sample_inputs_five_crop_image_tensor():
    for size in _FIVE_TEN_CROP_SIZES:
        for image_loader in make_image_loaders(sizes=[_get_five_ten_crop_image_size(size)]):
            yield ArgsKwargs(image_loader, size=size)


def reference_inputs_five_crop_image_tensor():
    for size in _FIVE_TEN_CROP_SIZES:
        for image_loader in make_image_loaders(sizes=[_get_five_ten_crop_image_size(size)], extra_dims=[()]):
            yield ArgsKwargs(image_loader, size=size)


def sample_inputs_ten_crop_image_tensor():
    for size, vertical_flip in itertools.product(_FIVE_TEN_CROP_SIZES, [False, True]):
        for image_loader in make_image_loaders(sizes=[_get_five_ten_crop_image_size(size)]):
            yield ArgsKwargs(image_loader, size=size, vertical_flip=vertical_flip)


def reference_inputs_ten_crop_image_tensor():
    for size, vertical_flip in itertools.product(_FIVE_TEN_CROP_SIZES, [False, True]):
        for image_loader in make_image_loaders(sizes=[_get_five_ten_crop_image_size(size)], extra_dims=[()]):
            yield ArgsKwargs(image_loader, size=size, vertical_flip=vertical_flip)


KERNEL_INFOS.extend(
    [
        KernelInfo(
            F.five_crop_image_tensor,
            sample_inputs_fn=sample_inputs_five_crop_image_tensor,
            reference_fn=pil_reference_wrapper(F.five_crop_image_pil),
            reference_inputs_fn=reference_inputs_five_crop_image_tensor,
            skips=[
                skip_integer_size_jit(),
                Skip("test_batched_vs_single", reason="Custom batching needed for five_crop_image_tensor."),
            ],
            closeness_kwargs=DEFAULT_IMAGE_CLOSENESS_KWARGS,
        ),
        KernelInfo(
            F.ten_crop_image_tensor,
            sample_inputs_fn=sample_inputs_ten_crop_image_tensor,
            reference_fn=pil_reference_wrapper(F.ten_crop_image_pil),
            reference_inputs_fn=reference_inputs_ten_crop_image_tensor,
            skips=[
                skip_integer_size_jit(),
                Skip("test_batched_vs_single", reason="Custom batching needed for ten_crop_image_tensor."),
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


KERNEL_INFOS.append(
    KernelInfo(
        F.normalize_image_tensor,
        kernel_name="normalize_image_tensor",
        sample_inputs_fn=sample_inputs_normalize_image_tensor,
    )
)
