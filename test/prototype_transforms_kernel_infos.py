import dataclasses
import functools
import itertools
import math
from typing import Any, Callable, Dict, Iterable, Optional

import numpy as np
import pytest
import torch.testing
import torchvision.prototype.transforms.functional as F
from datasets_utils import combinations_grid
from prototype_common_utils import ArgsKwargs, make_bounding_box_loaders, make_image_loaders, make_mask_loaders

from torchvision.prototype import features

__all__ = ["KernelInfo", "KERNEL_INFOS"]


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

    def __post_init__(self):
        self.kernel_name = self.kernel_name or self.kernel.__name__
        self.reference_inputs_fn = self.reference_inputs_fn or self.sample_inputs_fn


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


KERNEL_INFOS = []


def sample_inputs_horizontal_flip_image_tensor():
    for image_loader in make_image_loaders(sizes=[None], dtypes=[torch.float32]):
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
    for image_loader in make_mask_loaders(sizes=[None], dtypes=[torch.uint8]):
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
    yield height, width
    yield int(height * 0.75), int(width * 1.25)


def sample_inputs_resize_image_tensor():
    for image_loader, interpolation in itertools.product(
        make_image_loaders(dtypes=[torch.float32]),
        [
            F.InterpolationMode.BILINEAR,
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
        ),
        KernelInfo(
            F.resize_bounding_box,
            sample_inputs_fn=sample_inputs_resize_bounding_box,
        ),
        KernelInfo(
            F.resize_mask,
            sample_inputs_fn=sample_inputs_resize_mask,
            reference_fn=reference_resize_mask,
            reference_inputs_fn=reference_inputs_resize_mask,
            closeness_kwargs=DEFAULT_IMAGE_CLOSENESS_KWARGS,
        ),
    ]
)


_AFFINE_KWARGS = combinations_grid(
    angle=[-87, 15, 90],
    translate=[(5, 5), (-5, -5)],
    scale=[0.77, 1.27],
    shear=[(12, 12), (0, 0)],
)


def sample_inputs_affine_image_tensor():
    for image_loader, interpolation_mode, center in itertools.product(
        make_image_loaders(dtypes=[torch.float32]),
        [
            F.InterpolationMode.NEAREST,
            F.InterpolationMode.BILINEAR,
        ],
        [None, (0, 0)],
    ):
        for fill in [None, [0.5] * image_loader.num_channels]:
            yield ArgsKwargs(
                image_loader,
                interpolation=interpolation_mode,
                center=center,
                fill=fill,
                **_AFFINE_KWARGS[0],
            )


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
            # FIXME: re-add this as soon as the kernel is fixed to also retain the dtype
            # dtype=bbox.dtype,
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
        make_mask_loaders(dtypes=[torch.uint8]),
        [None, (0, 0)],
    ):
        yield ArgsKwargs(mask_loader, center=center, **_AFFINE_KWARGS[0])


@pil_reference_wrapper
def reference_affine_mask(*args, **kwargs):
    return F.affine_image_pil(*args, interpolation=F.InterpolationMode.NEAREST, **kwargs)


def reference_inputs_resize_mask():
    for mask_loader, affine_kwargs in itertools.product(
        make_mask_loaders(extra_dims=[()], num_objects=[1]), _AFFINE_KWARGS
    ):
        yield ArgsKwargs(mask_loader, **affine_kwargs)


KERNEL_INFOS.extend(
    [
        KernelInfo(
            F.affine_image_tensor,
            sample_inputs_fn=sample_inputs_affine_image_tensor,
            reference_fn=pil_reference_wrapper(F.affine_image_pil),
            reference_inputs_fn=reference_inputs_affine_image_tensor,
            closeness_kwargs=DEFAULT_IMAGE_CLOSENESS_KWARGS,
        ),
        KernelInfo(
            F.affine_bounding_box,
            sample_inputs_fn=sample_inputs_affine_bounding_box,
            reference_fn=reference_affine_bounding_box,
            reference_inputs_fn=reference_inputs_affine_bounding_box,
        ),
        KernelInfo(
            F.affine_mask,
            sample_inputs_fn=sample_inputs_affine_image_mask,
            reference_fn=reference_affine_mask,
            reference_inputs_fn=reference_inputs_resize_mask,
            closeness_kwargs=DEFAULT_IMAGE_CLOSENESS_KWARGS,
        ),
    ]
)
