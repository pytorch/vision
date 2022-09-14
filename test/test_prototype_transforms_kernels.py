import dataclasses
import functools
import itertools
import math
from typing import Any, Callable, Dict, Iterable, Optional

import numpy as np
import pytest
import torch.testing
import torchvision.prototype.transforms.functional as F
from common_utils import cpu_and_gpu, needs_cuda
from datasets_utils import combinations_grid
from prototype_common_utils import (
    ArgsKwargs,
    assert_close,
    make_bounding_box_loaders,
    make_image_loaders,
    make_mask_loaders,
)

from torch.utils._pytree import tree_map
from torchvision.prototype import features


@dataclasses.dataclass
class KernelInfo:
    kernel: Callable
    # Most common tests use these inputs to check the kernel. As such it should cover all valid code paths, but should
    # not include extensive parameter combinations to keep to overall test count moderate.
    sample_inputs_fn: Callable[[], Iterable[ArgsKwargs]]
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
    for image_loader in make_image_loaders(dtypes=[torch.float32]):
        yield ArgsKwargs(image_loader)


def reference_inputs_horizontal_flip_image_tensor():
    for image_loader in make_image_loaders(extra_dims=[()]):
        yield ArgsKwargs(image_loader)


def sample_inputs_horizontal_flip_bounding_box():
    for bounding_box_loader in make_bounding_box_loaders():
        yield ArgsKwargs(
            bounding_box_loader, format=bounding_box_loader.format, image_size=bounding_box_loader.image_size
        )


def sample_inputs_horizontal_flip_mask():
    for image_loader in make_mask_loaders(dtypes=[torch.uint8]):
        yield ArgsKwargs(image_loader)


KERNEL_INFOS.extend(
    [
        KernelInfo(
            F.horizontal_flip_image_tensor,
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


def sample_inputs_resize_image_tensor():
    for image_loader, interpolation in itertools.product(
        make_image_loaders(dtypes=[torch.float32]),
        [
            F.InterpolationMode.NEAREST,
            F.InterpolationMode.BILINEAR,
            F.InterpolationMode.BICUBIC,
        ],
    ):
        height, width = image_loader.image_size
        for size in [
            (height, width),
            (int(height * 0.75), int(width * 1.25)),
        ]:
            yield ArgsKwargs(image_loader, size=size, interpolation=interpolation)


def reference_inputs_resize_image_tensor():
    for image_loader, interpolation in itertools.product(
        make_image_loaders(extra_dims=[()]),
        [
            F.InterpolationMode.NEAREST,
            F.InterpolationMode.BILINEAR,
            F.InterpolationMode.BICUBIC,
        ],
    ):
        height, width = image_loader.image_size
        for size in [
            (height, width),
            (int(height * 0.75), int(width * 1.25)),
        ]:
            yield ArgsKwargs(image_loader, size=size, interpolation=interpolation)


def sample_inputs_resize_bounding_box():
    for bounding_box_loader in make_bounding_box_loaders():
        height, width = bounding_box_loader.image_size
        for size in [
            (height, width),
            (int(height * 0.75), int(width * 1.25)),
        ]:
            yield ArgsKwargs(bounding_box_loader, size=size, image_size=bounding_box_loader.image_size)


KERNEL_INFOS.extend(
    [
        KernelInfo(
            F.resize_image_tensor,
            sample_inputs_fn=sample_inputs_resize_image_tensor,
            reference_fn=pil_reference_wrapper(F.resize_image_pil),
            reference_inputs_fn=reference_inputs_resize_image_tensor,
            closeness_kwargs=DEFAULT_IMAGE_CLOSENESS_KWARGS,
        ),
        KernelInfo(
            F.resize_bounding_box,
            sample_inputs_fn=sample_inputs_resize_bounding_box,
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
    for image, affine_kwargs in itertools.product(make_image_loaders(extra_dims=[()]), _AFFINE_KWARGS):
        yield ArgsKwargs(
            image,
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


def reference_affine_bounding_box(bounding_box, *, format, image_size, angle, translate, scale, shear, center):
    if center is None:
        center = [s * 0.5 for s in image_size[::-1]]

    def transform(bbox):
        affine_matrix = _compute_affine_matrix(angle, translate, scale, shear, center)
        affine_matrix = affine_matrix[:2, :]

        bbox_xyxy = F.convert_bounding_box_format(bbox, old_format=format, new_format=features.BoundingBoxFormat.XYXY)
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
        return F.convert_bounding_box_format(
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
    for bounding_box_loader, angle, translate, scale, shear, center in itertools.product(
        make_bounding_box_loaders(extra_dims=[(4,)], image_size=(32, 38), dtypes=[torch.float32]),
        range(-90, 90, 56),
        range(-10, 10, 8),
        [0.77, 1.0, 1.27],
        range(-15, 15, 8),
        [None, (12, 14)],
    ):
        yield ArgsKwargs(
            bounding_box_loader,
            format=bounding_box_loader.format,
            image_size=bounding_box_loader.image_size,
            angle=angle,
            translate=(translate, translate),
            scale=scale,
            shear=(shear, shear),
            center=center,
        )


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
    ]
)


class TestCommon:
    sample_inputs = pytest.mark.parametrize(
        ("info", "args_kwargs"),
        [
            pytest.param(info, args_kwargs, id=f"{info.kernel.__name__}")
            for info in KERNEL_INFOS
            for args_kwargs in info.sample_inputs_fn()
        ],
    )

    @sample_inputs
    @pytest.mark.parametrize("device", cpu_and_gpu())
    def test_scripted_vs_eager(self, info, args_kwargs, device):
        kernel_eager = info.kernel
        try:
            kernel_scripted = torch.jit.script(kernel_eager)
        except Exception as error:
            raise AssertionError("Trying to `torch.jit.script` the kernel raised the error above.") from error

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

        assert_close(output_cuda, output_cpu, check_device=False)

    @pytest.mark.parametrize(
        ("info", "args_kwargs"),
        [
            pytest.param(info, args_kwargs, id=f"{info.kernel.__name__}")
            for info in KERNEL_INFOS
            for args_kwargs in info.reference_inputs_fn()
            if info.reference_fn is not None
        ],
    )
    def test_against_reference(self, info, args_kwargs):
        args, kwargs = args_kwargs.load("cpu")

        actual = info.kernel(*args, **kwargs)
        expected = info.reference_fn(*args, **kwargs)

        assert_close(actual, expected, **info.closeness_kwargs, check_dtype=False)
