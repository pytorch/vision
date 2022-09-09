import functools
import itertools
import math

import numpy as np
import pytest
import torch.testing
import torchvision.prototype.transforms.functional as F
from common_utils import cpu_and_gpu, needs_cuda
from prototype_common_utils import ArgsKwargs, assert_close, make_bounding_box_loaders, make_image_loaders
from torchvision.prototype import features


class KernelInfo:
    def __init__(
        self,
        kernel,
        *,
        sample_inputs_fn,
        reference=None,
        reference_inputs_fn=None,
        **closeness_kwargs,
    ):
        self.kernel = kernel
        # smoke test that should hit all valid code paths
        self.sample_inputs_fn = sample_inputs_fn
        self.reference = reference
        self.reference_inputs_fn = reference_inputs_fn or sample_inputs_fn
        self.closeness_kwargs = closeness_kwargs

    def __str__(self):
        return self.kernel.__name__


def pil_reference_wrapper(pil_kernel):
    @functools.wraps(pil_kernel)
    def wrapper(image_tensor, *other_args, **kwargs):
        if image_tensor.ndim > 3:
            raise pytest.UsageError("ADDME")

        return pil_kernel(F.to_image_pil(image_tensor), *other_args, **kwargs)

    return wrapper


KERNEL_INFOS = []


def sample_inputs_horizontal_flip_image_tensor():
    for image_loader in make_image_loaders(dtypes=[torch.float32]):
        yield ArgsKwargs(image_loader.unwrap())


def reference_inputs_horizontal_flip_image_tensor():
    for image_loader in make_image_loaders(extra_dims=[()]):
        yield ArgsKwargs(image_loader.unwrap())


def sample_inputs_horizontal_flip_bounding_box():
    for bounding_box_loader in make_bounding_box_loaders():
        yield ArgsKwargs(
            bounding_box_loader.unwrap(), format=bounding_box_loader.format, image_size=bounding_box_loader.image_size
        )


KERNEL_INFOS.extend(
    [
        KernelInfo(
            F.horizontal_flip_image_tensor,
            sample_inputs_fn=sample_inputs_horizontal_flip_image_tensor,
            reference=pil_reference_wrapper(F.horizontal_flip_image_pil),
            reference_inputs_fn=reference_inputs_horizontal_flip_image_tensor,
            atol=1e-5,
            rtol=0,
            agg_method="mean",
        ),
        KernelInfo(
            F.horizontal_flip_bounding_box,
            sample_inputs_fn=sample_inputs_horizontal_flip_bounding_box,
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
            yield ArgsKwargs(image_loader.unwrap(), size=size, interpolation=interpolation)


def reference_inputs_resize_image_tensor():
    for image, interpolation in itertools.product(
        make_image_loaders(extra_dims=[()]),
        [
            F.InterpolationMode.NEAREST,
            F.InterpolationMode.BILINEAR,
            F.InterpolationMode.BICUBIC,
        ],
    ):
        height, width = image.shape[-2:]
        for size in [
            (height, width),
            (int(height * 0.75), int(width * 1.25)),
        ]:
            yield ArgsKwargs(image, size=size, interpolation=interpolation)


def sample_inputs_resize_bounding_box():
    for bounding_box_loader in make_bounding_box_loaders():
        height, width = bounding_box_loader.image_size
        for size in [
            (height, width),
            (int(height * 0.75), int(width * 1.25)),
        ]:
            yield ArgsKwargs(bounding_box_loader.unwrap(), size=size, image_size=bounding_box_loader.image_size)


KERNEL_INFOS.extend(
    [
        KernelInfo(
            F.resize_image_tensor,
            sample_inputs_fn=sample_inputs_resize_image_tensor,
            reference=pil_reference_wrapper(F.resize_image_pil),
            reference_inputs_fn=reference_inputs_resize_image_tensor,
            atol=1e-5,
            rtol=0,
            agg_method="mean",
        ),
        KernelInfo(
            F.resize_bounding_box,
            sample_inputs_fn=sample_inputs_resize_bounding_box,
        ),
    ]
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
                image_loader.unwrap(),
                angle=-87,
                translate=(5, -5),
                scale=0.77,
                shear=(0, 12),
                interpolation=interpolation_mode,
                center=center,
                fill=fill,
            )


def reference_inputs_affine_image_tensor():
    for image, angle, translate, scale, shear in itertools.product(
        make_image_loaders(extra_dims=[()]),
        [-87, 15, 90],  # angle
        [5, -5],  # translate
        [0.77, 1.27],  # scale
        [0, 12],  # shear
    ):
        yield ArgsKwargs(
            image.unwrap(),
            angle=angle,
            translate=(translate, translate),
            scale=scale,
            shear=(shear, shear),
            interpolation=F.InterpolationMode.NEAREST,
        )


def sample_inputs_affine_bounding_box():
    # FIXME
    return
    yield


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
            out_bbox,
            old_format=features.BoundingBoxFormat.XYXY,
            new_format=format,
            copy=False,
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
            bounding_box_loader.unwrap(),
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
            reference=pil_reference_wrapper(F.affine_image_pil),
            reference_inputs_fn=reference_inputs_affine_image_tensor,
            atol=1e-5,
            rtol=0,
            agg_method="mean",
        ),
        KernelInfo(
            F.affine_bounding_box,
            sample_inputs_fn=sample_inputs_affine_bounding_box,
            reference=reference_affine_bounding_box,
            reference_inputs_fn=reference_inputs_affine_bounding_box,
        ),
    ]
)

sample_inputs = pytest.mark.parametrize(
    ("info", "args_kwargs"),
    [
        pytest.param(info, args_kwargs, id=f"{info}({args_kwargs})")
        for info in KERNEL_INFOS
        for args_kwargs in info.sample_inputs_fn()
    ],
)

reference_inputs = pytest.mark.parametrize(
    ("info", "args_kwargs"),
    [
        pytest.param(info, args_kwargs, id=f"{info}({args_kwargs})")
        for info in KERNEL_INFOS
        for args_kwargs in info.reference_inputs_fn()
        if info.reference is not None
    ],
)


class TestCommon:
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
        (batched_input, *other_args), kwargs = args_kwargs.load(device)

        feature_type = features.Image if features.is_simple_tensor(batched_input) else type(batched_input)
        # This dictionary contains the number of rightmost dimensions that contain the actual data.
        # Everything to the left is considered a batch dimension.
        data_ndim = {
            features.Image: 3,
            features.BoundingBox: 1,
            features.SegmentationMask: 3,
        }.get(feature_type)
        if data_ndim is None:
            raise pytest.UsageError(
                f"The number of data dimensions cannot be determined for input of type {feature_type.__name__}."
            ) from None
        elif batched_input.ndim <= data_ndim:
            pytest.skip("Input is not batched.")
        elif batched_input.ndim > data_ndim + 1:
            # FIXME: We also need to test samples with more than one batch dimension
            pytest.skip("Test currently only supports a single batch dimension")

        actual = info.kernel(batched_input, *other_args, **kwargs).unbind()
        expected = [info.kernel(single_input, *other_args, **kwargs) for single_input in batched_input.unbind()]

        assert_close(actual, expected, **info.closeness_kwargs)

    @sample_inputs
    @pytest.mark.parametrize("device", cpu_and_gpu())
    def test_no_inplace(self, info, args_kwargs, device):
        (input, *other_args), kwargs = args_kwargs.load(device)
        input_version = input._version

        output = info.kernel(input, *other_args, **kwargs)

        assert output is not input or output._version == input_version

    @needs_cuda
    @sample_inputs
    def test_cuda_vs_cpu(self, info, args_kwargs):
        (input_cpu, *other_args), kwargs = args_kwargs.load("cpu")
        input_cuda = input_cpu.to("cuda")

        output_cpu = info.kernel(input_cpu, *other_args, **kwargs)
        output_cuda = info.kernel(input_cuda, *other_args, **kwargs)

        assert_close(output_cuda, output_cpu, check_device=False)

    # FIXME: enforce this only runs on CPU machines
    @reference_inputs
    def test_against_reference(self, info, args_kwargs):
        args, kwargs = args_kwargs.load("cpu")

        actual = info.kernel(*args, **kwargs)
        expected = info.reference(*args, **kwargs)

        assert_close(actual, expected, **info.closeness_kwargs, check_dtype=False)
