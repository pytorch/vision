import functools
import itertools
import math

import numpy as np
import pytest
import torch.testing
import torchvision.prototype.transforms.functional as F
from common_utils import cpu_and_gpu
from torch import jit
from torch.nn.functional import one_hot
from torchvision.prototype import features
from torchvision.prototype.transforms.functional._meta import convert_bounding_box_format
from torchvision.transforms.functional_tensor import _max_value as get_max_value

make_tensor = functools.partial(torch.testing.make_tensor, device="cpu")


def make_image(size=None, *, color_space, extra_dims=(), dtype=torch.float32, constant_alpha=True):
    size = size or torch.randint(16, 33, (2,)).tolist()

    try:
        num_channels = {
            features.ColorSpace.GRAY: 1,
            features.ColorSpace.GRAY_ALPHA: 2,
            features.ColorSpace.RGB: 3,
            features.ColorSpace.RGB_ALPHA: 4,
        }[color_space]
    except KeyError as error:
        raise pytest.UsageError() from error

    shape = (*extra_dims, num_channels, *size)
    max_value = get_max_value(dtype)
    data = make_tensor(shape, low=0, high=max_value, dtype=dtype)
    if color_space in {features.ColorSpace.GRAY_ALPHA, features.ColorSpace.RGB_ALPHA} and constant_alpha:
        data[..., -1, :, :] = max_value
    return features.Image(data, color_space=color_space)


make_grayscale_image = functools.partial(make_image, color_space=features.ColorSpace.GRAY)
make_rgb_image = functools.partial(make_image, color_space=features.ColorSpace.RGB)


def make_images(
    sizes=((16, 16), (7, 33), (31, 9)),
    color_spaces=(
        features.ColorSpace.GRAY,
        features.ColorSpace.GRAY_ALPHA,
        features.ColorSpace.RGB,
        features.ColorSpace.RGB_ALPHA,
    ),
    dtypes=(torch.float32, torch.uint8),
    extra_dims=((4,), (2, 3)),
):
    for size, color_space, dtype in itertools.product(sizes, color_spaces, dtypes):
        yield make_image(size, color_space=color_space, dtype=dtype)

    for color_space, dtype, extra_dims_ in itertools.product(color_spaces, dtypes, extra_dims):
        yield make_image(color_space=color_space, extra_dims=extra_dims_, dtype=dtype)


def randint_with_tensor_bounds(arg1, arg2=None, **kwargs):
    low, high = torch.broadcast_tensors(
        *[torch.as_tensor(arg) for arg in ((0, arg1) if arg2 is None else (arg1, arg2))]
    )
    return torch.stack(
        [
            torch.randint(low_scalar, high_scalar, (), **kwargs)
            for low_scalar, high_scalar in zip(low.flatten().tolist(), high.flatten().tolist())
        ]
    ).reshape(low.shape)


def make_bounding_box(*, format, image_size=(32, 32), extra_dims=(), dtype=torch.int64):
    if isinstance(format, str):
        format = features.BoundingBoxFormat[format]

    height, width = image_size

    if format == features.BoundingBoxFormat.XYXY:
        x1 = torch.randint(0, width // 2, extra_dims)
        y1 = torch.randint(0, height // 2, extra_dims)
        x2 = randint_with_tensor_bounds(x1 + 1, width - x1) + x1
        y2 = randint_with_tensor_bounds(y1 + 1, height - y1) + y1
        parts = (x1, y1, x2, y2)
    elif format == features.BoundingBoxFormat.XYWH:
        x = torch.randint(0, width // 2, extra_dims)
        y = torch.randint(0, height // 2, extra_dims)
        w = randint_with_tensor_bounds(1, width - x)
        h = randint_with_tensor_bounds(1, height - y)
        parts = (x, y, w, h)
    elif format == features.BoundingBoxFormat.CXCYWH:
        cx = torch.randint(1, width - 1, ())
        cy = torch.randint(1, height - 1, ())
        w = randint_with_tensor_bounds(1, torch.minimum(cx, width - cx) + 1)
        h = randint_with_tensor_bounds(1, torch.minimum(cy, width - cy) + 1)
        parts = (cx, cy, w, h)
    else:
        raise pytest.UsageError()

    return features.BoundingBox(torch.stack(parts, dim=-1).to(dtype), format=format, image_size=image_size)


make_xyxy_bounding_box = functools.partial(make_bounding_box, format=features.BoundingBoxFormat.XYXY)


def make_bounding_boxes(
    formats=(features.BoundingBoxFormat.XYXY, features.BoundingBoxFormat.XYWH, features.BoundingBoxFormat.CXCYWH),
    image_sizes=((32, 32),),
    dtypes=(torch.int64, torch.float32),
    extra_dims=((4,), (2, 3)),
):
    for format, image_size, dtype in itertools.product(formats, image_sizes, dtypes):
        yield make_bounding_box(format=format, image_size=image_size, dtype=dtype)

    for format, extra_dims_ in itertools.product(formats, extra_dims):
        yield make_bounding_box(format=format, extra_dims=extra_dims_)


def make_label(size=(), *, categories=("category0", "category1")):
    return features.Label(torch.randint(0, len(categories) if categories else 10, size), categories=categories)


def make_one_hot_label(*args, **kwargs):
    label = make_label(*args, **kwargs)
    return features.OneHotLabel(one_hot(label, num_classes=len(label.categories)), categories=label.categories)


def make_one_hot_labels(
    *,
    num_categories=(1, 2, 10),
    extra_dims=((4,), (2, 3)),
):
    for num_categories_ in num_categories:
        yield make_one_hot_label(categories=[f"category{idx}" for idx in range(num_categories_)])

    for extra_dims_ in extra_dims:
        yield make_one_hot_label(extra_dims_)


class SampleInput:
    def __init__(self, *args, **kwargs):
        self.args = args
        self.kwargs = kwargs


class FunctionalInfo:
    def __init__(self, name, *, sample_inputs_fn):
        self.name = name
        self.functional = getattr(F, name)
        self._sample_inputs_fn = sample_inputs_fn

    def sample_inputs(self):
        yield from self._sample_inputs_fn()

    def __call__(self, *args, **kwargs):
        if len(args) == 1 and not kwargs and isinstance(args[0], SampleInput):
            sample_input = args[0]
            return self.functional(*sample_input.args, **sample_input.kwargs)

        return self.functional(*args, **kwargs)


FUNCTIONAL_INFOS = []


def register_kernel_info_from_sample_inputs_fn(sample_inputs_fn):
    FUNCTIONAL_INFOS.append(FunctionalInfo(sample_inputs_fn.__name__, sample_inputs_fn=sample_inputs_fn))
    return sample_inputs_fn


@register_kernel_info_from_sample_inputs_fn
def horizontal_flip_image_tensor():
    for image in make_images():
        yield SampleInput(image)


@register_kernel_info_from_sample_inputs_fn
def horizontal_flip_bounding_box():
    for bounding_box in make_bounding_boxes(formats=[features.BoundingBoxFormat.XYXY]):
        yield SampleInput(bounding_box, format=bounding_box.format, image_size=bounding_box.image_size)


@register_kernel_info_from_sample_inputs_fn
def resize_image_tensor():
    for image, interpolation in itertools.product(
        make_images(),
        [
            F.InterpolationMode.BILINEAR,
            F.InterpolationMode.NEAREST,
        ],
    ):
        height, width = image.shape[-2:]
        for size in [
            (height, width),
            (int(height * 0.75), int(width * 1.25)),
        ]:
            yield SampleInput(image, size=size, interpolation=interpolation)


@register_kernel_info_from_sample_inputs_fn
def resize_bounding_box():
    for bounding_box in make_bounding_boxes():
        height, width = bounding_box.image_size
        for size in [
            (height, width),
            (int(height * 0.75), int(width * 1.25)),
        ]:
            yield SampleInput(bounding_box, size=size, image_size=bounding_box.image_size)


@register_kernel_info_from_sample_inputs_fn
def affine_image_tensor():
    for image, angle, translate, scale, shear in itertools.product(
        make_images(extra_dims=()),
        [-87, 15, 90],  # angle
        [5, -5],  # translate
        [0.77, 1.27],  # scale
        [0, 12],  # shear
    ):
        yield SampleInput(
            image,
            angle=angle,
            translate=(translate, translate),
            scale=scale,
            shear=(shear, shear),
            interpolation=F.InterpolationMode.NEAREST,
        )


@register_kernel_info_from_sample_inputs_fn
def affine_bounding_box():
    for bounding_box, angle, translate, scale, shear in itertools.product(
        make_bounding_boxes(),
        [-87, 15, 90],  # angle
        [5, -5],  # translate
        [0.77, 1.27],  # scale
        [0, 12],  # shear
    ):
        yield SampleInput(
            bounding_box,
            format=bounding_box.format,
            image_size=bounding_box.image_size,
            angle=angle,
            translate=(translate, translate),
            scale=scale,
            shear=(shear, shear),
        )


@pytest.mark.parametrize(
    "kernel",
    [
        pytest.param(kernel, id=name)
        for name, kernel in F.__dict__.items()
        if not name.startswith("_")
        and callable(kernel)
        and any(feature_type in name for feature_type in {"image", "segmentation_mask", "bounding_box", "label"})
        and "pil" not in name
    ],
)
def test_scriptable(kernel):
    jit.script(kernel)


@pytest.mark.parametrize(
    ("functional_info", "sample_input"),
    [
        pytest.param(functional_info, sample_input, id=f"{functional_info.name}-{idx}")
        for functional_info in FUNCTIONAL_INFOS
        for idx, sample_input in enumerate(functional_info.sample_inputs())
    ],
)
def test_eager_vs_scripted(functional_info, sample_input):
    eager = functional_info(sample_input)
    scripted = jit.script(functional_info.functional)(*sample_input.args, **sample_input.kwargs)

    torch.testing.assert_close(eager, scripted)


def _compute_affine_matrix(angle_, translate_, scale_, shear_, center_):
    rot = math.radians(angle_)
    cx, cy = center_
    tx, ty = translate_
    sx, sy = [math.radians(sh_) for sh_ in shear_]

    c_matrix = np.array([[1, 0, cx], [0, 1, cy], [0, 0, 1]])
    t_matrix = np.array([[1, 0, tx], [0, 1, ty], [0, 0, 1]])
    c_matrix_inv = np.linalg.inv(c_matrix)
    rs_matrix = np.array(
        [
            [scale_ * math.cos(rot), -scale_ * math.sin(rot), 0],
            [scale_ * math.sin(rot), scale_ * math.cos(rot), 0],
            [0, 0, 1],
        ]
    )
    shear_x_matrix = np.array([[1, -math.tan(sx), 0], [0, 1, 0], [0, 0, 1]])
    shear_y_matrix = np.array([[1, 0, 0], [-math.tan(sy), 1, 0], [0, 0, 1]])
    rss_matrix = np.matmul(rs_matrix, np.matmul(shear_y_matrix, shear_x_matrix))
    true_matrix = np.matmul(t_matrix, np.matmul(c_matrix, np.matmul(rss_matrix, c_matrix_inv)))
    return true_matrix


@pytest.mark.parametrize("angle", range(-90, 90, 56))
@pytest.mark.parametrize("translate", range(-10, 10, 8))
@pytest.mark.parametrize("scale", [0.77, 1.0, 1.27])
@pytest.mark.parametrize("shear", range(-15, 15, 8))
@pytest.mark.parametrize("center", [None, (12, 14)])
def test_correctness_affine_bounding_box(angle, translate, scale, shear, center):
    def _compute_expected_bbox(bbox, angle_, translate_, scale_, shear_, center_):
        affine_matrix = _compute_affine_matrix(angle_, translate_, scale_, shear_, center_)
        affine_matrix = affine_matrix[:2, :]

        bbox_xyxy = convert_bounding_box_format(
            bbox, old_format=bbox.format, new_format=features.BoundingBoxFormat.XYXY
        )
        points = np.array(
            [
                [bbox_xyxy[0].item(), bbox_xyxy[1].item(), 1.0],
                [bbox_xyxy[2].item(), bbox_xyxy[1].item(), 1.0],
                [bbox_xyxy[0].item(), bbox_xyxy[3].item(), 1.0],
                [bbox_xyxy[2].item(), bbox_xyxy[3].item(), 1.0],
            ]
        )
        transformed_points = np.matmul(points, affine_matrix.T)
        out_bbox = [
            np.min(transformed_points[:, 0]),
            np.min(transformed_points[:, 1]),
            np.max(transformed_points[:, 0]),
            np.max(transformed_points[:, 1]),
        ]
        out_bbox = features.BoundingBox(
            out_bbox, format=features.BoundingBoxFormat.XYXY, image_size=(32, 32), dtype=torch.float32
        )
        out_bbox = convert_bounding_box_format(
            out_bbox, old_format=features.BoundingBoxFormat.XYXY, new_format=bbox.format, copy=False
        )
        return out_bbox.to(bbox.device)

    image_size = (32, 38)

    for bboxes in make_bounding_boxes(
        image_sizes=[
            image_size,
        ],
        extra_dims=((4,),),
    ):
        output_bboxes = F.affine_bounding_box(
            bboxes,
            bboxes.format,
            image_size=image_size,
            angle=angle,
            translate=(translate, translate),
            scale=scale,
            shear=(shear, shear),
            center=center,
        )
        if center is None:
            center = [s // 2 for s in image_size[::-1]]

        bboxes_format = bboxes.format
        bboxes_image_size = bboxes.image_size
        if bboxes.ndim < 2:
            bboxes = [
                bboxes,
            ]

        expected_bboxes = []
        for bbox in bboxes:
            bbox = features.BoundingBox(bbox, format=bboxes_format, image_size=bboxes_image_size)
            expected_bboxes.append(
                _compute_expected_bbox(bbox, angle, (translate, translate), scale, (shear, shear), center)
            )
        if len(expected_bboxes) > 1:
            expected_bboxes = torch.stack(expected_bboxes)
        else:
            expected_bboxes = expected_bboxes[0]
        torch.testing.assert_close(output_bboxes, expected_bboxes)


@pytest.mark.parametrize("device", cpu_and_gpu())
def test_correctness_affine_bounding_box_on_fixed_input(device):
    # Check transformation against known expected output
    image_size = (64, 64)
    # xyxy format
    in_boxes = [
        [20, 25, 35, 45],
        [50, 5, 70, 22],
        [image_size[1] // 2 - 10, image_size[0] // 2 - 10, image_size[1] // 2 + 10, image_size[0] // 2 + 10],
        [1, 1, 5, 5],
    ]
    in_boxes = features.BoundingBox(
        in_boxes, format=features.BoundingBoxFormat.XYXY, image_size=image_size, dtype=torch.float64
    ).to(device)
    # Tested parameters
    angle = 63
    scale = 0.89
    dx = 0.12
    dy = 0.23

    # Expected bboxes computed using albumentations:
    # from albumentations.augmentations.geometric.functional import bbox_shift_scale_rotate
    # from albumentations.augmentations.geometric.functional import normalize_bbox, denormalize_bbox
    # expected_bboxes = []
    # for in_box in in_boxes:
    #     n_in_box = normalize_bbox(in_box, *image_size)
    #     n_out_box = bbox_shift_scale_rotate(n_in_box, -angle, scale, dx, dy, *image_size)
    #     out_box = denormalize_bbox(n_out_box, *image_size)
    #     expected_bboxes.append(out_box)
    expected_bboxes = [
        (24.522435977922218, 34.375689508290854, 46.443125279998114, 54.3516575015695),
        (54.88288587110401, 50.08453280875634, 76.44484547743795, 72.81332520036864),
        (27.709526487041554, 34.74952648704156, 51.650473512958435, 58.69047351295844),
        (48.56528888843238, 9.611532109828834, 53.35347829361575, 14.39972151501221),
    ]

    output_boxes = F.affine_bounding_box(
        in_boxes,
        in_boxes.format,
        in_boxes.image_size,
        angle,
        (dx * image_size[1], dy * image_size[0]),
        scale,
        shear=(0, 0),
    )

    assert len(output_boxes) == len(expected_bboxes)
    for a_out_box, out_box in zip(expected_bboxes, output_boxes.cpu()):
        np.testing.assert_allclose(out_box.cpu().numpy(), a_out_box)
