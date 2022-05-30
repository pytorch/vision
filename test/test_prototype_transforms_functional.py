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
from torchvision.prototype.transforms.functional._geometry import _center_crop_compute_padding
from torchvision.prototype.transforms.functional._meta import convert_bounding_box_format
from torchvision.transforms.functional import _get_perspective_coeffs
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
        h = randint_with_tensor_bounds(1, torch.minimum(cy, height - cy) + 1)
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


def make_segmentation_mask(size=None, *, num_categories=80, extra_dims=(), dtype=torch.long):
    size = size or torch.randint(16, 33, (2,)).tolist()
    shape = (*extra_dims, 1, *size)
    data = make_tensor(shape, low=0, high=num_categories, dtype=dtype)
    return features.SegmentationMask(data)


def make_segmentation_masks(
    image_sizes=((16, 16), (7, 33), (31, 9)),
    dtypes=(torch.long,),
    extra_dims=((), (4,), (2, 3)),
):
    for image_size, dtype, extra_dims_ in itertools.product(image_sizes, dtypes, extra_dims):
        yield make_segmentation_mask(size=image_size, dtype=dtype, extra_dims=extra_dims_)


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
        make_images(extra_dims=((), (4,))),
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


@register_kernel_info_from_sample_inputs_fn
def affine_segmentation_mask():
    for mask, angle, translate, scale, shear in itertools.product(
        make_segmentation_masks(extra_dims=((), (4,))),
        [-87, 15, 90],  # angle
        [5, -5],  # translate
        [0.77, 1.27],  # scale
        [0, 12],  # shear
    ):
        yield SampleInput(
            mask,
            angle=angle,
            translate=(translate, translate),
            scale=scale,
            shear=(shear, shear),
        )


@register_kernel_info_from_sample_inputs_fn
def rotate_bounding_box():
    for bounding_box, angle, expand, center in itertools.product(
        make_bounding_boxes(), [-87, 15, 90], [True, False], [None, [12, 23]]
    ):
        if center is not None and expand:
            # Skip warning: The provided center argument is ignored if expand is True
            continue

        yield SampleInput(
            bounding_box,
            format=bounding_box.format,
            image_size=bounding_box.image_size,
            angle=angle,
            expand=expand,
            center=center,
        )


@register_kernel_info_from_sample_inputs_fn
def rotate_segmentation_mask():
    for mask, angle, expand, center in itertools.product(
        make_segmentation_masks(extra_dims=((), (4,))),
        [-87, 15, 90],  # angle
        [True, False],  # expand
        [None, [12, 23]],  # center
    ):
        if center is not None and expand:
            # Skip warning: The provided center argument is ignored if expand is True
            continue

        yield SampleInput(
            mask,
            angle=angle,
            expand=expand,
            center=center,
        )


@register_kernel_info_from_sample_inputs_fn
def crop_bounding_box():
    for bounding_box, top, left in itertools.product(make_bounding_boxes(), [-8, 0, 9], [-8, 0, 9]):
        yield SampleInput(
            bounding_box,
            format=bounding_box.format,
            top=top,
            left=left,
        )


@register_kernel_info_from_sample_inputs_fn
def crop_segmentation_mask():
    for mask, top, left, height, width in itertools.product(
        make_segmentation_masks(), [-8, 0, 9], [-8, 0, 9], [12, 20], [12, 20]
    ):
        yield SampleInput(
            mask,
            top=top,
            left=left,
            height=height,
            width=width,
        )


@register_kernel_info_from_sample_inputs_fn
def vertical_flip_segmentation_mask():
    for mask in make_segmentation_masks():
        yield SampleInput(mask)


@register_kernel_info_from_sample_inputs_fn
def resized_crop_bounding_box():
    for bounding_box, top, left, height, width, size in itertools.product(
        make_bounding_boxes(), [-8, 9], [-8, 9], [32, 22], [34, 20], [(32, 32), (16, 18)]
    ):
        yield SampleInput(
            bounding_box, format=bounding_box.format, top=top, left=left, height=height, width=width, size=size
        )


@register_kernel_info_from_sample_inputs_fn
def resized_crop_segmentation_mask():
    for mask, top, left, height, width, size in itertools.product(
        make_segmentation_masks(), [-8, 0, 9], [-8, 0, 9], [12, 20], [12, 20], [(32, 32), (16, 18)]
    ):
        yield SampleInput(mask, top=top, left=left, height=height, width=width, size=size)


@register_kernel_info_from_sample_inputs_fn
def pad_segmentation_mask():
    for mask, padding, padding_mode in itertools.product(
        make_segmentation_masks(),
        [[1], [1, 1], [1, 1, 2, 2]],  # padding
        ["constant", "symmetric", "edge", "reflect"],  # padding mode,
    ):
        yield SampleInput(mask, padding=padding, padding_mode=padding_mode)


@register_kernel_info_from_sample_inputs_fn
def pad_bounding_box():
    for bounding_box, padding in itertools.product(
        make_bounding_boxes(),
        [[1], [1, 1], [1, 1, 2, 2]],
    ):
        yield SampleInput(bounding_box, padding=padding, format=bounding_box.format)


@register_kernel_info_from_sample_inputs_fn
def perspective_bounding_box():
    for bounding_box, perspective_coeffs in itertools.product(
        make_bounding_boxes(),
        [
            [1.2405, 0.1772, -6.9113, 0.0463, 1.251, -5.235, 0.00013, 0.0018],
            [0.7366, -0.11724, 1.45775, -0.15012, 0.73406, 2.6019, -0.0072, -0.0063],
        ],
    ):
        yield SampleInput(
            bounding_box,
            format=bounding_box.format,
            perspective_coeffs=perspective_coeffs,
        )


@register_kernel_info_from_sample_inputs_fn
def perspective_segmentation_mask():
    for mask, perspective_coeffs in itertools.product(
        make_segmentation_masks(extra_dims=((), (4,))),
        [
            [1.2405, 0.1772, -6.9113, 0.0463, 1.251, -5.235, 0.00013, 0.0018],
            [0.7366, -0.11724, 1.45775, -0.15012, 0.73406, 2.6019, -0.0072, -0.0063],
        ],
    ):
        yield SampleInput(
            mask,
            perspective_coeffs=perspective_coeffs,
        )


@register_kernel_info_from_sample_inputs_fn
def center_crop_bounding_box():
    for bounding_box, output_size in itertools.product(make_bounding_boxes(), [(24, 12), [16, 18], [46, 48], [12]]):
        yield SampleInput(
            bounding_box, format=bounding_box.format, output_size=output_size, image_size=bounding_box.image_size
        )


def center_crop_segmentation_mask():
    for mask, output_size in itertools.product(
        make_segmentation_masks(image_sizes=((16, 16), (7, 33), (31, 9))),
        [[4, 3], [42, 70], [4]],  # crop sizes < image sizes, crop_sizes > image sizes, single crop size
    ):
        yield SampleInput(mask, output_size)


@pytest.mark.parametrize(
    "kernel",
    [
        pytest.param(kernel, id=name)
        for name, kernel in F.__dict__.items()
        if not name.startswith("_")
        and callable(kernel)
        and any(feature_type in name for feature_type in {"image", "segmentation_mask", "bounding_box", "label"})
        and "pil" not in name
        and name
        not in {
            "to_image_tensor",
        }
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
            out_bbox,
            format=features.BoundingBoxFormat.XYXY,
            image_size=bbox.image_size,
            dtype=torch.float32,
            device=bbox.device,
        )
        return convert_bounding_box_format(
            out_bbox, old_format=features.BoundingBoxFormat.XYXY, new_format=bbox.format, copy=False
        )

    image_size = (32, 38)

    for bboxes in make_bounding_boxes(
        image_sizes=[
            image_size,
        ],
        extra_dims=((4,),),
    ):
        bboxes_format = bboxes.format
        bboxes_image_size = bboxes.image_size

        output_bboxes = F.affine_bounding_box(
            bboxes,
            bboxes_format,
            image_size=bboxes_image_size,
            angle=angle,
            translate=(translate, translate),
            scale=scale,
            shear=(shear, shear),
            center=center,
        )

        center_ = center
        if center_ is None:
            center_ = [s * 0.5 for s in bboxes_image_size[::-1]]

        if bboxes.ndim < 2:
            bboxes = [bboxes]

        expected_bboxes = []
        for bbox in bboxes:
            bbox = features.BoundingBox(bbox, format=bboxes_format, image_size=bboxes_image_size)
            expected_bboxes.append(
                _compute_expected_bbox(bbox, angle, (translate, translate), scale, (shear, shear), center_)
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
        in_boxes, format=features.BoundingBoxFormat.XYXY, image_size=image_size, dtype=torch.float64, device=device
    )
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

    torch.testing.assert_close(output_boxes.tolist(), expected_bboxes)


@pytest.mark.parametrize("angle", [-54, 56])
@pytest.mark.parametrize("translate", [-7, 8])
@pytest.mark.parametrize("scale", [0.89, 1.12])
@pytest.mark.parametrize("shear", [4])
@pytest.mark.parametrize("center", [None, (12, 14)])
def test_correctness_affine_segmentation_mask(angle, translate, scale, shear, center):
    def _compute_expected_mask(mask, angle_, translate_, scale_, shear_, center_):
        assert mask.ndim == 3 and mask.shape[0] == 1
        affine_matrix = _compute_affine_matrix(angle_, translate_, scale_, shear_, center_)
        inv_affine_matrix = np.linalg.inv(affine_matrix)
        inv_affine_matrix = inv_affine_matrix[:2, :]

        expected_mask = torch.zeros_like(mask.cpu())
        for out_y in range(expected_mask.shape[1]):
            for out_x in range(expected_mask.shape[2]):
                output_pt = np.array([out_x + 0.5, out_y + 0.5, 1.0])
                input_pt = np.floor(np.dot(inv_affine_matrix, output_pt)).astype(np.int32)
                in_x, in_y = input_pt[:2]
                if 0 <= in_x < mask.shape[2] and 0 <= in_y < mask.shape[1]:
                    expected_mask[0, out_y, out_x] = mask[0, in_y, in_x]
        return expected_mask.to(mask.device)

    for mask in make_segmentation_masks(extra_dims=((), (4,))):
        output_mask = F.affine_segmentation_mask(
            mask,
            angle=angle,
            translate=(translate, translate),
            scale=scale,
            shear=(shear, shear),
            center=center,
        )

        center_ = center
        if center_ is None:
            center_ = [s * 0.5 for s in mask.shape[-2:][::-1]]

        if mask.ndim < 4:
            masks = [mask]
        else:
            masks = [m for m in mask]

        expected_masks = []
        for mask in masks:
            expected_mask = _compute_expected_mask(mask, angle, (translate, translate), scale, (shear, shear), center_)
            expected_masks.append(expected_mask)
        if len(expected_masks) > 1:
            expected_masks = torch.stack(expected_masks)
        else:
            expected_masks = expected_masks[0]
        torch.testing.assert_close(output_mask, expected_masks)


@pytest.mark.parametrize("device", cpu_and_gpu())
def test_correctness_affine_segmentation_mask_on_fixed_input(device):
    # Check transformation against known expected output and CPU/CUDA devices

    # Create a fixed input segmentation mask with 2 square masks
    # in top-left, bottom-left corners
    mask = torch.zeros(1, 32, 32, dtype=torch.long, device=device)
    mask[0, 2:10, 2:10] = 1
    mask[0, 32 - 9 : 32 - 3, 3:9] = 2

    # Rotate 90 degrees and scale
    expected_mask = torch.rot90(mask, k=-1, dims=(-2, -1))
    expected_mask = torch.nn.functional.interpolate(expected_mask[None, :].float(), size=(64, 64), mode="nearest")
    expected_mask = expected_mask[0, :, 16 : 64 - 16, 16 : 64 - 16].long()

    out_mask = F.affine_segmentation_mask(mask, 90, [0.0, 0.0], 64.0 / 32.0, [0.0, 0.0])

    torch.testing.assert_close(out_mask, expected_mask)


@pytest.mark.parametrize("angle", range(-90, 90, 56))
@pytest.mark.parametrize("expand, center", [(True, None), (False, None), (False, (12, 14))])
def test_correctness_rotate_bounding_box(angle, expand, center):
    def _compute_expected_bbox(bbox, angle_, expand_, center_):
        affine_matrix = _compute_affine_matrix(angle_, [0.0, 0.0], 1.0, [0.0, 0.0], center_)
        affine_matrix = affine_matrix[:2, :]

        image_size = bbox.image_size
        bbox_xyxy = convert_bounding_box_format(
            bbox, old_format=bbox.format, new_format=features.BoundingBoxFormat.XYXY
        )
        points = np.array(
            [
                [bbox_xyxy[0].item(), bbox_xyxy[1].item(), 1.0],
                [bbox_xyxy[2].item(), bbox_xyxy[1].item(), 1.0],
                [bbox_xyxy[0].item(), bbox_xyxy[3].item(), 1.0],
                [bbox_xyxy[2].item(), bbox_xyxy[3].item(), 1.0],
                # image frame
                [0.0, 0.0, 1.0],
                [0.0, image_size[0], 1.0],
                [image_size[1], image_size[0], 1.0],
                [image_size[1], 0.0, 1.0],
            ]
        )
        transformed_points = np.matmul(points, affine_matrix.T)
        out_bbox = [
            np.min(transformed_points[:4, 0]),
            np.min(transformed_points[:4, 1]),
            np.max(transformed_points[:4, 0]),
            np.max(transformed_points[:4, 1]),
        ]
        if expand_:
            tr_x = np.min(transformed_points[4:, 0])
            tr_y = np.min(transformed_points[4:, 1])
            out_bbox[0] -= tr_x
            out_bbox[1] -= tr_y
            out_bbox[2] -= tr_x
            out_bbox[3] -= tr_y

        out_bbox = features.BoundingBox(
            out_bbox,
            format=features.BoundingBoxFormat.XYXY,
            image_size=image_size,
            dtype=torch.float32,
            device=bbox.device,
        )
        return convert_bounding_box_format(
            out_bbox, old_format=features.BoundingBoxFormat.XYXY, new_format=bbox.format, copy=False
        )

    image_size = (32, 38)

    for bboxes in make_bounding_boxes(
        image_sizes=[
            image_size,
        ],
        extra_dims=((4,),),
    ):
        bboxes_format = bboxes.format
        bboxes_image_size = bboxes.image_size

        output_bboxes = F.rotate_bounding_box(
            bboxes,
            bboxes_format,
            image_size=bboxes_image_size,
            angle=angle,
            expand=expand,
            center=center,
        )

        center_ = center
        if center_ is None:
            center_ = [s * 0.5 for s in bboxes_image_size[::-1]]

        if bboxes.ndim < 2:
            bboxes = [bboxes]

        expected_bboxes = []
        for bbox in bboxes:
            bbox = features.BoundingBox(bbox, format=bboxes_format, image_size=bboxes_image_size)
            expected_bboxes.append(_compute_expected_bbox(bbox, -angle, expand, center_))
        if len(expected_bboxes) > 1:
            expected_bboxes = torch.stack(expected_bboxes)
        else:
            expected_bboxes = expected_bboxes[0]
        torch.testing.assert_close(output_bboxes, expected_bboxes)


@pytest.mark.parametrize("device", cpu_and_gpu())
@pytest.mark.parametrize("expand", [False])  # expand=True does not match D2
def test_correctness_rotate_bounding_box_on_fixed_input(device, expand):
    # Check transformation against known expected output
    image_size = (64, 64)
    # xyxy format
    in_boxes = [
        [1, 1, 5, 5],
        [1, image_size[0] - 6, 5, image_size[0] - 2],
        [image_size[1] - 6, image_size[0] - 6, image_size[1] - 2, image_size[0] - 2],
        [image_size[1] // 2 - 10, image_size[0] // 2 - 10, image_size[1] // 2 + 10, image_size[0] // 2 + 10],
    ]
    in_boxes = features.BoundingBox(
        in_boxes, format=features.BoundingBoxFormat.XYXY, image_size=image_size, dtype=torch.float64, device=device
    )
    # Tested parameters
    angle = 45
    center = None if expand else [12, 23]

    # # Expected bboxes computed using Detectron2:
    # from detectron2.data.transforms import RotationTransform, AugmentationList
    # from detectron2.data.transforms import AugInput
    # import cv2
    # inpt = AugInput(im1, boxes=np.array(in_boxes, dtype="float32"))
    # augs = AugmentationList([RotationTransform(*size, angle, expand=expand, center=center, interp=cv2.INTER_NEAREST), ])
    # out = augs(inpt)
    # print(inpt.boxes)
    if expand:
        expected_bboxes = [
            [1.65937957, 42.67157288, 7.31623382, 48.32842712],
            [41.96446609, 82.9766594, 47.62132034, 88.63351365],
            [82.26955262, 42.67157288, 87.92640687, 48.32842712],
            [31.35786438, 31.35786438, 59.64213562, 59.64213562],
        ]
    else:
        expected_bboxes = [
            [-11.33452378, 12.39339828, -5.67766953, 18.05025253],
            [28.97056275, 52.69848481, 34.627417, 58.35533906],
            [69.27564928, 12.39339828, 74.93250353, 18.05025253],
            [18.36396103, 1.07968978, 46.64823228, 29.36396103],
        ]

    output_boxes = F.rotate_bounding_box(
        in_boxes,
        in_boxes.format,
        in_boxes.image_size,
        angle,
        expand=expand,
        center=center,
    )

    torch.testing.assert_close(output_boxes.tolist(), expected_bboxes)


@pytest.mark.parametrize("angle", range(-90, 90, 37))
@pytest.mark.parametrize("expand, center", [(True, None), (False, None), (False, (12, 14))])
def test_correctness_rotate_segmentation_mask(angle, expand, center):
    def _compute_expected_mask(mask, angle_, expand_, center_):
        assert mask.ndim == 3 and mask.shape[0] == 1
        image_size = mask.shape[-2:]
        affine_matrix = _compute_affine_matrix(angle_, [0.0, 0.0], 1.0, [0.0, 0.0], center_)
        inv_affine_matrix = np.linalg.inv(affine_matrix)

        if expand_:
            # Pillow implementation on how to perform expand:
            # https://github.com/python-pillow/Pillow/blob/11de3318867e4398057373ee9f12dcb33db7335c/src/PIL/Image.py#L2054-L2069
            height, width = image_size
            points = np.array(
                [
                    [0.0, 0.0, 1.0],
                    [0.0, 1.0 * height, 1.0],
                    [1.0 * width, 1.0 * height, 1.0],
                    [1.0 * width, 0.0, 1.0],
                ]
            )
            new_points = points @ inv_affine_matrix.T
            min_vals = np.min(new_points, axis=0)[:2]
            max_vals = np.max(new_points, axis=0)[:2]
            cmax = np.ceil(np.trunc(max_vals * 1e4) * 1e-4)
            cmin = np.floor(np.trunc((min_vals + 1e-8) * 1e4) * 1e-4)
            new_width, new_height = (cmax - cmin).astype("int32").tolist()
            tr = np.array([-(new_width - width) / 2.0, -(new_height - height) / 2.0, 1.0]) @ inv_affine_matrix.T

            inv_affine_matrix[:2, 2] = tr[:2]
            image_size = [new_height, new_width]

        inv_affine_matrix = inv_affine_matrix[:2, :]
        expected_mask = torch.zeros(1, *image_size, dtype=mask.dtype)

        for out_y in range(expected_mask.shape[1]):
            for out_x in range(expected_mask.shape[2]):
                output_pt = np.array([out_x + 0.5, out_y + 0.5, 1.0])
                input_pt = np.floor(np.dot(inv_affine_matrix, output_pt)).astype(np.int32)
                in_x, in_y = input_pt[:2]
                if 0 <= in_x < mask.shape[2] and 0 <= in_y < mask.shape[1]:
                    expected_mask[0, out_y, out_x] = mask[0, in_y, in_x]
        return expected_mask.to(mask.device)

    for mask in make_segmentation_masks(extra_dims=((), (4,))):
        output_mask = F.rotate_segmentation_mask(
            mask,
            angle=angle,
            expand=expand,
            center=center,
        )

        center_ = center
        if center_ is None:
            center_ = [s * 0.5 for s in mask.shape[-2:][::-1]]

        if mask.ndim < 4:
            masks = [mask]
        else:
            masks = [m for m in mask]

        expected_masks = []
        for mask in masks:
            expected_mask = _compute_expected_mask(mask, -angle, expand, center_)
            expected_masks.append(expected_mask)
        if len(expected_masks) > 1:
            expected_masks = torch.stack(expected_masks)
        else:
            expected_masks = expected_masks[0]
        torch.testing.assert_close(output_mask, expected_masks)


@pytest.mark.parametrize("device", cpu_and_gpu())
def test_correctness_rotate_segmentation_mask_on_fixed_input(device):
    # Check transformation against known expected output and CPU/CUDA devices

    # Create a fixed input segmentation mask with 2 square masks
    # in top-left, bottom-left corners
    mask = torch.zeros(1, 32, 32, dtype=torch.long, device=device)
    mask[0, 2:10, 2:10] = 1
    mask[0, 32 - 9 : 32 - 3, 3:9] = 2

    # Rotate 90 degrees
    expected_mask = torch.rot90(mask, k=1, dims=(-2, -1))
    out_mask = F.rotate_segmentation_mask(mask, 90, expand=False)
    torch.testing.assert_close(out_mask, expected_mask)


@pytest.mark.parametrize("device", cpu_and_gpu())
@pytest.mark.parametrize(
    "format",
    [features.BoundingBoxFormat.XYXY, features.BoundingBoxFormat.XYWH, features.BoundingBoxFormat.CXCYWH],
)
@pytest.mark.parametrize(
    "top, left, height, width, expected_bboxes",
    [
        [8, 12, 30, 40, [(-2.0, 7.0, 13.0, 27.0), (38.0, -3.0, 58.0, 14.0), (33.0, 38.0, 44.0, 54.0)]],
        [-8, 12, 70, 40, [(-2.0, 23.0, 13.0, 43.0), (38.0, 13.0, 58.0, 30.0), (33.0, 54.0, 44.0, 70.0)]],
    ],
)
def test_correctness_crop_bounding_box(device, format, top, left, height, width, expected_bboxes):

    # Expected bboxes computed using Albumentations:
    # import numpy as np
    # from albumentations.augmentations.crops.functional import crop_bbox_by_coords, normalize_bbox, denormalize_bbox
    # expected_bboxes = []
    # for in_box in in_boxes:
    #     n_in_box = normalize_bbox(in_box, *size)
    #     n_out_box = crop_bbox_by_coords(
    #         n_in_box, (left, top, left + width, top + height), height, width, *size
    #     )
    #     out_box = denormalize_bbox(n_out_box, height, width)
    #     expected_bboxes.append(out_box)

    size = (64, 76)
    # xyxy format
    in_boxes = [
        [10.0, 15.0, 25.0, 35.0],
        [50.0, 5.0, 70.0, 22.0],
        [45.0, 46.0, 56.0, 62.0],
    ]
    in_boxes = features.BoundingBox(in_boxes, format=features.BoundingBoxFormat.XYXY, image_size=size, device=device)
    if format != features.BoundingBoxFormat.XYXY:
        in_boxes = convert_bounding_box_format(in_boxes, features.BoundingBoxFormat.XYXY, format)

    output_boxes = F.crop_bounding_box(
        in_boxes,
        format,
        top,
        left,
    )

    if format != features.BoundingBoxFormat.XYXY:
        output_boxes = convert_bounding_box_format(output_boxes, format, features.BoundingBoxFormat.XYXY)

    torch.testing.assert_close(output_boxes.tolist(), expected_bboxes)


@pytest.mark.parametrize("device", cpu_and_gpu())
@pytest.mark.parametrize(
    "top, left, height, width",
    [
        [4, 6, 30, 40],
        [-8, 6, 70, 40],
        [-8, -6, 70, 8],
    ],
)
def test_correctness_crop_segmentation_mask(device, top, left, height, width):
    def _compute_expected_mask(mask, top_, left_, height_, width_):
        h, w = mask.shape[-2], mask.shape[-1]
        if top_ >= 0 and left_ >= 0 and top_ + height_ < h and left_ + width_ < w:
            expected = mask[..., top_ : top_ + height_, left_ : left_ + width_]
        else:
            # Create output mask
            expected_shape = mask.shape[:-2] + (height_, width_)
            expected = torch.zeros(expected_shape, device=mask.device, dtype=mask.dtype)

            out_y1 = abs(top_) if top_ < 0 else 0
            out_y2 = h - top_ if top_ + height_ >= h else height_
            out_x1 = abs(left_) if left_ < 0 else 0
            out_x2 = w - left_ if left_ + width_ >= w else width_

            in_y1 = 0 if top_ < 0 else top_
            in_y2 = h if top_ + height_ >= h else top_ + height_
            in_x1 = 0 if left_ < 0 else left_
            in_x2 = w if left_ + width_ >= w else left_ + width_
            # Paste input mask into output
            expected[..., out_y1:out_y2, out_x1:out_x2] = mask[..., in_y1:in_y2, in_x1:in_x2]

        return expected

    for mask in make_segmentation_masks():
        if mask.device != torch.device(device):
            mask = mask.to(device)
        output_mask = F.crop_segmentation_mask(mask, top, left, height, width)
        expected_mask = _compute_expected_mask(mask, top, left, height, width)
        torch.testing.assert_close(output_mask, expected_mask)


@pytest.mark.parametrize("device", cpu_and_gpu())
def test_correctness_vertical_flip_segmentation_mask_on_fixed_input(device):
    mask = torch.zeros((3, 3, 3), dtype=torch.long, device=device)
    mask[:, 0, :] = 1

    out_mask = F.vertical_flip_segmentation_mask(mask)

    expected_mask = torch.zeros((3, 3, 3), dtype=torch.long, device=device)
    expected_mask[:, -1, :] = 1
    torch.testing.assert_close(out_mask, expected_mask)


@pytest.mark.parametrize("device", cpu_and_gpu())
@pytest.mark.parametrize(
    "format",
    [features.BoundingBoxFormat.XYXY, features.BoundingBoxFormat.XYWH, features.BoundingBoxFormat.CXCYWH],
)
@pytest.mark.parametrize(
    "top, left, height, width, size",
    [
        [0, 0, 30, 30, (60, 60)],
        [-5, 5, 35, 45, (32, 34)],
    ],
)
def test_correctness_resized_crop_bounding_box(device, format, top, left, height, width, size):
    def _compute_expected_bbox(bbox, top_, left_, height_, width_, size_):
        # bbox should be xyxy
        bbox[0] = (bbox[0] - left_) * size_[1] / width_
        bbox[1] = (bbox[1] - top_) * size_[0] / height_
        bbox[2] = (bbox[2] - left_) * size_[1] / width_
        bbox[3] = (bbox[3] - top_) * size_[0] / height_
        return bbox

    image_size = (100, 100)
    # xyxy format
    in_boxes = [
        [10.0, 10.0, 20.0, 20.0],
        [5.0, 10.0, 15.0, 20.0],
    ]
    expected_bboxes = []
    for in_box in in_boxes:
        expected_bboxes.append(_compute_expected_bbox(list(in_box), top, left, height, width, size))
    expected_bboxes = torch.tensor(expected_bboxes, device=device)

    in_boxes = features.BoundingBox(
        in_boxes, format=features.BoundingBoxFormat.XYXY, image_size=image_size, device=device
    )
    if format != features.BoundingBoxFormat.XYXY:
        in_boxes = convert_bounding_box_format(in_boxes, features.BoundingBoxFormat.XYXY, format)

    output_boxes = F.resized_crop_bounding_box(in_boxes, format, top, left, height, width, size)

    if format != features.BoundingBoxFormat.XYXY:
        output_boxes = convert_bounding_box_format(output_boxes, format, features.BoundingBoxFormat.XYXY)

    torch.testing.assert_close(output_boxes, expected_bboxes)


@pytest.mark.parametrize("device", cpu_and_gpu())
@pytest.mark.parametrize(
    "top, left, height, width, size",
    [
        [0, 0, 30, 30, (60, 60)],
        [5, 5, 35, 45, (32, 34)],
    ],
)
def test_correctness_resized_crop_segmentation_mask(device, top, left, height, width, size):
    def _compute_expected_mask(mask, top_, left_, height_, width_, size_):
        output = mask.clone()
        output = output[:, top_ : top_ + height_, left_ : left_ + width_]
        output = torch.nn.functional.interpolate(output[None, :].float(), size=size_, mode="nearest")
        output = output[0, :].long()
        return output

    in_mask = torch.zeros(1, 100, 100, dtype=torch.long, device=device)
    in_mask[0, 10:20, 10:20] = 1
    in_mask[0, 5:15, 12:23] = 2

    expected_mask = _compute_expected_mask(in_mask, top, left, height, width, size)
    output_mask = F.resized_crop_segmentation_mask(in_mask, top, left, height, width, size)
    torch.testing.assert_close(output_mask, expected_mask)


def _parse_padding(padding):
    if isinstance(padding, int):
        return [padding] * 4
    if isinstance(padding, list):
        if len(padding) == 1:
            return padding * 4
        if len(padding) == 2:
            return padding * 2  # [left, up, right, down]

    return padding


@pytest.mark.parametrize("device", cpu_and_gpu())
@pytest.mark.parametrize("padding", [[1], [1, 1], [1, 1, 2, 2]])
def test_correctness_pad_bounding_box(device, padding):
    def _compute_expected_bbox(bbox, padding_):
        pad_left, pad_up, _, _ = _parse_padding(padding_)

        bbox_format = bbox.format
        bbox_dtype = bbox.dtype
        bbox = convert_bounding_box_format(bbox, old_format=bbox_format, new_format=features.BoundingBoxFormat.XYXY)

        bbox[0::2] += pad_left
        bbox[1::2] += pad_up

        bbox = convert_bounding_box_format(
            bbox, old_format=features.BoundingBoxFormat.XYXY, new_format=bbox_format, copy=False
        )
        if bbox.dtype != bbox_dtype:
            # Temporary cast to original dtype
            # e.g. float32 -> int
            bbox = bbox.to(bbox_dtype)
        return bbox

    for bboxes in make_bounding_boxes():
        bboxes = bboxes.to(device)
        bboxes_format = bboxes.format
        bboxes_image_size = bboxes.image_size

        output_boxes = F.pad_bounding_box(bboxes, padding, format=bboxes_format)

        if bboxes.ndim < 2:
            bboxes = [bboxes]

        expected_bboxes = []
        for bbox in bboxes:
            bbox = features.BoundingBox(bbox, format=bboxes_format, image_size=bboxes_image_size)
            expected_bboxes.append(_compute_expected_bbox(bbox, padding))

        if len(expected_bboxes) > 1:
            expected_bboxes = torch.stack(expected_bboxes)
        else:
            expected_bboxes = expected_bboxes[0]
        torch.testing.assert_close(output_boxes, expected_bboxes)


@pytest.mark.parametrize("device", cpu_and_gpu())
def test_correctness_pad_segmentation_mask_on_fixed_input(device):
    mask = torch.ones((1, 3, 3), dtype=torch.long, device=device)

    out_mask = F.pad_segmentation_mask(mask, padding=[1, 1, 1, 1])

    expected_mask = torch.zeros((1, 5, 5), dtype=torch.long, device=device)
    expected_mask[:, 1:-1, 1:-1] = 1
    torch.testing.assert_close(out_mask, expected_mask)


@pytest.mark.parametrize("padding", [[1, 2, 3, 4], [1], 1, [1, 2]])
@pytest.mark.parametrize("padding_mode", ["constant", "edge", "reflect", "symmetric"])
def test_correctness_pad_segmentation_mask(padding, padding_mode):
    def _compute_expected_mask(mask, padding_, padding_mode_):
        h, w = mask.shape[-2], mask.shape[-1]
        pad_left, pad_up, pad_right, pad_down = _parse_padding(padding_)

        if any(pad <= 0 for pad in [pad_left, pad_up, pad_right, pad_down]):
            raise pytest.UsageError(
                "Expected output can be computed on positive pad values only, "
                "but F.pad_* can also crop for negative values"
            )

        new_h = h + pad_up + pad_down
        new_w = w + pad_left + pad_right

        new_shape = (*mask.shape[:-2], new_h, new_w) if len(mask.shape) > 2 else (new_h, new_w)
        output = torch.zeros(new_shape, dtype=mask.dtype)
        output[..., pad_up:-pad_down, pad_left:-pad_right] = mask

        if padding_mode_ == "edge":
            # pad top-left corner, left vertical block, bottom-left corner
            output[..., :pad_up, :pad_left] = mask[..., 0, 0].unsqueeze(-1).unsqueeze(-2)
            output[..., pad_up:-pad_down, :pad_left] = mask[..., :, 0].unsqueeze(-1)
            output[..., -pad_down:, :pad_left] = mask[..., -1, 0].unsqueeze(-1).unsqueeze(-2)
            # pad top-right corner, right vertical block, bottom-right corner
            output[..., :pad_up, -pad_right:] = mask[..., 0, -1].unsqueeze(-1).unsqueeze(-2)
            output[..., pad_up:-pad_down, -pad_right:] = mask[..., :, -1].unsqueeze(-1)
            output[..., -pad_down:, -pad_right:] = mask[..., -1, -1].unsqueeze(-1).unsqueeze(-2)
            # pad top and bottom horizontal blocks
            output[..., :pad_up, pad_left:-pad_right] = mask[..., 0, :].unsqueeze(-2)
            output[..., -pad_down:, pad_left:-pad_right] = mask[..., -1, :].unsqueeze(-2)
        elif padding_mode_ in ("reflect", "symmetric"):
            d1 = 1 if padding_mode_ == "reflect" else 0
            d2 = -1 if padding_mode_ == "reflect" else None
            both = (-1, -2)
            # pad top-left corner, left vertical block, bottom-left corner
            output[..., :pad_up, :pad_left] = mask[..., d1 : pad_up + d1, d1 : pad_left + d1].flip(both)
            output[..., pad_up:-pad_down, :pad_left] = mask[..., :, d1 : pad_left + d1].flip(-1)
            output[..., -pad_down:, :pad_left] = mask[..., -pad_down - d1 : d2, d1 : pad_left + d1].flip(both)
            # pad top-right corner, right vertical block, bottom-right corner
            output[..., :pad_up, -pad_right:] = mask[..., d1 : pad_up + d1, -pad_right - d1 : d2].flip(both)
            output[..., pad_up:-pad_down, -pad_right:] = mask[..., :, -pad_right - d1 : d2].flip(-1)
            output[..., -pad_down:, -pad_right:] = mask[..., -pad_down - d1 : d2, -pad_right - d1 : d2].flip(both)
            # pad top and bottom horizontal blocks
            output[..., :pad_up, pad_left:-pad_right] = mask[..., d1 : pad_up + d1, :].flip(-2)
            output[..., -pad_down:, pad_left:-pad_right] = mask[..., -pad_down - d1 : d2, :].flip(-2)

        return output

    for mask in make_segmentation_masks():
        out_mask = F.pad_segmentation_mask(mask, padding, padding_mode=padding_mode)

        expected_mask = _compute_expected_mask(mask, padding, padding_mode)
        torch.testing.assert_close(out_mask, expected_mask)


@pytest.mark.parametrize("device", cpu_and_gpu())
@pytest.mark.parametrize(
    "startpoints, endpoints",
    [
        [[[0, 0], [33, 0], [33, 25], [0, 25]], [[3, 2], [32, 3], [30, 24], [2, 25]]],
        [[[3, 2], [32, 3], [30, 24], [2, 25]], [[0, 0], [33, 0], [33, 25], [0, 25]]],
        [[[3, 2], [32, 3], [30, 24], [2, 25]], [[5, 5], [30, 3], [33, 19], [4, 25]]],
    ],
)
def test_correctness_perspective_bounding_box(device, startpoints, endpoints):
    def _compute_expected_bbox(bbox, pcoeffs_):
        m1 = np.array(
            [
                [pcoeffs_[0], pcoeffs_[1], pcoeffs_[2]],
                [pcoeffs_[3], pcoeffs_[4], pcoeffs_[5]],
            ]
        )
        m2 = np.array(
            [
                [pcoeffs_[6], pcoeffs_[7], 1.0],
                [pcoeffs_[6], pcoeffs_[7], 1.0],
            ]
        )

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
        numer = np.matmul(points, m1.T)
        denom = np.matmul(points, m2.T)
        transformed_points = numer / denom
        out_bbox = [
            np.min(transformed_points[:, 0]),
            np.min(transformed_points[:, 1]),
            np.max(transformed_points[:, 0]),
            np.max(transformed_points[:, 1]),
        ]
        out_bbox = features.BoundingBox(
            out_bbox,
            format=features.BoundingBoxFormat.XYXY,
            image_size=bbox.image_size,
            dtype=torch.float32,
            device=bbox.device,
        )
        return convert_bounding_box_format(
            out_bbox, old_format=features.BoundingBoxFormat.XYXY, new_format=bbox.format, copy=False
        )

    image_size = (32, 38)

    pcoeffs = _get_perspective_coeffs(startpoints, endpoints)
    inv_pcoeffs = _get_perspective_coeffs(endpoints, startpoints)

    for bboxes in make_bounding_boxes(
        image_sizes=[
            image_size,
        ],
        extra_dims=((4,),),
    ):
        bboxes = bboxes.to(device)
        bboxes_format = bboxes.format
        bboxes_image_size = bboxes.image_size

        output_bboxes = F.perspective_bounding_box(
            bboxes,
            bboxes_format,
            perspective_coeffs=pcoeffs,
        )

        if bboxes.ndim < 2:
            bboxes = [bboxes]

        expected_bboxes = []
        for bbox in bboxes:
            bbox = features.BoundingBox(bbox, format=bboxes_format, image_size=bboxes_image_size)
            expected_bboxes.append(_compute_expected_bbox(bbox, inv_pcoeffs))
        if len(expected_bboxes) > 1:
            expected_bboxes = torch.stack(expected_bboxes)
        else:
            expected_bboxes = expected_bboxes[0]
        torch.testing.assert_close(output_bboxes, expected_bboxes, rtol=1e-5, atol=1e-5)


@pytest.mark.parametrize("device", cpu_and_gpu())
@pytest.mark.parametrize(
    "startpoints, endpoints",
    [
        [[[0, 0], [33, 0], [33, 25], [0, 25]], [[3, 2], [32, 3], [30, 24], [2, 25]]],
        [[[3, 2], [32, 3], [30, 24], [2, 25]], [[0, 0], [33, 0], [33, 25], [0, 25]]],
        [[[3, 2], [32, 3], [30, 24], [2, 25]], [[5, 5], [30, 3], [33, 19], [4, 25]]],
    ],
)
def test_correctness_perspective_segmentation_mask(device, startpoints, endpoints):
    def _compute_expected_mask(mask, pcoeffs_):
        assert mask.ndim == 3 and mask.shape[0] == 1
        m1 = np.array(
            [
                [pcoeffs_[0], pcoeffs_[1], pcoeffs_[2]],
                [pcoeffs_[3], pcoeffs_[4], pcoeffs_[5]],
            ]
        )
        m2 = np.array(
            [
                [pcoeffs_[6], pcoeffs_[7], 1.0],
                [pcoeffs_[6], pcoeffs_[7], 1.0],
            ]
        )

        expected_mask = torch.zeros_like(mask.cpu())
        for out_y in range(expected_mask.shape[1]):
            for out_x in range(expected_mask.shape[2]):
                output_pt = np.array([out_x + 0.5, out_y + 0.5, 1.0])

                numer = np.matmul(output_pt, m1.T)
                denom = np.matmul(output_pt, m2.T)
                input_pt = np.floor(numer / denom).astype(np.int32)

                in_x, in_y = input_pt[:2]
                if 0 <= in_x < mask.shape[2] and 0 <= in_y < mask.shape[1]:
                    expected_mask[0, out_y, out_x] = mask[0, in_y, in_x]
        return expected_mask.to(mask.device)

    pcoeffs = _get_perspective_coeffs(startpoints, endpoints)

    for mask in make_segmentation_masks(extra_dims=((), (4,))):
        mask = mask.to(device)

        output_mask = F.perspective_segmentation_mask(
            mask,
            perspective_coeffs=pcoeffs,
        )

        if mask.ndim < 4:
            masks = [mask]
        else:
            masks = [m for m in mask]

        expected_masks = []
        for mask in masks:
            expected_mask = _compute_expected_mask(mask, pcoeffs)
            expected_masks.append(expected_mask)
        if len(expected_masks) > 1:
            expected_masks = torch.stack(expected_masks)
        else:
            expected_masks = expected_masks[0]
        torch.testing.assert_close(output_mask, expected_masks)


@pytest.mark.parametrize("device", cpu_and_gpu())
@pytest.mark.parametrize(
    "output_size",
    [(18, 18), [18, 15], (16, 19), [12], [46, 48]],
)
def test_correctness_center_crop_bounding_box(device, output_size):
    def _compute_expected_bbox(bbox, output_size_):
        format_ = bbox.format
        image_size_ = bbox.image_size
        bbox = convert_bounding_box_format(bbox, format_, features.BoundingBoxFormat.XYWH)

        if len(output_size_) == 1:
            output_size_.append(output_size_[-1])

        cy = int(round((image_size_[0] - output_size_[0]) * 0.5))
        cx = int(round((image_size_[1] - output_size_[1]) * 0.5))
        out_bbox = [
            bbox[0].item() - cx,
            bbox[1].item() - cy,
            bbox[2].item(),
            bbox[3].item(),
        ]
        out_bbox = features.BoundingBox(
            out_bbox,
            format=features.BoundingBoxFormat.XYWH,
            image_size=output_size_,
            dtype=bbox.dtype,
            device=bbox.device,
        )
        return convert_bounding_box_format(out_bbox, features.BoundingBoxFormat.XYWH, format_, copy=False)

    for bboxes in make_bounding_boxes(
        image_sizes=[(32, 32), (24, 33), (32, 25)],
        extra_dims=((4,),),
    ):
        bboxes = bboxes.to(device)
        bboxes_format = bboxes.format
        bboxes_image_size = bboxes.image_size

        output_boxes = F.center_crop_bounding_box(bboxes, bboxes_format, output_size, bboxes_image_size)

        if bboxes.ndim < 2:
            bboxes = [bboxes]

        expected_bboxes = []
        for bbox in bboxes:
            bbox = features.BoundingBox(bbox, format=bboxes_format, image_size=bboxes_image_size)
            expected_bboxes.append(_compute_expected_bbox(bbox, output_size))

        if len(expected_bboxes) > 1:
            expected_bboxes = torch.stack(expected_bboxes)
        else:
            expected_bboxes = expected_bboxes[0]
        torch.testing.assert_close(output_boxes, expected_bboxes)


@pytest.mark.parametrize("device", cpu_and_gpu())
@pytest.mark.parametrize("output_size", [[4, 2], [4], [7, 6]])
def test_correctness_center_crop_segmentation_mask(device, output_size):
    def _compute_expected_segmentation_mask(mask, output_size):
        crop_height, crop_width = output_size if len(output_size) > 1 else [output_size[0], output_size[0]]

        _, image_height, image_width = mask.shape
        if crop_width > image_height or crop_height > image_width:
            padding = _center_crop_compute_padding(crop_height, crop_width, image_height, image_width)
            mask = F.pad_image_tensor(mask, padding, fill=0)

        left = round((image_width - crop_width) * 0.5)
        top = round((image_height - crop_height) * 0.5)

        return mask[:, top : top + crop_height, left : left + crop_width]

    mask = torch.randint(0, 2, size=(1, 6, 6), dtype=torch.long, device=device)
    actual = F.center_crop_segmentation_mask(mask, output_size)

    expected = _compute_expected_segmentation_mask(mask, output_size)
    torch.testing.assert_close(expected, actual)
