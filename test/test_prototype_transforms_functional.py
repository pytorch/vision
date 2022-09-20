import math
import os

import numpy as np
import PIL.Image
import pytest
import torch.testing
import torchvision.prototype.transforms.functional as F
from common_utils import cpu_and_gpu
from prototype_common_utils import ArgsKwargs, make_bounding_boxes, make_image, make_images
from torch import jit
from torchvision.prototype import features
from torchvision.prototype.transforms.functional._geometry import _center_crop_compute_padding
from torchvision.prototype.transforms.functional._meta import convert_format_bounding_box
from torchvision.transforms.functional import _get_perspective_coeffs


class FunctionalInfo:
    def __init__(self, name, *, sample_inputs_fn):
        self.name = name
        self.functional = getattr(F, name)
        self._sample_inputs_fn = sample_inputs_fn

    def sample_inputs(self):
        yield from self._sample_inputs_fn()

    def __call__(self, *args, **kwargs):
        if len(args) == 1 and not kwargs and isinstance(args[0], ArgsKwargs):
            sample_input = args[0]
            return self.functional(*sample_input.args, **sample_input.kwargs)

        return self.functional(*args, **kwargs)


FUNCTIONAL_INFOS = []


def register_kernel_info_from_sample_inputs_fn(sample_inputs_fn):
    FUNCTIONAL_INFOS.append(FunctionalInfo(sample_inputs_fn.__name__, sample_inputs_fn=sample_inputs_fn))
    return sample_inputs_fn


@register_kernel_info_from_sample_inputs_fn
def erase_image_tensor():
    for image in make_images():
        c = image.shape[-3]
        yield ArgsKwargs(image, i=1, j=2, h=6, w=7, v=torch.rand(c, 6, 7))


@pytest.mark.parametrize(
    "kernel",
    [
        pytest.param(kernel, id=name)
        for name, kernel in F.__dict__.items()
        if not name.startswith("_")
        and callable(kernel)
        and any(feature_type in name for feature_type in {"image", "mask", "bounding_box", "label"})
        and "pil" not in name
        and name
        not in {
            "to_image_tensor",
            "get_num_channels",
            "get_spatial_size",
            "get_image_num_channels",
            "get_image_size",
        }
    ],
)
def test_scriptable(kernel):
    jit.script(kernel)


# Test below is intended to test mid-level op vs low-level ops it calls
# For example, resize -> resize_image_tensor, resize_bounding_boxes etc
# TODO: Rewrite this tests as sample args may include more or less params
# than needed by functions
@pytest.mark.parametrize(
    "func",
    [
        pytest.param(func, id=name)
        for name, func in F.__dict__.items()
        if not name.startswith("_")
        and callable(func)
        and all(feature_type not in name for feature_type in {"image", "mask", "bounding_box", "label", "pil"})
        and name
        not in {
            "to_image_tensor",
            "InterpolationMode",
            "decode_video_with_av",
            "crop",
            "perspective",
            "elastic_transform",
            "elastic",
        }
        # We skip 'crop' due to missing 'height' and 'width'
        # We skip 'perspective' as it requires different input args than perspective_image_tensor etc
        # Skip 'elastic', TODO: inspect why test is failing
    ],
)
def test_functional_mid_level(func):
    finfos = [finfo for finfo in FUNCTIONAL_INFOS if f"{func.__name__}_" in finfo.name]
    for finfo in finfos:
        for sample_input in finfo.sample_inputs():
            expected = finfo(sample_input)
            kwargs = dict(sample_input.kwargs)
            for key in ["format", "image_size"]:
                if key in kwargs:
                    del kwargs[key]
            output = func(*sample_input.args, **kwargs)
            torch.testing.assert_close(
                output, expected, msg=f"finfo={finfo.name}, output={output}, expected={expected}"
            )
            break


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


@pytest.mark.parametrize(
    ("functional_info", "sample_input"),
    [
        pytest.param(
            functional_info,
            sample_input,
            id=f"{functional_info.name}-{idx}",
        )
        for functional_info in FUNCTIONAL_INFOS
        for idx, sample_input in enumerate(functional_info.sample_inputs())
    ],
)
def test_dtype_consistency(functional_info, sample_input):
    (input, *other_args), kwargs = sample_input

    output = functional_info.functional(input, *other_args, **kwargs)

    assert output.dtype == input.dtype


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

    out_mask = F.affine_mask(mask, 90, [0.0, 0.0], 64.0 / 32.0, [0.0, 0.0])

    torch.testing.assert_close(out_mask, expected_mask)


@pytest.mark.parametrize("angle", range(-90, 90, 56))
@pytest.mark.parametrize("expand, center", [(True, None), (False, None), (False, (12, 14))])
def test_correctness_rotate_bounding_box(angle, expand, center):
    def _compute_expected_bbox(bbox, angle_, expand_, center_):
        affine_matrix = _compute_affine_matrix(angle_, [0.0, 0.0], 1.0, [0.0, 0.0], center_)
        affine_matrix = affine_matrix[:2, :]

        image_size = bbox.image_size
        bbox_xyxy = convert_format_bounding_box(
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

            # image_size should be updated, but it is OK here to skip its computation
            # as we do not compute it in F.rotate_bounding_box

        out_bbox = features.BoundingBox(
            out_bbox,
            format=features.BoundingBoxFormat.XYXY,
            image_size=image_size,
            dtype=bbox.dtype,
            device=bbox.device,
        )
        return convert_format_bounding_box(
            out_bbox, old_format=features.BoundingBoxFormat.XYXY, new_format=bbox.format, copy=False
        )

    image_size = (32, 38)

    for bboxes in make_bounding_boxes(image_size=image_size, extra_dims=((4,),)):
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
        torch.testing.assert_close(output_bboxes, expected_bboxes, atol=1, rtol=0)


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
    out_mask = F.rotate_mask(mask, 90, expand=False)
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
        in_boxes = convert_format_bounding_box(in_boxes, features.BoundingBoxFormat.XYXY, format)

    output_boxes = F.crop_bounding_box(
        in_boxes,
        format,
        top,
        left,
    )

    if format != features.BoundingBoxFormat.XYXY:
        output_boxes = convert_format_bounding_box(output_boxes, format, features.BoundingBoxFormat.XYXY)

    torch.testing.assert_close(output_boxes.tolist(), expected_bboxes)


@pytest.mark.parametrize("device", cpu_and_gpu())
def test_correctness_horizontal_flip_segmentation_mask_on_fixed_input(device):
    mask = torch.zeros((3, 3, 3), dtype=torch.long, device=device)
    mask[:, :, 0] = 1

    out_mask = F.horizontal_flip_mask(mask)

    expected_mask = torch.zeros((3, 3, 3), dtype=torch.long, device=device)
    expected_mask[:, :, -1] = 1
    torch.testing.assert_close(out_mask, expected_mask)


@pytest.mark.parametrize("device", cpu_and_gpu())
def test_correctness_vertical_flip_segmentation_mask_on_fixed_input(device):
    mask = torch.zeros((3, 3, 3), dtype=torch.long, device=device)
    mask[:, 0, :] = 1

    out_mask = F.vertical_flip_mask(mask)

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
        in_boxes = convert_format_bounding_box(in_boxes, features.BoundingBoxFormat.XYXY, format)

    output_boxes = F.resized_crop_bounding_box(in_boxes, format, top, left, height, width, size)

    if format != features.BoundingBoxFormat.XYXY:
        output_boxes = convert_format_bounding_box(output_boxes, format, features.BoundingBoxFormat.XYXY)

    torch.testing.assert_close(output_boxes, expected_bboxes)


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
        bbox = convert_format_bounding_box(bbox, old_format=bbox_format, new_format=features.BoundingBoxFormat.XYXY)

        bbox[0::2] += pad_left
        bbox[1::2] += pad_up

        bbox = convert_format_bounding_box(
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

        if bboxes.ndim < 2 or bboxes.shape[0] == 0:
            bboxes = [bboxes]

        expected_bboxes = []
        for bbox in bboxes:
            bbox = features.BoundingBox(bbox, format=bboxes_format, image_size=bboxes_image_size)
            expected_bboxes.append(_compute_expected_bbox(bbox, padding))

        if len(expected_bboxes) > 1:
            expected_bboxes = torch.stack(expected_bboxes)
        else:
            expected_bboxes = expected_bboxes[0]
        torch.testing.assert_close(output_boxes, expected_bboxes, atol=1, rtol=0)


@pytest.mark.parametrize("device", cpu_and_gpu())
def test_correctness_pad_segmentation_mask_on_fixed_input(device):
    mask = torch.ones((1, 3, 3), dtype=torch.long, device=device)

    out_mask = F.pad_mask(mask, padding=[1, 1, 1, 1])

    expected_mask = torch.zeros((1, 5, 5), dtype=torch.long, device=device)
    expected_mask[:, 1:-1, 1:-1] = 1
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

        bbox_xyxy = convert_format_bounding_box(
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
            dtype=bbox.dtype,
            device=bbox.device,
        )
        return convert_format_bounding_box(
            out_bbox, old_format=features.BoundingBoxFormat.XYXY, new_format=bbox.format, copy=False
        )

    image_size = (32, 38)

    pcoeffs = _get_perspective_coeffs(startpoints, endpoints)
    inv_pcoeffs = _get_perspective_coeffs(endpoints, startpoints)

    for bboxes in make_bounding_boxes(image_size=image_size, extra_dims=((4,),)):
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
        torch.testing.assert_close(output_bboxes, expected_bboxes, rtol=0, atol=1)


@pytest.mark.parametrize("device", cpu_and_gpu())
@pytest.mark.parametrize(
    "output_size",
    [(18, 18), [18, 15], (16, 19), [12], [46, 48]],
)
def test_correctness_center_crop_bounding_box(device, output_size):
    def _compute_expected_bbox(bbox, output_size_):
        format_ = bbox.format
        image_size_ = bbox.image_size
        bbox = convert_format_bounding_box(bbox, format_, features.BoundingBoxFormat.XYWH)

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
        return convert_format_bounding_box(out_bbox, features.BoundingBoxFormat.XYWH, format_, copy=False)

    for bboxes in make_bounding_boxes(extra_dims=((4,),)):
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
def test_correctness_center_crop_mask(device, output_size):
    def _compute_expected_mask(mask, output_size):
        crop_height, crop_width = output_size if len(output_size) > 1 else [output_size[0], output_size[0]]

        _, image_height, image_width = mask.shape
        if crop_width > image_height or crop_height > image_width:
            padding = _center_crop_compute_padding(crop_height, crop_width, image_height, image_width)
            mask = F.pad_image_tensor(mask, padding, fill=0)

        left = round((image_width - crop_width) * 0.5)
        top = round((image_height - crop_height) * 0.5)

        return mask[:, top : top + crop_height, left : left + crop_width]

    mask = torch.randint(0, 2, size=(1, 6, 6), dtype=torch.long, device=device)
    actual = F.center_crop_mask(mask, output_size)

    expected = _compute_expected_mask(mask, output_size)
    torch.testing.assert_close(expected, actual)


# Copied from test/test_functional_tensor.py
@pytest.mark.parametrize("device", cpu_and_gpu())
@pytest.mark.parametrize("image_size", ("small", "large"))
@pytest.mark.parametrize("dt", [None, torch.float32, torch.float64, torch.float16])
@pytest.mark.parametrize("ksize", [(3, 3), [3, 5], (23, 23)])
@pytest.mark.parametrize("sigma", [[0.5, 0.5], (0.5, 0.5), (0.8, 0.8), (1.7, 1.7)])
def test_correctness_gaussian_blur_image_tensor(device, image_size, dt, ksize, sigma):
    fn = F.gaussian_blur_image_tensor

    # true_cv2_results = {
    #     # np_img = np.arange(3 * 10 * 12, dtype="uint8").reshape((10, 12, 3))
    #     # cv2.GaussianBlur(np_img, ksize=(3, 3), sigmaX=0.8)
    #     "3_3_0.8": ...
    #     # cv2.GaussianBlur(np_img, ksize=(3, 3), sigmaX=0.5)
    #     "3_3_0.5": ...
    #     # cv2.GaussianBlur(np_img, ksize=(3, 5), sigmaX=0.8)
    #     "3_5_0.8": ...
    #     # cv2.GaussianBlur(np_img, ksize=(3, 5), sigmaX=0.5)
    #     "3_5_0.5": ...
    #     # np_img2 = np.arange(26 * 28, dtype="uint8").reshape((26, 28))
    #     # cv2.GaussianBlur(np_img2, ksize=(23, 23), sigmaX=1.7)
    #     "23_23_1.7": ...
    # }
    p = os.path.join(os.path.dirname(os.path.abspath(__file__)), "assets", "gaussian_blur_opencv_results.pt")
    true_cv2_results = torch.load(p)

    if image_size == "small":
        tensor = (
            torch.from_numpy(np.arange(3 * 10 * 12, dtype="uint8").reshape((10, 12, 3))).permute(2, 0, 1).to(device)
        )
    else:
        tensor = torch.from_numpy(np.arange(26 * 28, dtype="uint8").reshape((1, 26, 28))).to(device)

    if dt == torch.float16 and device == "cpu":
        # skip float16 on CPU case
        return

    if dt is not None:
        tensor = tensor.to(dtype=dt)

    _ksize = (ksize, ksize) if isinstance(ksize, int) else ksize
    _sigma = sigma[0] if sigma is not None else None
    shape = tensor.shape
    gt_key = f"{shape[-2]}_{shape[-1]}_{shape[-3]}__{_ksize[0]}_{_ksize[1]}_{_sigma}"
    if gt_key not in true_cv2_results:
        return

    true_out = (
        torch.tensor(true_cv2_results[gt_key]).reshape(shape[-2], shape[-1], shape[-3]).permute(2, 0, 1).to(tensor)
    )

    image = features.Image(tensor)

    out = fn(image, kernel_size=ksize, sigma=sigma)
    torch.testing.assert_close(out, true_out, rtol=0.0, atol=1.0, msg=f"{ksize}, {sigma}")


def test_midlevel_normalize_output_type():
    inpt = torch.rand(1, 3, 32, 32)
    output = F.normalize(inpt, mean=[0.5, 0.5, 0.5], std=[1.0, 1.0, 1.0])
    assert isinstance(output, torch.Tensor)
    torch.testing.assert_close(inpt - 0.5, output)

    inpt = make_image(color_space=features.ColorSpace.RGB)
    output = F.normalize(inpt, mean=[0.5, 0.5, 0.5], std=[1.0, 1.0, 1.0])
    assert isinstance(output, torch.Tensor)
    torch.testing.assert_close(inpt - 0.5, output)


@pytest.mark.parametrize(
    "inpt",
    [
        127 * np.ones((32, 32, 3), dtype="uint8"),
        PIL.Image.new("RGB", (32, 32), 122),
    ],
)
def test_to_image_tensor(inpt):
    output = F.to_image_tensor(inpt)
    assert isinstance(output, torch.Tensor)

    assert np.asarray(inpt).sum() == output.sum().item()

    if isinstance(inpt, PIL.Image.Image):
        # we can't check this option
        # as PIL -> numpy is always copying
        return

    inpt[0, 0, 0] = 11
    assert output[0, 0, 0] == 11


@pytest.mark.parametrize(
    "inpt",
    [
        torch.randint(0, 256, size=(3, 32, 32), dtype=torch.uint8),
        127 * np.ones((32, 32, 3), dtype="uint8"),
    ],
)
@pytest.mark.parametrize("mode", [None, "RGB"])
def test_to_image_pil(inpt, mode):
    output = F.to_image_pil(inpt, mode=mode)
    assert isinstance(output, PIL.Image.Image)

    assert np.asarray(inpt).sum() == np.asarray(output).sum()
