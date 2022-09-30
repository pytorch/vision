import math
import os

import numpy as np
import PIL.Image
import pytest

import torch
from common_utils import cache, cpu_and_gpu, needs_cuda
from prototype_common_utils import assert_close, make_bounding_boxes, make_image
from prototype_transforms_dispatcher_infos import DISPATCHER_INFOS
from prototype_transforms_kernel_infos import KERNEL_INFOS
from torch.utils._pytree import tree_map
from torchvision.prototype import features
from torchvision.prototype.transforms import functional as F
from torchvision.prototype.transforms.functional._geometry import _center_crop_compute_padding
from torchvision.prototype.transforms.functional._meta import convert_format_bounding_box
from torchvision.transforms.functional import _get_perspective_coeffs


@cache
def script(fn):
    try:
        return torch.jit.script(fn)
    except Exception as error:
        raise AssertionError(f"Trying to `torch.jit.script` '{fn.__name__}' raised the error above.") from error


@pytest.fixture(autouse=True)
def maybe_skip(request):
    # In case the test uses no parametrization or fixtures, the `callspec` attribute does not exist
    try:
        callspec = request.node.callspec
    except AttributeError:
        return

    try:
        info = callspec.params["info"]
        args_kwargs = callspec.params["args_kwargs"]
    except KeyError:
        return

    info.maybe_skip(
        test_name=request.node.originalname, args_kwargs=args_kwargs, device=callspec.params.get("device", "cpu")
    )


class TestKernels:
    sample_inputs = pytest.mark.parametrize(
        ("info", "args_kwargs"),
        [
            pytest.param(info, args_kwargs, id=f"{info.kernel_name}-{idx}")
            for info in KERNEL_INFOS
            for idx, args_kwargs in enumerate(info.sample_inputs_fn())
        ],
    )

    @sample_inputs
    @pytest.mark.parametrize("device", cpu_and_gpu())
    def test_scripted_vs_eager(self, info, args_kwargs, device):
        kernel_eager = info.kernel
        kernel_scripted = script(kernel_eager)

        args, kwargs = args_kwargs.load(device)

        actual = kernel_scripted(*args, **kwargs)
        expected = kernel_eager(*args, **kwargs)

        assert_close(actual, expected, **info.closeness_kwargs)

    def _unbatch(self, batch, *, data_dims):
        if isinstance(batch, torch.Tensor):
            batched_tensor = batch
            metadata = ()
        else:
            batched_tensor, *metadata = batch

        if batched_tensor.ndim == data_dims:
            return batch

        return [
            self._unbatch(unbatched, data_dims=data_dims)
            for unbatched in (
                batched_tensor.unbind(0) if not metadata else [(t, *metadata) for t in batched_tensor.unbind(0)]
            )
        ]

    @sample_inputs
    @pytest.mark.parametrize("device", cpu_and_gpu())
    def test_batched_vs_single(self, info, args_kwargs, device):
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

        batched_output = info.kernel(batched_input, *other_args, **kwargs)
        actual = self._unbatch(batched_output, data_dims=data_dims)

        single_inputs = self._unbatch(batched_input, data_dims=data_dims)
        expected = tree_map(lambda single_input: info.kernel(single_input, *other_args, **kwargs), single_inputs)

        assert_close(actual, expected, **info.closeness_kwargs)

    @sample_inputs
    @pytest.mark.parametrize("device", cpu_and_gpu())
    def test_no_inplace(self, info, args_kwargs, device):
        (input, *other_args), kwargs = args_kwargs.load(device)

        if input.numel() == 0:
            pytest.skip("The input has a degenerate shape.")

        input_version = input._version
        info.kernel(input, *other_args, **kwargs)

        assert input._version == input_version

    @sample_inputs
    @needs_cuda
    def test_cuda_vs_cpu(self, info, args_kwargs):
        (input_cpu, *other_args), kwargs = args_kwargs.load("cpu")
        input_cuda = input_cpu.to("cuda")

        output_cpu = info.kernel(input_cpu, *other_args, **kwargs)
        output_cuda = info.kernel(input_cuda, *other_args, **kwargs)

        assert_close(output_cuda, output_cpu, check_device=False, **info.closeness_kwargs)

    @sample_inputs
    @pytest.mark.parametrize("device", cpu_and_gpu())
    def test_dtype_and_device_consistency(self, info, args_kwargs, device):
        (input, *other_args), kwargs = args_kwargs.load(device)

        output = info.kernel(input, *other_args, **kwargs)
        # Most kernels just return a tensor, but some also return some additional metadata
        if not isinstance(output, torch.Tensor):
            output, *_ = output

        assert output.dtype == input.dtype
        assert output.device == input.device

    @pytest.mark.parametrize(
        ("info", "args_kwargs"),
        [
            pytest.param(info, args_kwargs, id=f"{info.kernel_name}-{idx}")
            for info in KERNEL_INFOS
            for idx, args_kwargs in enumerate(info.reference_inputs_fn())
            if info.reference_fn is not None
        ],
    )
    def test_against_reference(self, info, args_kwargs):
        args, kwargs = args_kwargs.load("cpu")

        actual = info.kernel(*args, **kwargs)
        expected = info.reference_fn(*args, **kwargs)

        assert_close(actual, expected, check_dtype=False, **info.closeness_kwargs)


class TestDispatchers:
    @pytest.mark.parametrize(
        ("info", "args_kwargs"),
        [
            pytest.param(info, args_kwargs, id=f"{info.dispatcher.__name__}-{idx}")
            for info in DISPATCHER_INFOS
            for idx, args_kwargs in enumerate(info.sample_inputs(features.Image))
            if features.Image in info.kernels
        ],
    )
    @pytest.mark.parametrize("device", cpu_and_gpu())
    def test_scripted_smoke(self, info, args_kwargs, device):
        dispatcher = script(info.dispatcher)

        (image_feature, *other_args), kwargs = args_kwargs.load(device)
        image_simple_tensor = torch.Tensor(image_feature)

        dispatcher(image_simple_tensor, *other_args, **kwargs)

    # TODO: We need this until the dispatchers below also have `DispatcherInfo`'s. If they do, `test_scripted_smoke`
    #  replaces this test for them.
    @pytest.mark.parametrize(
        "dispatcher",
        [
            F.convert_color_space,
            F.convert_image_dtype,
            F.get_dimensions,
            F.get_image_num_channels,
            F.get_image_size,
            F.get_spatial_size,
            F.rgb_to_grayscale,
        ],
        ids=lambda dispatcher: dispatcher.__name__,
    )
    def test_scriptable(self, dispatcher):
        script(dispatcher)


@pytest.mark.parametrize(
    ("alias", "target"),
    [
        pytest.param(alias, target, id=alias.__name__)
        for alias, target in [
            (F.hflip, F.horizontal_flip),
            (F.vflip, F.vertical_flip),
            (F.get_image_num_channels, F.get_num_channels),
            (F.to_pil_image, F.to_image_pil),
            (F.elastic_transform, F.elastic),
        ]
    ],
)
def test_alias(alias, target):
    assert alias is target


# TODO: All correctness checks below this line should be ported to be references on a `KernelInfo` in
#  `prototype_transforms_kernel_infos.py`


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

        height, width = bbox.image_size
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
                [0.0, height, 1.0],
                [width, height, 1.0],
                [width, 0.0, 1.0],
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

            height = int(height - 2 * tr_y)
            width = int(width - 2 * tr_x)

        out_bbox = features.BoundingBox(
            out_bbox,
            format=features.BoundingBoxFormat.XYXY,
            image_size=(height, width),
            dtype=bbox.dtype,
            device=bbox.device,
        )
        return (
            convert_format_bounding_box(
                out_bbox, old_format=features.BoundingBoxFormat.XYXY, new_format=bbox.format, copy=False
            ),
            (height, width),
        )

    image_size = (32, 38)

    for bboxes in make_bounding_boxes(image_size=image_size, extra_dims=((4,),)):
        bboxes_format = bboxes.format
        bboxes_image_size = bboxes.image_size

        output_bboxes, output_image_size = F.rotate_bounding_box(
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
            expected_bbox, expected_image_size = _compute_expected_bbox(bbox, -angle, expand, center_)
            expected_bboxes.append(expected_bbox)
        if len(expected_bboxes) > 1:
            expected_bboxes = torch.stack(expected_bboxes)
        else:
            expected_bboxes = expected_bboxes[0]
        torch.testing.assert_close(output_bboxes, expected_bboxes, atol=1, rtol=0)
        torch.testing.assert_close(output_image_size, expected_image_size, atol=1, rtol=0)


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

    output_boxes, _ = F.rotate_bounding_box(
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

    output_boxes, output_image_size = F.crop_bounding_box(
        in_boxes,
        format,
        top,
        left,
        size[0],
        size[1],
    )

    if format != features.BoundingBoxFormat.XYXY:
        output_boxes = convert_format_bounding_box(output_boxes, format, features.BoundingBoxFormat.XYXY)

    torch.testing.assert_close(output_boxes.tolist(), expected_bboxes)
    torch.testing.assert_close(output_image_size, size)


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

    output_boxes, output_image_size = F.resized_crop_bounding_box(in_boxes, format, top, left, height, width, size)

    if format != features.BoundingBoxFormat.XYXY:
        output_boxes = convert_format_bounding_box(output_boxes, format, features.BoundingBoxFormat.XYXY)

    torch.testing.assert_close(output_boxes, expected_bboxes)
    torch.testing.assert_close(output_image_size, size)


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

    def _compute_expected_image_size(bbox, padding_):
        pad_left, pad_up, pad_right, pad_down = _parse_padding(padding_)
        height, width = bbox.image_size
        return height + pad_up + pad_down, width + pad_left + pad_right

    for bboxes in make_bounding_boxes():
        bboxes = bboxes.to(device)
        bboxes_format = bboxes.format
        bboxes_image_size = bboxes.image_size

        output_boxes, output_image_size = F.pad_bounding_box(
            bboxes, format=bboxes_format, image_size=bboxes_image_size, padding=padding
        )

        torch.testing.assert_close(output_image_size, _compute_expected_image_size(bboxes, padding))

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
            np.array(out_bbox),
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

        output_boxes, output_image_size = F.center_crop_bounding_box(
            bboxes, bboxes_format, bboxes_image_size, output_size
        )

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
        torch.testing.assert_close(output_image_size, output_size)


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


def test_normalize_output_type():
    inpt = torch.rand(1, 3, 32, 32)
    output = F.normalize(inpt, mean=[0.5, 0.5, 0.5], std=[1.0, 1.0, 1.0])
    assert type(output) is torch.Tensor
    torch.testing.assert_close(inpt - 0.5, output)

    inpt = make_image(color_space=features.ColorSpace.RGB)
    output = F.normalize(inpt, mean=[0.5, 0.5, 0.5], std=[1.0, 1.0, 1.0])
    assert type(output) is torch.Tensor
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
