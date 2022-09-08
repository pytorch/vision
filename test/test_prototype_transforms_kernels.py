import itertools
import math

import numpy as np
import PIL.Image
import pytest
import torch.testing
import torchvision.prototype.transforms.functional as F
from common_utils import cpu_and_gpu, needs_cuda
from prototype_common_utils import assert_close
from torch.nn.functional import one_hot
from torchvision.prototype import features
from torchvision.prototype.transforms._utils import is_simple_tensor
from torchvision.prototype.transforms.functional._meta import _COLOR_SPACE_TO_PIL_MODE
from torchvision.prototype.transforms.functional._meta import convert_bounding_box_format
from torchvision.transforms.functional_tensor import _max_value as get_max_value


DEFAULT_LANDSCAPE_IMAGE_SIZE = DEFAULT_IMAGE_SIZE = (7, 33)
DEFAULT_PORTRAIT_IMAGE_SIZE = (31, 9)
DEFAULT_SQUARE_IMAGE_SIZE = (16, 16)


def make_image(
    size=DEFAULT_IMAGE_SIZE,
    *,
    extra_dims=(),
    color_space=features.ColorSpace.RGB,
    device="cpu",
    dtype=torch.float32,
    constant_alpha=True,
):
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
    data = torch.testing.make_tensor(shape, low=0, high=max_value, dtype=dtype, device=device)
    if color_space in {features.ColorSpace.GRAY_ALPHA, features.ColorSpace.RGB_ALPHA} and constant_alpha:
        data[..., -1, :, :] = max_value
    return features.Image(data, color_space=color_space)


def make_images(
    *,
    sizes=(DEFAULT_LANDSCAPE_IMAGE_SIZE, DEFAULT_PORTRAIT_IMAGE_SIZE, DEFAULT_SQUARE_IMAGE_SIZE),
    extra_dims=((), (4,), (2, 3)),
    color_spaces=(
        features.ColorSpace.GRAY,
        features.ColorSpace.GRAY_ALPHA,
        features.ColorSpace.RGB,
        features.ColorSpace.RGB_ALPHA,
    ),
    device="cpu",
    dtypes=(torch.float32, torch.uint8),
):
    for size, color_space, dtype in itertools.product(sizes, color_spaces, dtypes):
        yield make_image(size, color_space=color_space, device=device, dtype=dtype)

    for color_space, dtype, extra_dims_ in itertools.product(color_spaces, dtypes, extra_dims):
        yield make_image(color_space=color_space, extra_dims=extra_dims_, device=device, dtype=dtype)


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


def make_bounding_box(
    *,
    extra_dims=(),
    format,
    image_size=(32, 32),
    device="cpu",
    dtype=torch.int64,
):
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

    return features.BoundingBox(torch.stack(parts, dim=-1).to(dtype).to(device), format=format, image_size=image_size)


def make_bounding_boxes(
    *,
    extra_dims=((4,), (2, 3)),
    formats=(features.BoundingBoxFormat.XYXY, features.BoundingBoxFormat.XYWH, features.BoundingBoxFormat.CXCYWH),
    image_sizes=((32, 32),),
    device="cpu",
    dtypes=(torch.int64, torch.float32),
):
    for format, image_size, dtype in itertools.product(formats, image_sizes, dtypes):
        yield make_bounding_box(format=format, image_size=image_size, device=device, dtype=dtype)

    for format, extra_dims_, dtype in itertools.product(formats, extra_dims, dtypes):
        yield make_bounding_box(format=format, extra_dims=extra_dims_, device=device, dtype=dtype)


def make_label(size=(), *, device="cpu", dtype=torch.int64, categories=("category0", "category1")):
    return features.Label(
        torch.randint(0, len(categories) if categories else 10, size), categories=categories, device=device, dtype=dtype
    )


def make_one_hot_label(*args, **kwargs):
    label = make_label(*args, **kwargs)
    return features.OneHotLabel(one_hot(label, num_classes=len(label.categories)), categories=label.categories)


def make_one_hot_labels(
    *,
    extra_dims=((4,), (2, 3)),
    num_categories=(1, 2, 10),
    device="cpu",
    dtypes=(torch.int64,),
):
    for num_categories_, dtype in itertools.product(num_categories, dtypes):
        yield make_one_hot_label(
            device=device, dtype=dtype, categories=[f"category{idx}" for idx in range(num_categories_)]
        )

    for extra_dims_, dtype in itertools.product(extra_dims, dtypes):
        yield make_one_hot_label(extra_dims=extra_dims_, device=device, dtype=dtype)


def make_segmentation_mask(
    size=DEFAULT_IMAGE_SIZE,
    *,
    extra_dims=(),
    device="cpu",
    dtype=torch.int64,
    num_categories=80,
):
    shape = (*extra_dims, 1, *size)
    data = torch.testing.make_tensor(shape, low=0, high=num_categories, device=device, dtype=dtype)
    return features.SegmentationMask(data)


def make_segmentation_masks(
    *,
    sizes=(DEFAULT_LANDSCAPE_IMAGE_SIZE, DEFAULT_PORTRAIT_IMAGE_SIZE, DEFAULT_SQUARE_IMAGE_SIZE),
    extra_dims=((), (4,), (2, 3)),
    device="cpu",
    dtypes=(torch.long,),
):
    for size, dtype, extra_dims_ in itertools.product(sizes, dtypes, extra_dims):
        yield make_segmentation_mask(size, device=device, dtype=dtype, extra_dims=extra_dims_)


class SampleInput:
    def __init__(self, *args, **kwargs):
        self.args = args
        self.kwargs = kwargs

    def __iter__(self):
        yield self.args
        yield self.kwargs

    def __str__(self):
        def format(value):
            if isinstance(value, torch.Tensor) and (value.ndim > 2 or value.numel() > 5):
                shape = "x".join(str(dim) for dim in value.shape)
                return f"tensor({shape}, dtype={value.dtype}, device={value.device})"
            elif isinstance(value, str):
                return repr(value)
            else:
                return str(value)

        return ", ".join(
            itertools.chain(
                [format(arg) for arg in self.args],
                [f"{param}={format(kwarg)}" for param, kwarg in self.kwargs.items()],
            )
        )


class FunctionalInfo:
    """
    Args:
        functional:
        sample_inputs_fn:
        reference:
        reference_inputs_fn:
        **closeness_kwargs:
    """

    def __init__(
        self,
        functional,
        *,
        sample_inputs_fn,
        reference=None,
        reference_inputs_fn=None,
        **closeness_kwargs,
    ):
        self.functional = functional
        # smoke test that should hit all valid code paths
        self.sample_inputs_fn = sample_inputs_fn
        self.reference = reference
        self.reference_inputs_fn = reference_inputs_fn or sample_inputs_fn
        self.closeness_kwargs = closeness_kwargs

    def __str__(self):
        return self.functional.__name__


def pil_reference_wrapper(pil_functional):
    def wrapper(image_tensor, *other_args, **kwargs):
        if image_tensor.device.type != "cpu":
            raise pytest.UsageError("ADDME")
        elif image_tensor.ndim > 3:
            raise pytest.UsageError("ADDME")

        try:
            data = image_tensor.permute(1, 2, 0)
            if data.shape[-1] == 1:
                data.squeeze_(-1)
            image_pil = PIL.Image.fromarray(
                data.numpy(), mode=_COLOR_SPACE_TO_PIL_MODE.get(image_tensor.color_space, None)
            )
        except Exception as error:
            raise pytest.UsageError("Converting image tensor to PIL failed with the error above.") from error

        return pil_functional(image_pil, *other_args, **kwargs)

    return wrapper


FUNCTIONAL_INFOS = []


def sample_inputs_horizontal_flip_image_tensor(device):
    for image in make_images(sizes=[DEFAULT_IMAGE_SIZE], device=device, dtypes=[torch.float32]):
        yield SampleInput(image)


def reference_inputs_horizontal_flip_image_tensor():
    for image in make_images(extra_dims=[()]):
        yield SampleInput(image)


def sample_inputs_horizontal_flip_bounding_box(device):
    for bounding_box in make_bounding_boxes(device=device):
        yield SampleInput(bounding_box, format=bounding_box.format, image_size=bounding_box.image_size)


FUNCTIONAL_INFOS.extend(
    [
        FunctionalInfo(
            F.horizontal_flip_image_tensor,
            sample_inputs_fn=sample_inputs_horizontal_flip_image_tensor,
            reference=pil_reference_wrapper(F.horizontal_flip_image_pil),
            reference_inputs_fn=reference_inputs_horizontal_flip_image_tensor,
            atol=1e-5,
            rtol=0,
            agg_method="mean",
        ),
        FunctionalInfo(
            F.horizontal_flip_bounding_box,
            sample_inputs_fn=sample_inputs_horizontal_flip_bounding_box,
        ),
    ]
)


def sample_inputs_resize_image_tensor(device):
    for image, interpolation in itertools.product(
        make_images(sizes=[DEFAULT_IMAGE_SIZE], device=device, dtypes=[torch.float32]),
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
            yield SampleInput(image, size=size, interpolation=interpolation)


def reference_inputs_resize_image_tensor():
    for image, interpolation in itertools.product(
        make_images(extra_dims=[()]),
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
            yield SampleInput(image, size=size, interpolation=interpolation)


def sample_inputs_resize_bounding_box(device):
    for bounding_box in make_bounding_boxes(device=device):
        height, width = bounding_box.image_size
        for size in [
            (height, width),
            (int(height * 0.75), int(width * 1.25)),
        ]:
            yield SampleInput(bounding_box, size=size, image_size=bounding_box.image_size)


FUNCTIONAL_INFOS.extend(
    [
        FunctionalInfo(
            F.resize_image_tensor,
            sample_inputs_fn=sample_inputs_resize_image_tensor,
            reference=pil_reference_wrapper(F.resize_image_pil),
            reference_inputs_fn=reference_inputs_resize_image_tensor,
            atol=1e-5,
            rtol=0,
            agg_method="mean",
        ),
        FunctionalInfo(
            F.resize_bounding_box,
            sample_inputs_fn=sample_inputs_resize_bounding_box,
        ),
    ]
)


def sample_inputs_affine_image_tensor(device):
    for image, interpolation_mode, center in itertools.product(
        make_images(
            sizes=[DEFAULT_IMAGE_SIZE],
            extra_dims=[(), (4,)],  # FIXME: the kernel should support multiple batch dimensions!
            device=device,
            dtypes=[torch.float32],
        ),
        [
            F.InterpolationMode.NEAREST,
            F.InterpolationMode.BILINEAR,
        ],
        [None, (0, 0)],
    ):
        for fill in [None, [0.5] * image.shape[-3]]:
            yield SampleInput(
                image,
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
        make_images(extra_dims=[()]),
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


def sample_inputs_affine_bounding_box(device):
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

        bbox_xyxy = convert_bounding_box_format(bbox, old_format=format, new_format=features.BoundingBoxFormat.XYXY)
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
        return convert_bounding_box_format(
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
    for bounding_box, angle, translate, scale, shear, center in itertools.product(
        make_bounding_boxes(extra_dims=[(4,)], image_sizes=[(32, 38)], dtypes=[torch.float32]),
        range(-90, 90, 56),
        range(-10, 10, 8),
        [0.77, 1.0, 1.27],
        range(-15, 15, 8),
        [None, (12, 14)],
    ):
        yield SampleInput(
            bounding_box,
            format=bounding_box.format,
            image_size=bounding_box.image_size,
            angle=angle,
            translate=(translate, translate),
            scale=scale,
            shear=(shear, shear),
            center=center,
        )


FUNCTIONAL_INFOS.extend(
    [
        FunctionalInfo(
            F.affine_image_tensor,
            sample_inputs_fn=sample_inputs_affine_image_tensor,
            reference=pil_reference_wrapper(F.affine_image_pil),
            reference_inputs_fn=reference_inputs_affine_image_tensor,
            atol=1e-5,
            rtol=0,
            agg_method="mean",
        ),
        FunctionalInfo(
            F.affine_bounding_box,
            sample_inputs_fn=sample_inputs_affine_bounding_box,
            reference=reference_affine_bounding_box,
            reference_inputs_fn=reference_inputs_affine_bounding_box,
        ),
    ]
)


class TestCommon:
    @pytest.mark.parametrize("device", cpu_and_gpu())
    @pytest.mark.parametrize("info", FUNCTIONAL_INFOS, ids=str)
    def test_eager_vs_scripted(self, subtests, device, info):
        functional_eager = info.functional
        try:
            functional_scripted = torch.jit.script(functional_eager)
        except Exception as error:
            raise AssertionError("Trying to `torch.jit.script` the functional raised the error above.") from error

        for idx, sample_input in enumerate(info.sample_inputs_fn(device)):
            with subtests.test(f"{idx}, ({sample_input})"):
                args, kwargs = sample_input

                actual = functional_scripted(*args, **kwargs)
                expected = functional_eager(*args, **kwargs)

                assert_close(actual, expected, **info.closeness_kwargs)

    @pytest.mark.parametrize("device", cpu_and_gpu())
    @pytest.mark.parametrize("info", FUNCTIONAL_INFOS, ids=str)
    def test_batched_vs_single(self, subtests, device, info):
        for idx, sample_input in enumerate(info.sample_inputs_fn(device)):
            with subtests.test(f"{idx}, ({sample_input})"):
                (batched_input, *other_args), kwargs = sample_input

                feature_type = features.Image if is_simple_tensor(batched_input) else type(batched_input)
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
                    pytest.skip("REMOVEME")

                actual = info.functional(batched_input, *other_args, **kwargs).unbind()
                expected = [
                    info.functional(single_input, *other_args, **kwargs) for single_input in batched_input.unbind()
                ]

                assert_close(actual, expected, **info.closeness_kwargs)

    @pytest.mark.parametrize("device", cpu_and_gpu())
    @pytest.mark.parametrize("info", FUNCTIONAL_INFOS, ids=str)
    def test_no_inplace(self, subtests, device, info):
        for idx, sample_input in enumerate(info.sample_inputs_fn(device)):
            with subtests.test(f"{idx}, ({sample_input})"):
                (input, *other_args), kwargs = sample_input
                input_version = input._version

                output = info.functional(input, *other_args, **kwargs)

                assert output is not input or output._version == input_version

    @needs_cuda
    @pytest.mark.parametrize("info", FUNCTIONAL_INFOS, ids=str)
    def test_cpu_vs_cuda(self, subtests, info):
        for idx, sample_input in enumerate(info.sample_inputs_fn("cpu")):
            with subtests.test(f"{idx}, ({sample_input})"):
                (input_cpu, *other_args), kwargs = sample_input
                input_cuda = input_cpu.to("cuda")

                output_cpu = info.functional(input_cpu, *other_args, **kwargs)
                output_cuda = info.functional(input_cuda, *other_args, **kwargs)

                assert_close(output_cuda, output_cpu, check_device=False)

    @pytest.mark.parametrize("info", [info for info in FUNCTIONAL_INFOS if info.reference], ids=str)
    def test_against_reference(self, subtests, info):
        for idx, sample_input in enumerate(info.reference_inputs_fn()):
            with subtests.test(f"{idx}, ({sample_input})"):
                args, kwargs = sample_input

                actual = info.functional(*args, **kwargs)
                expected = info.reference(*args, **kwargs)

                assert_close(actual, expected, **info.closeness_kwargs)


class TestAffine:
    @pytest.mark.parametrize("device", cpu_and_gpu())
    def test_bounding_box_against_fixed_reference(self, device):
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

        assert_close(output_boxes.tolist(), expected_bboxes)

    @pytest.mark.parametrize("device", cpu_and_gpu())
    def test_segmentation_mask_against_fixed_reference(self, device):
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
