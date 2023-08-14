from copy import deepcopy

import pytest
import torch
from common_utils import assert_equal
from PIL import Image

from torchvision import datapoints
from common_utils import (
    make_bounding_box,
    make_detection_mask,
    make_image,
    make_image_tensor,
    make_segmentation_mask,
    make_video,
)


@pytest.fixture(autouse=True)
def preserve_default_wrapping_behaviour():
    yield
    datapoints.set_return_type("Tensor")


@pytest.mark.parametrize("data", [torch.rand(3, 32, 32), Image.new("RGB", (32, 32), color=123)])
def test_image_instance(data):
    image = datapoints.Image(data)
    assert isinstance(image, torch.Tensor)
    assert image.ndim == 3 and image.shape[0] == 3


@pytest.mark.parametrize("data", [torch.randint(0, 10, size=(1, 32, 32)), Image.new("L", (32, 32), color=2)])
def test_mask_instance(data):
    mask = datapoints.Mask(data)
    assert isinstance(mask, torch.Tensor)
    assert mask.ndim == 3 and mask.shape[0] == 1


@pytest.mark.parametrize("data", [torch.randint(0, 32, size=(5, 4)), [[0, 0, 5, 5], [2, 2, 7, 7]], [1, 2, 3, 4]])
@pytest.mark.parametrize(
    "format", ["XYXY", "CXCYWH", datapoints.BoundingBoxFormat.XYXY, datapoints.BoundingBoxFormat.XYWH]
)
def test_bbox_instance(data, format):
    bboxes = datapoints.BoundingBoxes(data, format=format, canvas_size=(32, 32))
    assert isinstance(bboxes, torch.Tensor)
    assert bboxes.ndim == 2 and bboxes.shape[1] == 4
    if isinstance(format, str):
        format = datapoints.BoundingBoxFormat[(format.upper())]
    assert bboxes.format == format


def test_bbox_dim_error():
    data_3d = [[[1, 2, 3, 4]]]
    with pytest.raises(ValueError, match="Expected a 1D or 2D tensor, got 3D"):
        datapoints.BoundingBoxes(data_3d, format="XYXY", canvas_size=(32, 32))


@pytest.mark.parametrize(
    ("data", "input_requires_grad", "expected_requires_grad"),
    [
        ([[[0.0, 1.0], [0.0, 1.0]]], None, False),
        ([[[0.0, 1.0], [0.0, 1.0]]], False, False),
        ([[[0.0, 1.0], [0.0, 1.0]]], True, True),
        (torch.rand(3, 16, 16, requires_grad=False), None, False),
        (torch.rand(3, 16, 16, requires_grad=False), False, False),
        (torch.rand(3, 16, 16, requires_grad=False), True, True),
        (torch.rand(3, 16, 16, requires_grad=True), None, True),
        (torch.rand(3, 16, 16, requires_grad=True), False, False),
        (torch.rand(3, 16, 16, requires_grad=True), True, True),
    ],
)
def test_new_requires_grad(data, input_requires_grad, expected_requires_grad):
    datapoint = datapoints.Image(data, requires_grad=input_requires_grad)
    assert datapoint.requires_grad is expected_requires_grad


def test_isinstance():
    assert isinstance(datapoints.Image(torch.rand(3, 16, 16)), torch.Tensor)


def test_wrapping_no_copy():
    tensor = torch.rand(3, 16, 16)
    image = datapoints.Image(tensor)

    assert image.data_ptr() == tensor.data_ptr()


def test_to_wrapping():
    image = datapoints.Image(torch.rand(3, 16, 16))

    image_to = image.to(torch.float64)

    assert type(image_to) is datapoints.Image
    assert image_to.dtype is torch.float64


@pytest.mark.parametrize("return_type", ["Tensor", "datapoint"])
def test_to_datapoint_reference(return_type):
    tensor = torch.rand((3, 16, 16), dtype=torch.float64)
    image = datapoints.Image(tensor)

    with datapoints.set_return_type(return_type):
        tensor_to = tensor.to(image)

    assert type(tensor_to) is (datapoints.Image if return_type == "datapoint" else torch.Tensor)
    assert tensor_to.dtype is torch.float64


@pytest.mark.parametrize("return_type", ["Tensor", "datapoint"])
def test_clone_wrapping(return_type):
    image = datapoints.Image(torch.rand(3, 16, 16))

    with datapoints.set_return_type(return_type):
        image_clone = image.clone()

    assert type(image_clone) is datapoints.Image
    assert image_clone.data_ptr() != image.data_ptr()


@pytest.mark.parametrize("return_type", ["Tensor", "datapoint"])
def test_requires_grad__wrapping(return_type):
    image = datapoints.Image(torch.rand(3, 16, 16))

    assert not image.requires_grad

    with datapoints.set_return_type(return_type):
        image_requires_grad = image.requires_grad_(True)

    assert type(image_requires_grad) is datapoints.Image
    assert image.requires_grad
    assert image_requires_grad.requires_grad


@pytest.mark.parametrize("return_type", ["Tensor", "datapoint"])
def test_detach_wrapping(return_type):
    image = datapoints.Image(torch.rand(3, 16, 16), requires_grad=True)

    with datapoints.set_return_type(return_type):
        image_detached = image.detach()

    assert type(image_detached) is datapoints.Image


@pytest.mark.parametrize("return_type", ["Tensor", "datapoint"])
def test_force_subclass_with_metadata(return_type):
    # Sanity checks for the ops in _FORCE_TORCHFUNCTION_SUBCLASS and datapoints with metadata
    format, canvas_size = "XYXY", (32, 32)
    bbox = datapoints.BoundingBoxes([[0, 0, 5, 5], [2, 2, 7, 7]], format=format, canvas_size=canvas_size)

    datapoints.set_return_type(return_type)
    bbox = bbox.clone()
    if return_type == "datapoint":
        assert bbox.format, bbox.canvas_size == (format, canvas_size)

    bbox = bbox.to(torch.float64)
    if return_type == "datapoint":
        assert bbox.format, bbox.canvas_size == (format, canvas_size)

    bbox = bbox.detach()
    if return_type == "datapoint":
        assert bbox.format, bbox.canvas_size == (format, canvas_size)

    assert not bbox.requires_grad
    bbox.requires_grad_(True)
    if return_type == "datapoint":
        assert bbox.format, bbox.canvas_size == (format, canvas_size)
        assert bbox.requires_grad


@pytest.mark.parametrize("return_type", ["Tensor", "datapoint"])
def test_other_op_no_wrapping(return_type):
    image = datapoints.Image(torch.rand(3, 16, 16))

    with datapoints.set_return_type(return_type):
        # any operation besides the ones listed in _FORCE_TORCHFUNCTION_SUBCLASS will do here
        output = image * 2

    assert type(output) is (datapoints.Image if return_type == "datapoint" else torch.Tensor)


@pytest.mark.parametrize(
    "op",
    [
        lambda t: t.numpy(),
        lambda t: t.tolist(),
        lambda t: t.max(dim=-1),
    ],
)
def test_no_tensor_output_op_no_wrapping(op):
    image = datapoints.Image(torch.rand(3, 16, 16))

    output = op(image)

    assert type(output) is not datapoints.Image


@pytest.mark.parametrize("return_type", ["Tensor", "datapoint"])
def test_inplace_op_no_wrapping(return_type):
    image = datapoints.Image(torch.rand(3, 16, 16))

    with datapoints.set_return_type(return_type):
        output = image.add_(0)

    assert type(output) is (datapoints.Image if return_type == "datapoint" else torch.Tensor)
    assert type(image) is datapoints.Image


def test_wrap_like():
    image = datapoints.Image(torch.rand(3, 16, 16))

    # any operation besides the ones listed in _FORCE_TORCHFUNCTION_SUBCLASS will do here
    output = image * 2

    image_new = datapoints.Image.wrap_like(image, output)

    assert type(image_new) is datapoints.Image
    assert image_new.data_ptr() == output.data_ptr()


@pytest.mark.parametrize(
    "datapoint",
    [
        datapoints.Image(torch.rand(3, 16, 16)),
        datapoints.Video(torch.rand(2, 3, 16, 16)),
        datapoints.BoundingBoxes([0.0, 1.0, 2.0, 3.0], format=datapoints.BoundingBoxFormat.XYXY, canvas_size=(10, 10)),
        datapoints.Mask(torch.randint(0, 256, (16, 16), dtype=torch.uint8)),
    ],
)
@pytest.mark.parametrize("requires_grad", [False, True])
def test_deepcopy(datapoint, requires_grad):
    if requires_grad and not datapoint.dtype.is_floating_point:
        return

    datapoint.requires_grad_(requires_grad)

    datapoint_deepcopied = deepcopy(datapoint)

    assert datapoint_deepcopied is not datapoint
    assert datapoint_deepcopied.data_ptr() != datapoint.data_ptr()
    assert_equal(datapoint_deepcopied, datapoint)

    assert type(datapoint_deepcopied) is type(datapoint)
    assert datapoint_deepcopied.requires_grad is requires_grad


@pytest.mark.parametrize("return_type", ["Tensor", "datapoint"])
def test_operations(return_type):
    datapoints.set_return_type(return_type)

    img = datapoints.Image(torch.rand(3, 10, 10))
    t = torch.rand(3, 10, 10)
    mask = datapoints.Mask(torch.rand(1, 10, 10))

    for out in (
        [
            img + t,
            t + img,
            img * t,
            t * img,
            img + 3,
            3 + img,
            img * 3,
            3 * img,
            img + img,
            img.sum(),
            img.reshape(-1),
            img.float(),
            torch.stack([img, img]),
        ]
        + list(torch.chunk(img, 2))
        + list(torch.unbind(img))
    ):
        assert type(out) is (datapoints.Image if return_type == "datapoint" else torch.Tensor)

    for out in (
        [
            mask + t,
            t + mask,
            mask * t,
            t * mask,
            mask + 3,
            3 + mask,
            mask * 3,
            3 * mask,
            mask + mask,
            mask.sum(),
            mask.reshape(-1),
            mask.float(),
            torch.stack([mask, mask]),
        ]
        + list(torch.chunk(mask, 2))
        + list(torch.unbind(mask))
    ):
        assert type(out) is (datapoints.Mask if return_type == "datapoint" else torch.Tensor)

    with pytest.raises(TypeError, match="unsupported operand type"):
        img + mask

    with pytest.raises(TypeError, match="unsupported operand type"):
        img * mask

    bboxes = datapoints.BoundingBoxes(
        [[17, 16, 344, 495], [0, 10, 0, 10]], format=datapoints.BoundingBoxFormat.XYXY, canvas_size=(1000, 1000)
    )
    t = torch.rand(2, 4)

    for out in (
        [
            bboxes + t,
            t + bboxes,
            bboxes * t,
            t * bboxes,
            bboxes + 3,
            3 + bboxes,
            bboxes * 3,
            3 * bboxes,
            bboxes + bboxes,
            bboxes.sum(),
            bboxes.reshape(-1),
            bboxes.float(),
            torch.stack([bboxes, bboxes]),
        ]
        + list(torch.chunk(bboxes, 2))
        + list(torch.unbind(bboxes))
    ):
        if return_type == "Tensor":
            assert type(out) is torch.Tensor
        else:
            assert isinstance(out, datapoints.BoundingBoxes)
            assert hasattr(out, "format")
            assert hasattr(out, "canvas_size")
